import os
import json
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# sentiment module now in menu_bot package
try:
    from menu_bot.sentiment_module import get_user_sentiment  # type: ignore
except ImportError:
    get_user_sentiment = None  # Fallback handled in recommend functions

_client = None
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("recommendation_engine")

def _client_or_raise() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

# -------------------- Formatting Helpers --------------------

def get_current_time() -> str:
    return datetime.now().strftime("%p %I시 %M분").replace("AM", "오전").replace("PM", "오후")

def format_recipe_context(recipes: List[Dict[str, str]]) -> str:
    if not recipes:
        return "(검색된 레시피 없음)"
    return "\n---\n".join([
        f"제목: {r['title']}\n내용: {r['content'][:160]}..." for r in recipes
    ])

def create_recommendation_prompt(user_input: str, sentiment_data: Dict[str, Any], recipes: List[Dict[str, str]], current_time: str) -> str:
    recipe_context = format_recipe_context(recipes)
    return f"""당신은 신뢰 기반 한국어 레시피 추천 전문가입니다. 검색된 레시피에 없는 재료나 조리법을 임의로 발명하지 마십시오.
아래 정보를 고려하여 JSON 형식으로만 답변하세요.

<CONTEXT>
시간: {current_time}
감성: {sentiment_data.get('description')} (점수: {sentiment_data.get('score', 0):.2f})
검색된 레시피:
{recipe_context}
</CONTEXT>

출력 JSON 스키마:
{{
    "recommendations": [
        {{"title": "메뉴명", "reason": "180자 이하 이유", "match_factors": ["감성","재료","시간"]}}
    ],
    "constraints_checked": {{
        "sentiment_used": true,
        "search_results": {len(recipes)},
        "fallback_used": {str(not bool(recipes)).lower()}
    }},
    "sentiment": "{sentiment_data.get('description')}",
    "timestamp": "{current_time}"
}}

규칙:
- 최대 2개 추천.
- 두 추천은 주재료/조리법이 과도하게 유사하지 않도록 다양성을 확보하세요 (예: 같은 우동 볶음 2개는 피함).
- reason 첫 문장에 감성을 자연스럽게 반영.
- "단백질"이 사용자 입력에 포함되거나 운동 관련 맥락이면, 각 추천의 reason에 단백질 근거(예: 닭가슴살/두부/달걀/콩/참치)를 명시하세요.
- 시간 제약이 언급되면 reason에 대략적인 조리 시간(예: ~15분)을 언급하세요.
- "데이트" 또는 "분위기"가 언급되면, 플레이팅/분위기/감성 요소를 이유에 반영하고, 요청된 음식 유형(예: 양식/한식/일식/중식)을 최대한 존중하세요.
- "모임", "단체", "파티", "회식", "동창회" 등 다인이 함께 먹는 상황이면, 1인용 도시락/개별 포장 메뉴보다 공유/파티형 메뉴(플래터, 핑거푸드, 대접용 파스타·샐러드, 한상 차림)를 우선 추천하세요. 도시락은 피하거나 보조 옵션으로만 언급하세요.
- 레시피가 없으면 영양/소화 측면에서 일반적인 안전 식단 1개만 추천하고 fallback_used=true 유지.
- JSON 외 다른 텍스트 출력 금지.
"""

# -------------------- Tools Definition (Function Calling) --------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": "벡터 검색 기반 레시피를 쿼리에 따라 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "검색할 쿼리 텍스트"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환 레시피 개수",
                        "default": 3
                    }
                },
                "required": ["query_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_sentiment",
            "description": "사용자 텍스트 감성 분석 (긍정 / 부정 / 중립).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "분석할 텍스트"}
                },
                "required": ["text"]
            }
        }
    }
]

# -------------------- Recommendation Core --------------------

def _chat_with_retry(messages: List[Dict[str, str]], model: str = "gpt-4o", max_retries: int = 4):
    client = _client_or_raise()
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=model, messages=messages)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Chat completion failed: {e}")
                raise
            sleep = (2 ** attempt) + random.uniform(0, 0.4)
            logger.warning(f"Chat retry {attempt+1}: {e} (sleep {sleep:.1f}s)")
            time.sleep(sleep)

def sanitize_user_input(user_input: str) -> str:
    if len(user_input) > 800:
        return user_input[:800] + " ...(truncated)"
    return user_input

def recommend_with_rag(user_input: str, search_fn, top_k: int = 3):
    """RAG 기반 단일 호출 추천.

    Args:
        user_input: 사용자 입력.
        search_fn: (query_text: str, top_k: int) -> List[Dict[str,str]] 검색 함수.
        top_k: 검색 결과 개수.
    Returns:
        GPT 모델 추천 텍스트.
    """
    user_input = sanitize_user_input(user_input)
    # Sentiment
    if get_user_sentiment:
        sentiment = get_user_sentiment(user_input)
    else:
        sentiment = {"label": "NEUTRAL", "score": 0.0, "description": "중립"}
    current_time = get_current_time()
    # Build intent-aware boost tokens to stabilize retrieval
    def build_boost_tokens(text: str) -> list[str]:
        t = text.lower()
        tokens: list[str] = []
        # Occasion & group size
        group_keys = ["모임", "단체", "파티", "회식", "동창회", "친목", "소모임", "포트럭", "피크닉", "홈파티"]
        if any(k in t for k in group_keys):
            tokens += ["플래터", "핑거푸드", "한상", "대용량", "공유", "샤링", "파티", "대접용 파스타", "대접용 샐러드"]
            tokens += ["도시락 제외", "개별 포장 제외"]
        # Cuisine anchors
        if any(k in t for k in ["양식", "서양", "western", "date", "데이트"]):
            tokens += ["파스타", "스테이크", "리소토", "라자냐", "브루스케타", "샐러드", "바게트", "플래터", "핑거푸드", "플레이팅", "로맨틱"]
        if any(k in t for k in ["한식", "korean"]):
            tokens += ["한상", "비빔", "구이", "찜", "전", "나물", "국", "탕"]
        if any(k in t for k in ["일식", "japanese"]):
            tokens += ["초밥", "사시미", "덮밥", "우동", "야키소바", "오니기리", "가라아게"]
        if any(k in t for k in ["중식", "chinese"]):
            tokens += ["딤섬", "마라", "볶음", "탕", "면", "교자", "훠궈"]
        # Diet & allergy
        if any(k in t for k in ["비건", "vegan"]):
            tokens += ["두부", "콩", "병아리콩", "렌틸", "버섯", "식물성"]
        if any(k in t for k in ["저염", "low salt", "low_salt"]):
            tokens += ["저염", "무염", "염도 낮음", "싱겁게"]
        if any(k in t for k in ["알레르기", "allergy", "allergies"]):
            tokens += ["새우 제외", "땅콩 제외", "우유 제외", "글루텐 프리", "난류 제외"]
        # Nutrition & goals
        if any(k in t for k in ["단백질", "protein", "헬스", "운동"]):
            tokens += ["닭가슴살", "두부", "달걀", "콩", "참치", "그릭 요거트"]
        if any(k in t for k in ["다이어트", "diet", "저칼로리", "라이트"]):
            tokens += ["저칼로리", "라이트", "샐러드", "구이", "에어프라이어"]
        # Comfort/soothing foods
        if any(k in t for k in ["속", "안 좋아", "편안", "위로", "따뜻", "부드럽", "자극", "comfort"]):
            tokens += ["죽", "미음", "스프", "따뜻한", "부드러운", "자극 적음"]
        # Desserts / mood uplift
        if any(k in t for k in ["디저트", "달콤", "기분", "우울", "uplift", "sweet", "디저트 추천"]):
            tokens += ["디저트", "케이크", "초코", "타르트", "쿠키", "무스"]
        # Time & effort
        if any(k in t for k in ["빨리", "빠른", "간단", "10분", "15분", "즉석"]):
            tokens += ["10분", "15분", "간단", "즉석", "스피드"]
        if any(k in t for k in ["정성", "코스", "오븐", "숙성", "슬로우"]):
            tokens += ["저온", "숙성", "오븐", "장시간", "슬로우"]
        # Flavor & texture
        if any(k in t for k in ["담백", "달콤", "매운", "새콤", "감칠", "고소"]):
            tokens += ["담백", "달콤", "매운", "새콤", "감칠맛", "고소"]
        if any(k in t for k in ["바삭", "촉촉", "부드러운"]):
            tokens += ["바삭", "촉촉", "부드러운"]
        # Weather/season
        if any(k in t for k in ["겨울", "추운", "스튜", "국물"]):
            tokens += ["따뜻한", "국", "탕", "스튜"]
        if any(k in t for k in ["여름", "더운", "상큼"]):
            tokens += ["상큼한", "차가운", "샐러드", "냉파스타", "냉모밀"]
        if any(k in t for k in ["비", "비오는", "장마"]):
            tokens += ["따뜻한", "국물", "전", "바삭"]
        # Deduplicate and cap length
        seen = set()
        capped = []
        for w in tokens:
            if w not in seen:
                seen.add(w)
                capped.append(w)
            if len(capped) >= 12:
                break
        return capped

    boost_tokens = build_boost_tokens(user_input)
    search_query = user_input + (" " + " ".join(boost_tokens) if boost_tokens else "")
    retrieved = search_fn(search_query, top_k=top_k)
    if not retrieved:
        retrieved = [{"title": "기본 위로 음식", "content": "속이 편안한 죽 / 미음 / 계란찜 등 자극 적은 메뉴"}]
    prompt = create_recommendation_prompt(user_input, sentiment, retrieved, current_time)
    response = _chat_with_retry([
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ])
    return response.choices[0].message.content

def recommend_with_function_calling(user_input: str, messages: List[Dict], search_fn, sentiment_fn=None):
    """Function Calling 기반 고급 추천.

    Args:
        user_input: 사용자 입력.
        messages: 초기 메시지 리스트 (system 포함 가능).
        search_fn: 검색 함수.
        sentiment_fn: 감성 분석 함수 (기본: get_user_sentiment).
    Returns:
        최종 GPT 응답 텍스트.
    """
    user_input = sanitize_user_input(user_input)
    client = _client_or_raise()
    if sentiment_fn is None:
        sentiment_fn = get_user_sentiment
    messages = list(messages) + [{"role": "user", "content": user_input}]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools, tool_choice="auto")
    first = response.choices[0].message
    if not first.tool_calls:
        return first.content

    for call in first.tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments)
        if name == 'search_recipes':
            tool_result = search_fn(args['query_text'], top_k=args.get('top_k', 3))
        elif name == 'get_user_sentiment' and sentiment_fn:
            tool_result = sentiment_fn(args['text'])
        else:
            tool_result = {"error": "unknown tool"}
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(tool_result, ensure_ascii=False)})

    final = _chat_with_retry(messages, model="gpt-4o")
    return final.choices[0].message.content
