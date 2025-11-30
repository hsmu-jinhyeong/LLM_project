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
    # Heuristic cuisine/occasion boosting for search terms
    lower = user_input.lower()
    cuisine_boost = []
    if any(k in lower for k in ["양식", "서양", "western", "date", "데이트"]):
        cuisine_boost += ["파스타", "스테이크", "샐러드", "피자", "그라탕", "리소토"]
    if any(k in lower for k in ["한식", "korean"]):
        cuisine_boost += ["찌개", "비빔밥", "불고기", "전", "국", "나물"]
    if any(k in lower for k in ["일식", "japanese"]):
        cuisine_boost += ["스시", "우동", "라멘", "덴푸라", "가츠"]
    if any(k in lower for k in ["중식", "chinese"]):
        cuisine_boost += ["짜장", "짬뽕", "마라", "탕수육", "볶음"]
    boost_str = (" " + " ".join(set(cuisine_boost))) if cuisine_boost else ""
    search_query = f"{user_input} ({sentiment['description']} 관련){boost_str}"
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
