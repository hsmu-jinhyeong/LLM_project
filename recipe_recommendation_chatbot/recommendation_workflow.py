"""LangGraph-based stateful recommendation workflow.

This module implements a graph-based workflow for menu recommendations
with state management, multi-step reasoning, and conditional routing.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Sequence, Literal, NotRequired
from datetime import datetime
import operator
import json
import logging

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

logger = logging.getLogger("recommendation_workflow")


# -------------------- State Definition --------------------

class RecommendationState(TypedDict):
    """State schema for recommendation workflow.
    
    Tracks conversation history, user input, search results,
    sentiment data, user profile, and final recommendations.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
    user_profile: NotRequired[dict]  # 사이드바 프로필 정보 (선택적)
    sentiment_data: dict
    search_results: list
    recommendations: list
    current_time: str
    iteration_count: int


# -------------------- Node Functions --------------------

def analyze_sentiment_node(state: RecommendationState) -> RecommendationState:
    """Analyze user sentiment from input.
    
    Args:
        state: Current workflow state.
    
    Returns:
        Updated state with sentiment_data.
    """
    try:
        from sentiment_module import get_user_sentiment
        sentiment = get_user_sentiment(state["user_input"])
        logger.info(f"Sentiment analysis: {sentiment.get('description')}")
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}, using neutral")
        sentiment = {"label": "NEUTRAL", "score": 0.5, "description": "중립적인 기분"}
    
    return {
        **state,
        "sentiment_data": sentiment,
        "messages": state["messages"] + [
            SystemMessage(content=f"감성 분석 완료: {sentiment.get('description')} (점수: {sentiment.get('score', 0):.2f})")
        ],
    }


def search_recipes_node(state: RecommendationState, retriever) -> RecommendationState:
    """Search for relevant recipes.
    
    Args:
        state: Current workflow state.
        retriever: FAISSRecipeRetriever instance.
    
    Returns:
        Updated state with search_results.
    """
    query = state["user_input"]
    
    # Enhance query with sentiment if available
    if state.get("sentiment_data"):
        sentiment_desc = state["sentiment_data"].get("description", "")
        if "긍정" in sentiment_desc or "부정" in sentiment_desc:
            query = f"{query} ({sentiment_desc})"
    
    # 프롬프트에서 비건/채식 감지 (검색 전에 미리 확인)
    user_input_lower = state.get('user_input', '').lower()
    prompt_vegan = any(k in user_input_lower for k in ['비건', 'vegan', '채식'])
    profile = state.get('user_profile', {})
    is_vegan = profile.get('diet') == 'vegan' or prompt_vegan
    
    # 쿼리 개선: 비건이면 식물성 단백질 키워드 추가
    if is_vegan and any(k in user_input_lower for k in ['단백질', '영양']):
        query = f"{query} 두부 콩 견과류"
        logger.info(f"[SEARCH] 비건 단백질 쿼리 강화: {query}")
    
    # 비건/채식이면 더 많이 검색 (필터링으로 많이 제거되므로)
    search_k = 50 if is_vegan else 10
    
    # Use vectorstore directly for k control
    docs = retriever.vectorstore.similarity_search(query, k=search_k)
    
    results = [
        {
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content[:200],
            "category": doc.metadata.get("category", ""),
            "full_content": doc.page_content  # 필터링용 전체 내용
        }
        for doc in docs
    ]
    
    logger.info(f"Retrieved {len(results)} recipes (before filtering)")
    
    # ==================== 프로필 + 프롬프트 통합 감지 ====================
    # 프롬프트 채식 키워드 감지
    prompt_vegetarian = any(k in user_input_lower for k in ['채식'])
    is_vegetarian = profile.get('diet') == 'vegetarian' or (prompt_vegetarian and not prompt_vegan)
    
    filtered_results = results
    
    # 식단 제약 필터링
    if is_vegan or is_vegetarian:
        # 식단 제약 필터링 (확장된 제외 목록)
        vegan_exclude = [
            # 육류
            '고기', '쇠고기', '돼지고기', '닭고기', '닭', '양고기', '오리고기', '삼격살', '목살', '항정살', '등심', '안심', '갈비', '차돌', '사태', '양지', '우삼격',
            # 달걀/유제품
            '달걀', '계란', '치즈', '버터', '우유', '크림', '요구르트', '생크림', '노른자', '흰자',
            # 해산물 (전체)
            '생선', '해산물', '다슬기', '굴', '조개', '새우', '게', '오징어', '낙지', '문어', '주꾸미',
            '고등어', '갈치', '꽁치', '참치', '연어', '광어', '우럭', '조기', '멸치', '북어', '명태', '대구', '동태',
            '조갯살', '홍합', '바지락', '가리비', '전복', '소라', '해물', '어묵', '오덱', '골뱅이',
            # 특수 동물성
            '선지', '곱창', '막창', '명란', '창란', '알탕', '젓갈', '까나리', '액젓',
            # 가공육
            '베이컨', '소시지', '햄', '스팸', '육포', '베컨'
        ]
        vegetarian_exclude = [
            # 육류
            '고기', '쇠고기', '돼지고기', '닭고기', '양고기', '오리고기', '삼겹살', '목살', '항정살', '등심', '안심', '갈비', '차돌', '사태', '양지',
            # 해산물
            '생선', '해산물', '다슬기', '굴', '조개', '새우', '게', '오징어', '낙지', '문어', '주꾸미',
            '고등어', '갈치', '꽁치', '참치', '연어', '광어', '우럭', '조기', '멸치', '북어', '명태', '대구',
            '어묵', '오뎅', '해물',
            # 특수 동물성
            '선지', '곱창', '막창', '명란', '창란', '알탕', '젓갈', '까나리', '액젓',
            # 가공육
            '베이컨', '소시지', '햄', '스팸', '육포'
        ]
        
        diet_exclude = []
        if is_vegan:
            diet_exclude = vegan_exclude
            source = "프롬프트" if prompt_vegan else "사이드바"
            logger.info(f"[FILTER] 비건 필터링 적용 ({source}): {len(vegan_exclude)}개 재료 제외")
        elif is_vegetarian:
            diet_exclude = vegetarian_exclude
            source = "프롬프트" if prompt_vegetarian else "사이드바"
            logger.info(f"[FILTER] 채식 필터링 적용 ({source}): {len(vegetarian_exclude)}개 재료 제외")
        
        if diet_exclude:
            before_count = len(filtered_results)
            filtered_results = [
                r for r in filtered_results
                if not any(excluded in r['title'] + r.get('full_content', '') for excluded in diet_exclude)
            ]
            removed = before_count - len(filtered_results)
            logger.info(f"[FILTER] 식단 제약 후: {before_count} -> {len(filtered_results)}개 ({removed}개 제거)")
    
    # 알레르기 필터링 (프로필만 사용)
    if profile and profile.get('allergies'):
        before_count = len(filtered_results)
        filtered_results = [
            r for r in filtered_results
            if not any(allergy.lower() in r['title'].lower() + r.get('full_content', '').lower() for allergy in profile['allergies'])
        ]
        removed = before_count - len(filtered_results)
        logger.info(f"[FILTER] 알레르기 필터 후: {before_count} -> {len(filtered_results)}개 ({removed}개 제거)")
    
    # 최종 결과 (최대 5개)
    final_results = filtered_results[:5]
    logger.info(f"Retrieved {len(final_results)} recipes (after filtering)")
    
    return {
        **state,
        "search_results": final_results,
        "messages": state["messages"] + [
            SystemMessage(content=f"레시피 검색 완료: {len(final_results)}개 발견")
        ],
    }


def generate_recommendations_node(state: RecommendationState, llm: ChatOpenAI) -> RecommendationState:
    """Generate final recommendations using LLM.
    
    Args:
        state: Current workflow state.
        llm: ChatOpenAI instance.
    
    Returns:
        Updated state with recommendations.
    """
    # Build context from search results
    if state["search_results"]:
        context = "\n---\n".join([
            f"제목: {r['title']}\n내용: {r['content']}"
            for r in state["search_results"]
        ])
    else:
        context = "검색된 레시피가 없습니다. 일반적인 추천을 제공하세요."
    
    sentiment_desc = state.get("sentiment_data", {}).get("description", "중립적인 기분")
    sentiment_score = state.get("sentiment_data", {}).get("score", 0.5)
    
    # ==================== 프로필 + 프롬프트 통합 분석 ====================
    user_input_lower = state['user_input'].lower()
    profile = state.get('user_profile', {})
    
    dietary_restrictions = []
    allergies_list = []
    preferences = []
    requirements = []
    
    # 1. 사이드바 프로필 정보 (최우선)
    if profile.get('diet'):
        if profile['diet'] == 'vegan':
            dietary_restrictions.append("🚫 비건 식단 (사이드바): 고기, 닭고기, 돼지고기, 쇠고기, 달걀, 우유, 치즈, 버터, 생선, 해산물 절대 금지")
        elif profile['diet'] == 'vegetarian':
            dietary_restrictions.append("🚫 채식 (사이드바): 고기, 닭고기, 돼지고기, 쇠고기, 생선, 해산물 절대 금지")
        elif profile['diet'] == 'low_sodium':
            dietary_restrictions.append("🚫 저염식 (사이드바): 소금, 간장, 액젓, 젓갈 최소화")
    
    if profile.get('allergies'):
        allergy_items = ', '.join(profile['allergies'])
        allergies_list.append(f"⚠️ 알레르기 (사이드바): {allergy_items} 포함 레시피 절대 금지")
    
    if profile.get('preferred_flavors'):
        pref_items = ', '.join(profile['preferred_flavors'])
        preferences.append(f"✅ 선호 맛 (사이드바): {pref_items} 우선")
    
    if profile.get('disliked_flavors'):
        dislike_items = ', '.join(profile['disliked_flavors'])
        preferences.append(f"❌ 비선호 맛 (사이드바): {dislike_items} 제외")
    
    # 2. 프롬프트 키워드 감지 (사이드바에 없으면 추가)
    # 식단 제약
    if any(k in user_input_lower for k in ['비건', 'vegan', '채식']) and not profile.get('diet'):
        dietary_restrictions.append("🚫 비건/채식 (입력): 고기, 닭고기, 돼지고기, 쇠고기, 달걀, 우유, 치즈, 버터, 생선, 해산물 절대 금지")
    if any(k in user_input_lower for k in ['저염', '소금금지', '싱겁게']) and profile.get('diet') != 'low_sodium':
        dietary_restrictions.append("🚫 저염식 (입력): 소금, 간장, 액젓, 젓갈 최소화")
    
    # 알레르기 키워드
    allergy_keywords = {
        '땅콩': ['땅콩', '피넛'],
        '우유': ['우유', '유제품', '유당불내'],
        '달걀': ['달걀', '계란'],
        '갑각류': ['새우', '게', '갑각류'],
        '견과류': ['호두', '아몬드', '견과류']
    }
    for allergy, keywords in allergy_keywords.items():
        if any(k in user_input_lower for k in keywords) and (not profile.get('allergies') or allergy not in profile['allergies']):
            allergies_list.append(f"⚠️ 알레르기 (입력): {allergy} 포함 레시피 절대 금지")
    
    # 선호/비선호 맛
    if any(k in user_input_lower for k in ['매운', '매콤', '칼칼']):
        preferences.append("✅ 선호 맛 (입력): 매운맛 우선")
    if any(k in user_input_lower for k in ['달콤', '단맛']):
        preferences.append("✅ 선호 맛 (입력): 단맛 우선")
    if any(k in user_input_lower for k in ['안매운', '맵지않은', '순한']):
        preferences.append("❌ 비선호 맛 (입력): 매운맛 제외")
    
    # 3. 요리 타입
    if any(k in user_input_lower for k in ['양식', '서양', '이탈리안', '파스타', '스테이크']):
        requirements.append("⚠️ 중요: 양식/서양 요리만 추천")
    elif any(k in user_input_lower for k in ['한식', '한국', '김치', '된장']):
        requirements.append("⚠️ 중요: 한식 요리만 추천")
    elif any(k in user_input_lower for k in ['중식', '중국', '짜장', '짬뽕']):
        requirements.append("⚠️ 중요: 중식 요리만 추천")
    elif any(k in user_input_lower for k in ['일식', '일본', '스시', '라멘']):
        requirements.append("⚠️ 중요: 일식 요리만 추천")
    
    # 4. 기타 제약
    if any(k in user_input_lower for k in ['간단', '쉬운', '빠른', '10분', '5분']):
        requirements.append("조리 시간이 짧은 레시피 우선")
    if any(k in user_input_lower for k in ['단백질', '두부', '콩']) and '고기' not in user_input_lower:
        requirements.append("단백질이 풍부한 레시피 우선")
    
    # 5. 통합 (우선순위: 식단 제약 > 알레르기 > 선호도 > 기타)
    all_requirements = dietary_restrictions + allergies_list + preferences + requirements
    requirements_text = "\\n".join(all_requirements) if all_requirements else "사용자 요청에 정확히 부합하는 레시피 추천"
    
    # 디버그 로그
    logger.info(f"[PROFILE+PROMPT] 통합 요구사항: {len(all_requirements)}개")
    if dietary_restrictions:
        logger.info(f"  - 식단 제약: {dietary_restrictions}")
    if allergies_list:
        logger.info(f"  - 알레르기: {allergies_list}")
    if preferences:
        logger.info(f"  - 선호도: {preferences}")
    
    prompt = f"""당신은 신뢰 기반 한국어 레시피 추천 전문가입니다.

<CONTEXT>
현재 시간: {state.get('current_time', get_current_time())}
사용자 감성: {sentiment_desc} (점수: {sentiment_score:.2f})

검색된 레시피:
{context}
</CONTEXT>

<USER_REQUEST>
{state['user_input']}

핵심 요구사항:
{requirements_text}
</USER_REQUEST>

아래 JSON 형식으로만 답변하세요:
{{
    "recommendations": [
        {{
            "title": "메뉴명",
            "reason": "추천 이유 (180자 이하)",
            "match_factors": ["감성", "재료", "시간"]
        }}
    ],
    "sentiment": "{sentiment_desc}",
    "timestamp": "{state.get('current_time', get_current_time())}"
}}

규칙:
- 최대 2개 추천
- **🚫 식단 제약을 절대적으로 준수** (예: 비건이면 달걀/우유/고기/생선 포함 레시피 절대 금지)
- **핵심 요구사항을 반드시 충족** (예: 양식 요청 시 한식 추천 금지)
- 다양성 확보 (주재료/조리법 중복 방지)
- reason 첫 문장에 감성 반영
- 검색 결과에 적합한 레시피가 없으면 "검색 결과에 적합한 레시피가 없습니다"라고 명시
- JSON만 출력
"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    try:
        # Parse JSON response
        content = response.content
        # Extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        recommendations_data = json.loads(content)
        recommendations = recommendations_data.get("recommendations", [])
        logger.info(f"Generated {len(recommendations)} recommendations")
    except Exception as e:
        logger.error(f"Failed to parse recommendations: {e}")
        recommendations = [
            {
                "title": "오류 발생",
                "reason": "추천 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
                "match_factors": []
            }
        ]
    
    return {
        **state,
        "recommendations": recommendations,
        "messages": state["messages"] + [AIMessage(content=json.dumps(recommendations_data, ensure_ascii=False, indent=2))],
    }


def get_current_time() -> str:
    """Get formatted current time."""
    return datetime.now().strftime("%p %I시 %M분").replace("AM", "오전").replace("PM", "오후")


# -------------------- Routing Functions --------------------

def should_search_recipes(state: RecommendationState) -> Literal["search", "skip_search"]:
    """Determine if recipe search is needed.
    
    Args:
        state: Current workflow state.
    
    Returns:
        "search" if search needed, "skip_search" otherwise.
    """
    # Always search unless user explicitly requests sentiment-only response
    user_input_lower = state["user_input"].lower()
    if "기분" in user_input_lower and "추천" not in user_input_lower:
        return "skip_search"
    return "search"


# -------------------- Graph Construction --------------------

def create_recommendation_graph(retriever, llm: ChatOpenAI | None = None):
    """Create LangGraph workflow for menu recommendations.
    
    Args:
        retriever: FAISSRecipeRetriever instance.
        llm: ChatOpenAI instance (default: gpt-4o).
    
    Returns:
        Compiled StateGraph.
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Create graph
    workflow = StateGraph(RecommendationState)
    
    # Add nodes
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("search_recipes", lambda state: search_recipes_node(state, retriever))
    workflow.add_node("generate_recommendations", lambda state: generate_recommendations_node(state, llm))
    
    # Add edges
    workflow.set_entry_point("analyze_sentiment")
    
    # Conditional routing after sentiment analysis
    workflow.add_conditional_edges(
        "analyze_sentiment",
        should_search_recipes,
        {
            "search": "search_recipes",
            "skip_search": "generate_recommendations",
        }
    )
    
    workflow.add_edge("search_recipes", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    
    # Compile graph
    app = workflow.compile()
    
    logger.info("✅ Created LangGraph recommendation workflow")
    return app


def run_recommendation_workflow(graph, user_input: str, user_profile: dict = None) -> dict:
    """Execute recommendation workflow.
    
    Args:
        graph: Compiled StateGraph.
        user_input: User query string.
        user_profile: User profile dict (allergies, diet, preferences).
    
    Returns:
        Final state dict with recommendations.
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_input": user_input,
        "user_profile": user_profile or {},
        "sentiment_data": {},
        "search_results": [],
        "recommendations": [],
        "current_time": get_current_time(),
        "iteration_count": 0,
    }
    
    logger.info(f"Starting workflow for: {user_input[:50]}...")
    logger.info(f"Profile: {user_profile}")
    final_state = graph.invoke(initial_state)
    
    return final_state
