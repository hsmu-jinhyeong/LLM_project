"""LangGraph-based stateful recommendation workflow.

This module implements a graph-based workflow for menu recommendations
with state management, multi-step reasoning, and conditional routing.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Sequence, Literal
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
    sentiment data, and final recommendations.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
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
    
    # Use invoke() method for LangChain BaseRetriever
    docs = retriever.invoke(query)
    
    results = [
        {
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content[:200],
            "category": doc.metadata.get("category", ""),
        }
        for doc in docs
    ]
    
    logger.info(f"Retrieved {len(results)} recipes")
    
    return {
        **state,
        "search_results": results,
        "messages": state["messages"] + [
            SystemMessage(content=f"레시피 검색 완료: {len(results)}개 발견")
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
    
    prompt = f"""당신은 신뢰 기반 한국어 레시피 추천 전문가입니다.

<CONTEXT>
현재 시간: {state.get('current_time', get_current_time())}
사용자 감성: {sentiment_desc} (점수: {sentiment_score:.2f})

검색된 레시피:
{context}
</CONTEXT>

사용자 요청: {state['user_input']}

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
- 다양성 확보 (주재료/조리법 중복 방지)
- reason 첫 문장에 감성 반영
- 검색 결과 기반 추천 (임의 발명 금지)
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


def run_recommendation_workflow(graph, user_input: str) -> dict:
    """Execute recommendation workflow.
    
    Args:
        graph: Compiled StateGraph.
        user_input: User query string.
    
    Returns:
        Final state dict with recommendations.
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_input": user_input,
        "sentiment_data": {},
        "search_results": [],
        "recommendations": [],
        "current_time": get_current_time(),
        "iteration_count": 0,
    }
    
    logger.info(f"Starting workflow for: {user_input[:50]}...")
    final_state = graph.invoke(initial_state)
    
    return final_state
