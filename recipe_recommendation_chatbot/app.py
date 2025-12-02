"""Streamlit interface for Menu Recommendation Bot with LangChain & LangGraph.

This is an enhanced version of the original app.py using:
- LangChain for retrieval and chain composition
- LangGraph for stateful workflow management
- Improved error handling and caching
- Phase 1-C: OpenAI cost tracking with callbacks
"""
import os
import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import logging
import time
from langchain_community.callbacks import get_openai_callback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("app")

# Add parent directory to path (for accessing data folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules (frequently used)
from data_loader import load_recipe_data, extract_essential_info
from embedding_utils import generate_embeddings
from user_profile import UserProfile
from recipe_retriever import create_recipe_retriever
from recommendation_workflow import create_recommendation_graph, run_recommendation_workflow

# Phase 1-B: Import memory management
from memory_manager import (
    get_session_id, add_user_message, add_ai_message,
    get_memory_summary, format_memory_for_context
)

# Lazy imports (loaded only when needed)
# - sentiment_module: get_user_sentiment (workflow에서만 사용)
# - recommendation_chains: create_recommendation_chain, create_simple_rag_chain (legacy 모드)
# - agent_system, recipe_tools: Hybrid 모드에서만 사용

# -------------------- Configuration --------------------

# Use PROJECT_ROOT to access data folder in parent directory
DATA_PATH = PROJECT_ROOT / "data" / "TB_RECIPE_SEARCH_241226.csv"
FULL_INDEX_PATH = PROJECT_ROOT / "data" / "recipe_full.index"
FULL_DATA_PATH = PROJECT_ROOT / "data" / "recipe_full_with_embeddings.parquet"

# -------------------- Caching Heavy Operations --------------------

@st.cache_resource(show_spinner=False)
def build_langchain_retriever(use_full_index: bool = False, sample_size: int = 300):
    """Build LangChain retriever with FAISS index.
    
    Args:
        use_full_index: If True, load full embeddings from parquet (23K recipes).
        sample_size: Number of samples if use_full_index=False.
    
    Returns:
        (retriever, dataframe) tuple
    """
    logger.info("=" * 60)
    logger.info("[RETRIEVER] Starting retriever initialization")
    start_time = time.time()
    
    # Load full embeddings if available
    if use_full_index and FULL_DATA_PATH.exists():
        logger.info(f"[RETRIEVER] Loading full embeddings from: {FULL_DATA_PATH}")
        load_start = time.time()
        
        df = pd.read_parquet(FULL_DATA_PATH)
        logger.info(f"[RETRIEVER] Loaded {len(df):,} recipes in {time.time() - load_start:.2f}s")
        
        if 'embedding' in df.columns:
            logger.info("[RETRIEVER] Converting embeddings to numpy array...")
            import numpy as np
            emb_start = time.time()
            # Optimized: stack directly without tolist() conversion
            embeddings = np.stack(df['embedding'].values).astype('float32')
            logger.info(f"[RETRIEVER] Embeddings ready: shape={embeddings.shape}, took {time.time() - emb_start:.2f}s")
            
            logger.info("[RETRIEVER] Creating FAISS retriever...")
            retriever = create_recipe_retriever(
                df,
                embeddings_array=embeddings,
                embedding_column='embedding',
                content_column='ESSENTIAL_CONTENT',
                title_column='RCP_TTL',
                top_k=5
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[RETRIEVER] Full index ready in {elapsed:.2f}s")
            logger.info("=" * 60)
            return retriever, df
    
    # Build sample index
    logger.info(f"[RETRIEVER] Building sample index ({sample_size} recipes)")
    df = load_recipe_data(str(DATA_PATH))  # Convert Path to string
    df = extract_essential_info(df)
    
    logger.info(f"[RETRIEVER] Generating embeddings for {min(sample_size, len(df))} recipes...")
    with st.spinner(f"Generating embeddings for {min(sample_size, len(df))} recipes..."):
        sample_df, emb_matrix = generate_embeddings(df, sample_size=sample_size, batch_size=32)
    
    logger.info("[RETRIEVER] Creating FAISS retriever...")
    retriever = create_recipe_retriever(
        sample_df,
        embeddings_array=emb_matrix,
        content_column='ESSENTIAL_CONTENT',
        title_column='RCP_TTL',
        top_k=5
    )
    
    elapsed = time.time() - start_time
    logger.info(f"[RETRIEVER] Sample index ready in {elapsed:.2f}s")
    logger.info("=" * 60)
    return retriever, sample_df


@st.cache_resource(show_spinner=False)
def build_langchain_chain(_retriever):
    """Build LangChain recommendation chain.
    
    Args:
        _retriever: LangChain retriever (prefixed with _ to avoid hashing)
    
    Returns:
        LangChain chain instance
    """
    logger.info("[CHAIN] Building LangChain recommendation chain...")
    from recommendation_chains import create_recommendation_chain
    chain = create_recommendation_chain(_retriever)
    logger.info("[CHAIN] Chain ready")
    return chain


@st.cache_resource(show_spinner=False)
def build_langgraph_workflow(_retriever):
    """Build LangGraph workflow.
    
    Args:
        _retriever: LangChain retriever (prefixed with _ to avoid hashing)
    
    Returns:
        Compiled LangGraph instance
    """
    logger.info("[LANGGRAPH] Building LangGraph workflow...")
    graph = create_recommendation_graph(_retriever)
    logger.info("[LANGGRAPH] Workflow ready")
    return graph


# -------------------- Helper Functions --------------------

def filter_by_profile(docs, profile: UserProfile):
    """Filter retrieved documents by user profile.
    
    Args:
        docs: List of Document objects
        profile: UserProfile instance
    
    Returns:
        Filtered list of documents
    """
    if not profile.allergies and not profile.disliked_flavors and not profile.diet:
        return docs
    
    filtered = []
    for doc in docs:
        content = doc.page_content.lower()
        
        # Check allergies
        if any(allergy.lower() in content for allergy in profile.allergies):
            continue
        
        # Check disliked flavors
        if any(flavor.lower() in content for flavor in profile.disliked_flavors):
            continue
        
        # Check diet restrictions
        if profile.diet == 'vegan':
            vegan_exclude = ['고기', '쇠고기', '돼지고기', '닭고기', '달걀', '치즈', '버터', '우유', '생선']
            if any(ingredient in content for ingredient in vegan_exclude):
                continue
        elif profile.diet == 'low_salt':
            salt_exclude = ['소금', '간장', '액젓', '젓갈']
            if any(ingredient in content for ingredient in salt_exclude):
                continue
        
        filtered.append(doc)
    
    return filtered


def format_recommendations_for_display(recommendations: list, user_input: str) -> list:
    """Post-process recommendations for better display.
    
    Args:
        recommendations: List of recommendation dicts
        user_input: Original user input
    
    Returns:
        Processed recommendations list
    """
    # Remove duplicates based on title similarity
    def _tokens(t):
        return set(str(t).lower().split())
    
    filtered = []
    for r in recommendations:
        title_tokens = _tokens(r.get("title", ""))
        # Check similarity with existing recommendations
        is_duplicate = any(
            len(title_tokens & _tokens(x.get("title", ""))) / 
            max(1, len(title_tokens | _tokens(x.get("title", "")))) > 0.6 
            for x in filtered
        )
        if not is_duplicate:
            filtered.append(r)
    
    # Augment protein-related recommendations
    protein_keywords = ["닭", "두부", "달걀", "콩", "참치", "요거트", "고기"]
    if "단백질" in user_input:
        for r in filtered:
            reason = r.get("reason", "")
            if not any(k in reason for k in protein_keywords):
                # Find protein source in title or add generic note
                title = r.get("title", "")
                protein_source = next((k for k in protein_keywords if k in title), "단백질")
                r["reason"] = f"{reason} ({protein_source} 함유)"
    
    return filtered[:2]  # Limit to top 2


# -------------------- Streamlit UI --------------------

def main():
    logger.info("=" * 60)
    logger.info("[APP] Starting Streamlit app...")
    logger.info("=" * 60)
    
    st.set_page_config(
        page_title="메뉴 추천 챗봇",
        page_icon="🍽️",
        layout="wide"
    )
    
    st.title("메뉴 추천 챗봇")
    st.caption("Menu Recommendation Bot v2.0")
    
    # Initialize session state
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    
    # Phase 1-C: Initialize cost tracking
    if "total_tokens" not in st.session_state:
        st.session_state["total_tokens"] = 0
    if "total_cost" not in st.session_state:
        st.session_state["total_cost"] = 0.0
    if "request_count" not in st.session_state:
        st.session_state["request_count"] = 0
    
    # Phase 1-B: Initialize session ID for memory
    session_id = get_session_id()
    
    # -------------------- Sidebar Configuration (Simplified) --------------------
    
    use_full_index = True
    sample_size = 300
    
    with st.sidebar:
        st.header("사용자 프로필")
        st.caption("알레르기나 선호도를 입력하면 맞춤 추천을 제공합니다")
        
        
        allergies_input = st.text_input(
            "알레르기",
            "",
            placeholder="예: 새우, 땅콩, 밀가루",
            help="쉼표로 구분해서 입력하세요"
        )
        
        preferred_input = st.text_input(
            "선호 맛",
            "",
            placeholder="예: 달콤, 담백, 고소한"
        )
        
        disliked_input = st.text_input(
            "비선호 맛",
            "",
            placeholder="예: 매운, 쓴, 신"
        )
        
        diet = st.selectbox(
            "식단 제약",
            ["없음", "비건 (Vegan)", "저염 (Low Salt)"]
        )
        
        # Build profile
        profile = UserProfile(
            allergies=[a.strip() for a in allergies_input.split(',') if a.strip()],
            preferred_flavors=[p.strip() for p in preferred_input.split(',') if p.strip()],
            disliked_flavors=[d.strip() for d in disliked_input.split(',') if d.strip()],
            diet=None if diet == "없음" else ("vegan" if "Vegan" in diet else "low_salt")
        )
        
        # Display active filters (compact)
        if any([profile.allergies, profile.preferred_flavors, profile.disliked_flavors, profile.diet]):
            st.divider()
            st.caption("**적용 중:**")
            if profile.allergies:
                st.caption(f"알레르기: {', '.join(profile.allergies)}")
            if profile.preferred_flavors:
                st.caption(f"선호: {', '.join(profile.preferred_flavors)}")
            if profile.disliked_flavors:
                st.caption(f"비선호: {', '.join(profile.disliked_flavors)}")
            if profile.diet:
                st.caption(f"식단: {diet}")
    
    # -------------------- Main Content --------------------
    
    # Hybrid mode is default (no selection needed)
    mode = "🔥 Hybrid Mode"  # Multi-Agent + LangGraph
    
    # Load retriever (silent loading)
    retriever, df = build_langchain_retriever(use_full_index, sample_size)
    
    # User input (expanded)
    user_input = st.text_area(
        "원하는 메뉴를 설명해주세요:",
        "",
        height=200,
        placeholder="예시: 운동 후 단백질 많은 메뉴 추천해줘"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        run_button = st.button("추천 받기", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("초기화", use_container_width=True)
    
    if clear_button:
        st.session_state["last_result"] = None
        st.session_state["conversation_history"] = []
        st.rerun()
    
    # -------------------- Recommendation Execution --------------------
    
    if run_button and user_input.strip():
        # Phase 1-B: Add user message to memory
        add_user_message(user_input)
        
        logger.info("=" * 60)
        logger.info(f"[REQUEST] User input: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'")
        logger.info(f"[PROFILE] Allergies: {profile.allergies}, Diet: {profile.diet}")
        logger.info(f"[MEMORY] Session: {session_id[:8]}..., History: {get_memory_summary()['total_messages']} messages")
        
        with st.spinner("추천을 생성하고 있습니다..."):
            try:
                request_start = time.time()
                
                # Create profile-aware retriever wrapper
                logger.info("[PROFILE] Creating profile-aware retriever wrapper...")
                class ProfileAwareRetriever:
                    def __init__(self, base_retriever, profile):
                        self.base_retriever = base_retriever
                        self.profile = profile
                    
                    def invoke(self, query, config=None):
                        """Invoke retriever and filter by profile."""
                        docs = self.base_retriever.invoke(query)
                        return filter_by_profile(docs, self.profile)
                    
                    def get_relevant_documents(self, query):
                        """Backward compatibility wrapper."""
                        return self.invoke(query)
                    
                    def __getattr__(self, name):
                        return getattr(self.base_retriever, name)
                
                profile_retriever = ProfileAwareRetriever(retriever, profile)
                
                # Phase 1-C: Wrap execution in cost tracking callback
                with get_openai_callback() as cb:
                    # Hybrid Mode: Multi-Agent + LangGraph Integration
                    if "Hybrid" in mode:
                        logger.info(f"[HYBRID] Starting Hybrid Mode (Multi-Agent + LangGraph)...")
                        
                        # Step 1: Progress indicator
                        status_container = st.empty()
                        status_container.markdown(
                            '<div style="opacity: 0.6; font-size: 0.85em; color: #666;">'
                            '🔍 LangGraph로 초기 분석 중...</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Step 1: LangGraph for structured analysis
                        logger.info(f"[HYBRID] Phase 1: LangGraph workflow...")
                        graph = build_langgraph_workflow(profile_retriever)
                        langgraph_result = run_recommendation_workflow(graph, user_input)
                        
                        status_container.markdown(
                            '<div style="opacity: 0.6; font-size: 0.85em; color: #666;">'
                            '🤖 Multi-Agent로 정교화 중...</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Step 2: Multi-Agent for refinement
                        logger.info(f"[HYBRID] Phase 2: Multi-Agent refinement...")
                        
                        try:
                            from langchain_openai import ChatOpenAI
                            from datetime import datetime
                            
                            # Lazy import (only when hybrid mode used)
                            from agent_system import create_recommendation_agent, run_agent_recommendation, AGENTS_AVAILABLE
                            from recipe_tools import create_tools
                            
                            if not AGENTS_AVAILABLE:
                                raise ImportError("Agent system not available")
                            
                            # Create tools with current retriever
                            tools = create_tools(profile_retriever)
                            
                            agent_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
                            agent_executor = create_recommendation_agent(
                                tools=tools,
                                llm=agent_llm,
                                max_iterations=2,  # Limited for efficiency
                                max_execution_time=20.0
                            )
                            
                            # Prepare context from LangGraph result
                            langgraph_recs = langgraph_result.get("recommendations", [])
                            context = f"초기 분석 결과: {len(langgraph_recs)}개 레시피 발견."
                            if langgraph_recs:
                                context += f" (예: {langgraph_recs[0].get('title', '')})"
                            
                            profile_summary = f"알레르기: {profile.allergies or '없음'}, 식단: {profile.diet or '제약 없음'}"
                            current_time = datetime.now().strftime("%p %I시 %M분").replace("AM", "오전").replace("PM", "오후")
                            
                            # Enhanced prompt for agent
                            enhanced_input = f"{user_input}\n\n[참고: {context}]"
                            
                            agent_result = run_agent_recommendation(
                                agent_executor=agent_executor,
                                user_input=enhanced_input,
                                current_time=current_time,
                                user_profile=profile_summary
                            )
                            
                            status_container.empty()  # Clear status
                            
                            # Merge results (prefer agent's refined recommendations)
                            result = {
                                "recommendations": agent_result.get("recommendations", langgraph_recs),
                                "sentiment": langgraph_result.get("sentiment_data", {}).get("description", "분석 완료"),
                                "search_count": len(langgraph_result.get("search_results", [])) + agent_result.get("tool_calls", 0),
                                "mode": "Hybrid (Multi-Agent + LangGraph)"
                            }
                            
                            logger.info(f"[HYBRID] Completed: {len(result['recommendations'])} recommendations")
                            
                        except Exception as e:
                            # Agent 실패 시 LangGraph 결과만 사용 (fallback)
                            logger.warning(f"[HYBRID] Agent system failed, using LangGraph only: {e}")
                            status_container.empty()
                            
                            result = {
                                "recommendations": langgraph_result.get("recommendations", []),
                                "sentiment": langgraph_result.get("sentiment_data", {}).get("description", "분석 완료"),
                                "search_count": len(langgraph_result.get("search_results", [])),
                                "mode": "LangGraph Workflow"
                            }
                            
                            logger.info(f"[HYBRID] Fallback completed: {len(result['recommendations'])} recommendations")
                        
                    elif "Multi-Agent" in mode:
                        # Phase 1-A: Multi-Agent System (legacy)
                        status_container = st.empty()
                        status_container.markdown(
                            '<div style="opacity: 0.6; font-size: 0.85em; color: #666;">'
                            '🤖 Multi-Agent 분석 중...</div>',
                            unsafe_allow_html=True
                        )
                        
                        logger.info(f"[AGENT] Starting Multi-Agent system...")
                        
                        # Create tools with current retriever
                        tools = create_tools(profile_retriever)
                        
                        # Create agent executor
                        from langchain_openai import ChatOpenAI
                        from datetime import datetime
                        
                        agent_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
                        agent_executor = create_recommendation_agent(
                            tools=tools,
                            llm=agent_llm,
                            max_iterations=3,
                            max_execution_time=30.0
                        )
                        
                        profile_summary = f"알레르기: {profile.allergies or '없음'}, 식단: {profile.diet or '제약 없음'}"
                        current_time = datetime.now().strftime("%p %I시 %M분").replace("AM", "오전").replace("PM", "오후")
                        
                        agent_result = run_agent_recommendation(
                            agent_executor=agent_executor,
                            user_input=user_input,
                            current_time=current_time,
                            user_profile=profile_summary
                        )
                        
                        status_container.empty()
                        
                        result = {
                            "recommendations": agent_result.get("recommendations", []),
                            "sentiment": "Agent 분석",
                            "search_count": agent_result.get("tool_calls", 0),
                            "mode": "Multi-Agent System"
                        }
                        
                        logger.info(f"[AGENT] Completed with {agent_result.get('tool_calls', 0)} tool calls")
                        
                    elif "LangGraph" in mode:
                        # LangGraph Workflow
                        status_container = st.empty()
                        status_container.markdown(
                            '<div style="opacity: 0.6; font-size: 0.85em; color: #666;">'
                            '🚀 LangGraph Workflow 실행 중...</div>',
                            unsafe_allow_html=True
                        )
                        
                        logger.info(f"[WORKFLOW] Starting LangGraph workflow...")
                        graph = build_langgraph_workflow(profile_retriever)
                        
                        # 프로필 정보를 딕셔너리로 변환
                        profile_dict = {
                            'diet': profile.diet,
                            'allergies': profile.allergies,
                            'preferred_flavors': profile.preferred_flavors,
                            'disliked_flavors': profile.disliked_flavors
                        }
                        
                        final_state = run_recommendation_workflow(graph, user_input, profile_dict)
                        
                        status_container.empty()
                        
                        result = {
                            "recommendations": final_state.get("recommendations", []),
                            "sentiment": final_state.get("sentiment_data", {}).get("description", "분석 안됨"),
                            "search_count": len(final_state.get("search_results", [])),
                            "mode": "LangGraph Workflow"
                        }
                        
                        logger.info(f"[WORKFLOW] Completed with {len(result['recommendations'])} recommendations")
                        
                    elif "Simple RAG" in mode:
                        # Simple RAG Chain
                        status_container = st.empty()
                        status_container.markdown(
                            '<div style="opacity: 0.6; font-size: 0.85em; color: #666;">'
                            '📝 검색 및 생성 중...</div>',
                            unsafe_allow_html=True
                        )
                        
                        logger.info(f"[CHAIN] Starting Simple RAG chain...")
                        from recommendation_chains import create_simple_rag_chain
                        chain = create_simple_rag_chain(profile_retriever)
                        text_result = chain.invoke(user_input)
                        
                        status_container.empty()
                        
                        # Parse text into structured format
                        result = {
                            "recommendations": [
                                {
                                    "title": "AI 추천",
                                    "reason": text_result,
                                    "match_factors": ["검색 기반"]
                                }
                            ],
                            "sentiment": "N/A",
                            "search_count": 3,
                            "mode": "Simple RAG"
                        }
                        
                        logger.info(f"[CHAIN] Simple RAG completed")
                        
                    else:  # Advanced Chain
                        # Advanced Chain with Sentiment
                        logger.info(f"[CHAIN] Starting Advanced chain with sentiment...")
                        chain = build_langchain_chain(profile_retriever)
                        
                        # Get sentiment
                        try:
                            from sentiment_module import get_user_sentiment
                            sentiment_data = get_user_sentiment(user_input)
                        except:
                            logger.warning("[SENTIMENT] Failed to analyze, using neutral")
                            sentiment_data = {
                                "label": "NEUTRAL",
                                "score": 0.5,
                                "description": "중립적인 기분"
                            }
                        
                        chain_result = chain.invoke({
                            "user_input": user_input,
                            "sentiment_data": sentiment_data
                        })
                        
                        result = {
                            "recommendations": chain_result.get("recommendations", []),
                            "sentiment": chain_result.get("sentiment", "분석 안됨"),
                            "search_count": 3,
                            "mode": "Advanced Chain"
                        }
                        
                        logger.info(f"[CHAIN] Advanced chain completed with {len(result['recommendations'])} recommendations")
                
                # Phase 1-C: Update cost tracking from callback
                st.session_state["total_tokens"] += cb.total_tokens
                st.session_state["total_cost"] += cb.total_cost
                st.session_state["request_count"] += 1
                
                logger.info(f"[COST] This request: {cb.total_tokens} tokens, ${cb.total_cost:.4f}")
                logger.info(f"[COST] Session total: {st.session_state['total_tokens']} tokens, ${st.session_state['total_cost']:.4f}")
                
                # Phase 1-B: Add AI response to memory
                recommendations_text = ", ".join([r.get("title", "") for r in result.get("recommendations", [])])
                ai_response = f"추천 메뉴: {recommendations_text}"
                add_ai_message(ai_response)
                
                # Post-process recommendations
                logger.info("[POST-PROCESS] Formatting recommendations...")
                if result["recommendations"]:
                    result["recommendations"] = format_recommendations_for_display(
                        result["recommendations"],
                        user_input
                    )
                
                # Store result
                st.session_state["last_result"] = {
                    "result": result,
                    "input": user_input,
                    "mode": mode
                }
                
                # Add to conversation history
                st.session_state["conversation_history"].append({
                    "query": user_input,
                    "recommendations": result["recommendations"]
                })
                
                elapsed = time.time() - request_start
                logger.info(f"[SUCCESS] Total request time: {elapsed:.2f}s")
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"[ERROR] Recommendation failed: {e}", exc_info=True)
                st.error("❌ 추천 생성 중 오류가 발생했습니다. 다시 시도해주세요.")
                # 상세 에러는 콘솔 로그에만 표시 (st.exception 제거)
    
    # -------------------- Display Results --------------------
    
    if st.session_state["last_result"]:
        last = st.session_state["last_result"]
        result = last["result"]
        
        st.divider()
        
        # Display recommendations (clean, user-focused)
        st.subheader("추천 결과")
        
        recommendations = result.get("recommendations", [])
        
        if not recommendations:
            st.warning("조건에 맞는 추천을 찾을 수 없습니다. 검색 조건을 조정해보세요.")
        else:
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    st.markdown(f"### {i}. {rec.get('title', '메뉴')}")
                    st.write(rec.get('reason', ''))
                    
                    # Match factors as tags (improved visibility)
                    factors = rec.get('match_factors', [])
                    if factors:
                        # Color palette for different tags
                        tag_colors = {
                            "감성": "#FF6B6B",    # Red
                            "재료": "#4ECDC4",    # Teal
                            "시간": "#95E1D3",    # Light teal
                            "계절": "#FFD93D",    # Yellow
                            "날씨": "#6BCB77",    # Green
                            "건강": "#4D96FF",    # Blue
                            "검색 기반": "#A8DADC"  # Light blue
                        }
                        
                        # Build tags HTML safely
                        tag_spans = []
                        for f in factors:
                            color = tag_colors.get(f, "#E0E0E0")
                            tag_html = (
                                f'<span style="'
                                f'background: linear-gradient(135deg, {color} 0%, {color}dd 100%); '
                                f'color: white; '
                                f'padding: 6px 12px; '
                                f'border-radius: 16px; '
                                f'margin-right: 6px; '
                                f'font-size: 0.9em; '
                                f'font-weight: 600; '
                                f'display: inline-block; '
                                f'box-shadow: 0 2px 4px rgba(0,0,0,0.1);'
                                f'">{f}</span>'
                            )
                            tag_spans.append(tag_html)
                        
                        tags_html = '<div style="margin-top: 8px; margin-bottom: 8px;">' + ''.join(tag_spans) + '</div>'
                        st.markdown(tags_html, unsafe_allow_html=True)
                    
                    st.divider()


if __name__ == "__main__":
    main()
