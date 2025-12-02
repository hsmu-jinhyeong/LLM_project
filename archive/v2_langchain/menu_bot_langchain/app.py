"""Streamlit interface for Menu Recommendation Bot with LangChain & LangGraph.

This is an enhanced version of the original app.py using:
- LangChain for retrieval and chain composition
- LangGraph for stateful workflow management
- Improved error handling and caching
"""
import os
import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import logging
import time

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

# Import from local menu_bot_langchain modules
from data_loader import load_recipe_data, extract_essential_info
from embedding_utils import generate_embeddings
from user_profile import UserProfile
from sentiment_module import get_user_sentiment
from faiss_index import load_faiss_index

from recipe_retriever import create_recipe_retriever
from recommendation_chains import create_recommendation_chain, create_simple_rag_chain
from recommendation_workflow import create_recommendation_graph, run_recommendation_workflow

# -------------------- Configuration --------------------

DATA_PATH = "data/TB_RECIPE_SEARCH_241226.csv"
FULL_INDEX_PATH = Path("data/recipe_full.index")
FULL_DATA_PATH = Path("data/recipe_full_with_embeddings.parquet")  # Updated to use embeddings

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
            embeddings = np.array(df['embedding'].tolist(), dtype='float32')
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
    df = load_recipe_data(DATA_PATH)
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
    
    # Initialize session state
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    
    # -------------------- Sidebar Configuration (Simplified) --------------------
    
    # Hidden settings - always use best mode
    mode = "🚀 LangGraph Workflow (권장)"  # Always use LangGraph
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
    
    # Load retriever (silent loading)
    retriever, df = build_langchain_retriever(use_full_index, sample_size)
    
    # User input
    user_input = st.text_area(
        "원하는 메뉴를 설명해주세요:",
        "",
        height=100,
        placeholder="예시:\n- 운동 후 단백질 많은 메뉴 추천해줘\n- 속이 불편해서 자극 없는 음식이 필요해\n- 비 오는 날 먹기 좋은 따뜻한 요리\n- 데이트할 때 먹을 분위기 있는 음식"
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
        logger.info("=" * 60)
        logger.info(f"[REQUEST] User input: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'")
        logger.info(f"[PROFILE] Allergies: {profile.allergies}, Diet: {profile.diet}")
        
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
                
                # Execute based on selected mode
                logger.info(f"[WORKFLOW] Starting LangGraph workflow...")
                if "LangGraph" in mode:
                    # LangGraph Workflow
                    graph = build_langgraph_workflow(profile_retriever)
                    final_state = run_recommendation_workflow(graph, user_input)
                    
                    result = {
                        "recommendations": final_state.get("recommendations", []),
                        "sentiment": final_state.get("sentiment_data", {}).get("description", "분석 안됨"),
                        "search_count": len(final_state.get("search_results", [])),
                        "mode": "LangGraph Workflow"
                    }
                    
                    logger.info(f"[WORKFLOW] Completed with {len(result['recommendations'])} recommendations")
                    
                elif "Simple RAG" in mode:
                    # Simple RAG Chain
                    logger.info(f"[CHAIN] Starting Simple RAG chain...")
                    chain = create_simple_rag_chain(profile_retriever)
                    text_result = chain.invoke(user_input)
                    
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
                st.error(f"❌ 추천 생성 중 오류가 발생했습니다: {str(e)}")
                st.exception(e)
    
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
    
    # -------------------- Footer --------------------
    
    st.divider()
    st.caption("🔧 Powered by LangChain & LangGraph | 🍽️ Menu Recommendation Bot v2.0")


if __name__ == "__main__":
    main()
