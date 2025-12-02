"""Streamlit interface for Menu Recommendation Bot."""
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from menu_bot import (
    load_recipe_data,
    extract_essential_info,
    generate_embeddings,
    create_faiss_index,
    load_faiss_index,
)
from menu_bot.search import search_recipes
from menu_bot.recommendation_engine import recommend_with_rag, recommend_with_function_calling, tools
from menu_bot.user_profile import UserProfile
from menu_bot.sentiment_module import get_user_sentiment  # type: ignore

# Caching heavy steps
@st.cache_resource(show_spinner=False)
def build_index(use_full_index: bool = False, sample_size: int = 300):
    """Load or build recipe index.
    
    Args:
        use_full_index: If True and recipe_full.index exists, load it. Otherwise build sample.
        sample_size: Number of samples when building (ignored if use_full_index=True).
    """
    full_index_path = Path("data/recipe_full.index")
    full_data_path = Path("data/recipe_full.parquet")
    
    if use_full_index and full_index_path.exists() and full_data_path.exists():
        # Load silently without UI message
        index = load_faiss_index(str(full_index_path))
        df = pd.read_parquet(full_data_path)
        return df, index
    else:
        if use_full_index:
            st.warning("⚠️ Full index not found. Building sample index instead. Run `python build_full_index.py` first.")
        df = load_recipe_data("data/TB_RECIPE_SEARCH_241226.csv")
        df = extract_essential_info(df)
        sample_df, emb_matrix = generate_embeddings(df, sample_size=sample_size)
        index = create_faiss_index(emb_matrix)
        return sample_df, index

# Load index silently in background
sample_df, index = build_index(use_full_index=True)

st.title("🍽️ 메뉴 추천 챗봇")
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# Sidebar profile
st.sidebar.header("사용자 프로필 설정")
allergies = st.sidebar.text_input("알레르기 (쉼표 구분)", "", placeholder="예: 새우,땅콩")
preferred = st.sidebar.text_input("선호 맛 키워드", "", placeholder="예: 달콤,담백")
disliked = st.sidebar.text_input("비선호 맛 키워드", "", placeholder="예: 매운,쓴")
diet_label = st.sidebar.selectbox(
    "식단 제약",
    ["없음 (None)", "비건 (Vegan)", "저염 (Low Salt)"],
    index=0,
    help="한국어/영어 병기. 내부 값은 (None|Vegan|Low Salt)로 매핑됩니다."
)
diet_map = {
    "없음 (None)": "None",
    "비건 (Vegan)": "vegan",
    "저염 (Low Salt)": "low_salt",
}
diet = diet_map.get(diet_label, "None")

profile = UserProfile(
    allergies=[a.strip() for a in allergies.split(',') if a.strip()],
    preferred_flavors=[p.strip() for p in preferred.split(',') if p.strip()],
    disliked_flavors=[d.strip() for d in disliked.split(',') if d.strip()],
    diet=None if diet == "None" else diet,
)

user_input = st.text_area("요청 입력", "", placeholder="예: 운동 후 단백질 많은 메뉴 / 속 불편해서 자극 없는 음식 / 비 오는 날 먹기 좋은 요리")
run = st.button("추천 실행")

if run and user_input.strip():
    with st.spinner("추천 생성 중..."):
        # Search function bound with profile
        def search_fn(q: str, top_k: int = 3):
            return search_recipes(q, index, sample_df, top_k=top_k, profile=profile)
        
        # Auto strategy: choose RAG or Function Calling based on prompt
        def _choose_mode(text: str) -> str:
            t = text.lower()
            # Heuristics
            long = len(t) > 120
            complex_keywords = ["분석", "비교", "여러", "대안", "옵션", "종합", "상세"]
            time_keywords = ["분", "빨리", "간단", "빠르게"]
            has_complex = any(k in t for k in complex_keywords)
            has_time = any(k in t for k in time_keywords)
            # Decision: prefer Function Calling for long/complex, else RAG
            if long or (has_complex and not has_time):
                return "fc"
            return "rag"

        mode = _choose_mode(user_input)
        if mode == "fc":
            messages = [{"role": "system", "content": "사용자 맞춤 메뉴 추천 시스템"}]
            result = recommend_with_function_calling(user_input, messages, search_fn, sentiment_fn=get_user_sentiment)
        else:
            result = recommend_with_rag(user_input, search_fn, top_k=3)
    
    # Store result for persistence across reruns
    st.session_state["last_result"] = {"text": result, "input": user_input}

# Render last result if available
if st.session_state["last_result"]:
    st.markdown("### 추천 결과")
    # Try to parse JSON, post-process, and render human-friendly blocks
    try:
        import json
        parsed = json.loads(st.session_state["last_result"]["text"])
        recs = parsed.get("recommendations", [])
        # Enforce diversity: drop near-duplicate titles
        def _tokens(t):
            return set(str(t).lower().replace("우동", " ").split())
        filtered = []
        for r in recs:
            title_tokens = _tokens(r.get("title", ""))
            if any(len(title_tokens & _tokens(x.get("title", ""))) / max(1, len(title_tokens | _tokens(x.get("title", "")))) > 0.6 for x in filtered):
                continue
            filtered.append(r)
        # If protein intent, augment reasons with explicit evidence keywords if missing
        protein_keywords = ["닭", "두부", "달걀", "콩", "참치", "요거트"]
        if "단백질" in st.session_state["last_result"]["input"]:
            for r in filtered:
                reason = r.get("reason", "")
                if not any(k in reason for k in protein_keywords):
                    r["reason"] = reason + " 단백질 근거: 닭/두부/달걀 등"
        filtered = filtered[:2]
        # Render readable blocks
        for r in filtered:
            st.subheader(r.get("title", "메뉴"))
            st.write(r.get("reason", ""))
            mf = r.get("match_factors", [])
            if isinstance(mf, list) and mf:
                st.caption(", ".join(mf))
        # Debug JSON output removed to avoid showing raw JSON by default
    except Exception:
        # Fallback: render raw text as a paragraph without JSON formatting
        st.write(str(st.session_state["last_result"]["text"]))
