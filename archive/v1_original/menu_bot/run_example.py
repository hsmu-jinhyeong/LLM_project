"""Minimal runnable example for the modular menu bot.

Usage (after setting OPENAI_API_KEY):
    python menu_bot/run_example.py

Optional: Uncomment token statistics for cost estimation (requires tiktoken)
"""
import os
from menu_bot import (
    load_recipe_data,
    extract_essential_info,
    generate_embeddings,
    create_faiss_index,
    recommend_with_rag,
)
# Optional: Token analysis (requires: pip install tiktoken)
# from menu_bot.token_utils import analyze_token_statistics

from menu_bot.search import search_recipes as core_search

DATA_PATH = "data/TB_RECIPE_SEARCH_241226.csv"


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY 환경 변수를 설정하세요.")

    print("[1] 데이터 로드 & 전처리")
    df = load_recipe_data(DATA_PATH)
    df = extract_essential_info(df)
    print(f"    로드 완료: {len(df):,}개 레시피")
    
    # Optional: 토큰 통계 분석 (비용 예측용)
    # analyze_token_statistics(df, content_column='RECIPE_CONTENT')

    print("[2] 임베딩 생성 (샘플 50개)")
    sample_df, emb_matrix = generate_embeddings(df, sample_size=50, batch_size=32)

    print("[3] FAISS 인덱스 생성")
    index = create_faiss_index(emb_matrix)

    # 래퍼 검색 함수 구성
    def search_fn(query, top_k=3):
        return core_search(query, index, sample_df, top_k=top_k)

    print("[4] 추천 테스트")
    user_text = "오늘 피곤하고 위로가 되는 따뜻한 음식 추천해줘"
    print(f"    질문: {user_text}")
    rec = recommend_with_rag(user_text, search_fn, top_k=3)
    print("\n--- 추천 결과 ---")
    print(rec)
    print("\n✅ 예제 실행 완료")

if __name__ == "__main__":
    main()
