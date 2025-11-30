"""Minimal runnable example for the modular menu bot.

Usage (after setting OPENAI_API_KEY):
    python menu_bot/run_example.py
"""
import os
from menu_bot import (
    load_recipe_data,
    prepare_embedding_data,
    extract_essential_info,
    analyze_token_statistics,
    generate_embeddings,
    create_faiss_index,
    search_recipes,
    recommend_with_rag,
)
from menu_bot.search import search_recipes as core_search

DATA_PATH = "data/TB_RECIPE_SEARCH_241226.csv"


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY 환경 변수를 설정하세요.")

    print("[1] 데이터 로드")
    df = load_recipe_data(DATA_PATH)
    df = extract_essential_info(df)

    print("[2] 토큰 통계")
    analyze_token_statistics(df, content_column='RECIPE_CONTENT')

    print("[3] 임베딩 (샘플 50)")
    sample_df, emb_matrix = generate_embeddings(df, sample_size=50)

    print("[4] FAISS 인덱스 생성")
    index = create_faiss_index(emb_matrix)

    # 래퍼 검색 함수 구성
    def search_fn(query, top_k=3):
        return core_search(query, index, sample_df, top_k=top_k)

    print("[5] 추천 테스트")
    user_text = "오늘 피곤하고 위로가 되는 따뜻한 음식 추천해줘"
    rec = recommend_with_rag(user_text, search_fn, top_k=3)
    print("\n--- 추천 결과 ---\n")
    print(rec)

if __name__ == "__main__":
    main()
