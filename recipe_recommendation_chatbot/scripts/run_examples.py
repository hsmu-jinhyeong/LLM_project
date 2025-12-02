"""Example usage of LangChain-based menu recommendation system.

This example demonstrates:
1. Basic RAG chain usage
2. Advanced recommendation chain with sentiment
3. LangGraph workflow execution
"""
import os
import sys
from pathlib import Path

# Add parent directory to path (for accessing data folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Import from menu_bot_langchain package
from menu_bot_langchain import (
    load_recipe_data,
    extract_essential_info,
    create_recipe_retriever,
    create_recommendation_chain,
    create_simple_rag_chain,
    create_recommendation_graph,
    run_recommendation_workflow,
    generate_embeddings,
    get_user_sentiment,
)

# Load environment
load_dotenv()

DATA_PATH = "data/TB_RECIPE_SEARCH_241226.csv"


def example_1_simple_rag():
    """Example 1: Simple RAG chain without sentiment analysis."""
    print("\n" + "="*60)
    print("Example 1: Simple RAG Chain")
    print("="*60)
    
    # Load and prepare data
    print("[1] Loading data...")
    df = load_recipe_data(DATA_PATH)
    df = extract_essential_info(df)
    
    # Generate embeddings (sample)
    print("[2] Generating embeddings (sample 50)...")
    sample_df, emb_matrix = generate_embeddings(df, sample_size=50, batch_size=32)
    
    # Create retriever
    print("[3] Creating LangChain retriever...")
    retriever = create_recipe_retriever(
        sample_df,
        embeddings_array=emb_matrix,
        top_k=3
    )
    
    # Create simple RAG chain
    print("[4] Creating simple RAG chain...")
    chain = create_simple_rag_chain(retriever)
    
    # Execute query
    query = "빠르게 만들 수 있는 아침 메뉴 추천해줘"
    print(f"\n[질문] {query}")
    
    result = chain.invoke(query)
    print("\n[추천 결과]")
    print(result)
    print()


def example_2_advanced_chain():
    """Example 2: Advanced recommendation chain with sentiment."""
    print("\n" + "="*60)
    print("Example 2: Advanced Recommendation Chain")
    print("="*60)
    
    # Load and prepare data
    print("[1] Loading data...")
    df = load_recipe_data(DATA_PATH)
    df = extract_essential_info(df)
    
    # Generate embeddings
    print("[2] Generating embeddings (sample 50)...")
    sample_df, emb_matrix = generate_embeddings(df, sample_size=50, batch_size=32)
    
    # Create retriever
    print("[3] Creating LangChain retriever...")
    retriever = create_recipe_retriever(
        sample_df,
        embeddings_array=emb_matrix,
        top_k=3
    )
    
    # Create recommendation chain
    print("[4] Creating recommendation chain...")
    chain = create_recommendation_chain(retriever)
    
    # Prepare input with sentiment
    user_input = "기분이 우울해서 따뜻하고 위로되는 음식 먹고 싶어"
    
    # Get sentiment
    try:
        sentiment_data = get_user_sentiment(user_input)
    except:
        sentiment_data = {"label": "NEUTRAL", "score": 0.5, "description": "중립적인 기분"}
    
    print(f"\n[질문] {user_input}")
    print(f"[감성 분석] {sentiment_data.get('description')} (점수: {sentiment_data.get('score', 0):.2f})")
    
    # Execute chain
    result = chain.invoke({
        "user_input": user_input,
        "sentiment_data": sentiment_data
    })
    
    print("\n[추천 결과 (JSON)]")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def example_3_langgraph_workflow():
    """Example 3: LangGraph stateful workflow."""
    print("\n" + "="*60)
    print("Example 3: LangGraph Workflow")
    print("="*60)
    
    # Load and prepare data
    print("[1] Loading data...")
    df = load_recipe_data(DATA_PATH)
    df = extract_essential_info(df)
    
    # Generate embeddings
    print("[2] Generating embeddings (sample 50)...")
    sample_df, emb_matrix = generate_embeddings(df, sample_size=50, batch_size=32)
    
    # Create retriever
    print("[3] Creating LangChain retriever...")
    retriever = create_recipe_retriever(
        sample_df,
        embeddings_array=emb_matrix,
        top_k=3
    )
    
    # Create LangGraph workflow
    print("[4] Creating LangGraph workflow...")
    graph = create_recommendation_graph(retriever)
    
    # Execute workflow
    user_input = "운동 끝나고 단백질 많은 메뉴 추천해줘"
    print(f"\n[질문] {user_input}")
    
    final_state = run_recommendation_workflow(graph, user_input)
    
    print("\n[워크플로우 완료]")
    print(f"감성: {final_state['sentiment_data'].get('description')}")
    print(f"검색 결과: {len(final_state['search_results'])}개")
    print(f"추천: {len(final_state['recommendations'])}개")
    
    print("\n[추천 내용]")
    for i, rec in enumerate(final_state['recommendations'], 1):
        print(f"\n{i}. {rec.get('title')}")
        print(f"   이유: {rec.get('reason')}")
        print(f"   요소: {', '.join(rec.get('match_factors', []))}")
    print()


def main():
    """Run all examples."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY 환경 변수를 설정하세요.")
    
    print("\n" + "🍽️ " * 20)
    print("LangChain & LangGraph Menu Recommendation Examples")
    print("🍽️ " * 20)
    
    # Run examples
    example_1_simple_rag()
    example_2_advanced_chain()
    example_3_langgraph_workflow()
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
