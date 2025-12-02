# Menu Bot LangChain

LangChain 및 LangGraph를 적용한 고급 메뉴 추천 시스템입니다.

**✨ 완전 독립 실행 가능**: 기존 `menu_bot` 패키지 없이 단독으로 실행됩니다.

## 🎯 주요 기능

### 1. LangChain 통합
- **Custom Retriever**: FAISS 기반 벡터 검색을 LangChain Retriever로 구현
- **LCEL Chains**: 선언적 체인으로 RAG 파이프라인 구성
- **LangChain Tools**: 레시피 검색 및 감성 분석 도구

### 2. LangGraph 워크플로우
- **상태 관리**: TypedDict 기반 대화 상태 추적
- **조건부 라우팅**: 사용자 의도에 따른 동적 워크플로우
- **노드 기반 설계**: 감성 분석 → 검색 → 추천 생성

## 📦 설치

```bash
pip install -r requirements.txt
```

## 🚀 빠른 시작

### Example 1: Simple RAG
```python
from menu_bot_langchain import (
    load_recipe_data,
    extract_essential_info,
    create_recipe_retriever,
    create_simple_rag_chain,
)
from menu_bot.embedding_utils import generate_embeddings

# 데이터 준비
df = load_recipe_data("data/TB_RECIPE_SEARCH_241226.csv")
df = extract_essential_info(df)
sample_df, emb_matrix = generate_embeddings(df, sample_size=50)

# Retriever 생성
retriever = create_recipe_retriever(sample_df, embeddings_array=emb_matrix)

# Chain 실행
chain = create_simple_rag_chain(retriever)
result = chain.invoke("빠른 아침 메뉴 추천해줘")
print(result)
```

### Example 2: Advanced Chain with Sentiment
```python
from menu_bot_langchain import create_recommendation_chain
from menu_bot.sentiment_module import get_user_sentiment

# Retriever 생성 (위와 동일)
retriever = create_recipe_retriever(sample_df, embeddings_array=emb_matrix)

# 감성 분석 포함 Chain
chain = create_recommendation_chain(retriever)

user_input = "우울해서 위로되는 음식 먹고 싶어"
sentiment_data = get_user_sentiment(user_input)

result = chain.invoke({
    "user_input": user_input,
    "sentiment_data": sentiment_data
})

print(result)  # JSON 형식 추천
```

### Example 3: LangGraph Workflow
```python
from menu_bot_langchain import (
    create_recommendation_graph,
    run_recommendation_workflow,
)

# Retriever 생성 (위와 동일)
retriever = create_recipe_retriever(sample_df, embeddings_array=emb_matrix)

# LangGraph 워크플로우
graph = create_recommendation_graph(retriever)

final_state = run_recommendation_workflow(
    graph, 
    "운동 후 단백질 많은 메뉴"
)

print(f"감성: {final_state['sentiment_data']['description']}")
print(f"추천: {final_state['recommendations']}")
```

## 📚 모듈 설명

### `langchain_retriever.py`
- `FAISSRecipeRetriever`: LangChain BaseRetriever 상속
- `create_recipe_retriever()`: Retriever 팩토리 함수

### `langchain_chains.py`
- `create_recommendation_chain()`: 감성 분석 + RAG 체인
- `create_simple_rag_chain()`: 기본 RAG 체인

### `langchain_tools.py`
- `RecipeSearchTool`: 레시피 검색 도구
- `SentimentAnalysisTool`: 감성 분석 도구

### `langgraph_workflow.py`
- `RecommendationState`: 워크플로우 상태 정의
- `create_recommendation_graph()`: LangGraph 생성
- `run_recommendation_workflow()`: 워크플로우 실행

## 🔧 환경 변수

`.env` 파일에 다음 설정:
```
OPENAI_API_KEY=your_api_key_here
```

## 📊 기존 menu_bot과의 차이점

| 기능 | menu_bot | menu_bot_langchain |
|------|----------|-------------------|
| 검색 | 함수 기반 | LangChain Retriever |
| 추천 | GPT 직접 호출 | LCEL Chain |
| 워크플로우 | 순차 실행 | LangGraph State Machine |
| 상태 관리 | 없음 | TypedDict State |
| 확장성 | 제한적 | 높음 |

## 🎓 주요 개념

### LCEL (LangChain Expression Language)
선언적 방식으로 체인 구성:
```python
chain = retriever | format_docs | prompt | llm | parser
```

### LangGraph State
상태 기반 워크플로우:
```python
class State(TypedDict):
    messages: List[BaseMessage]
    search_results: list
    recommendations: list
```

## 📝 전체 예제 실행

```bash
python menu_bot_langchain/run_examples.py
```

## 🔮 확장 가능성

1. **Multi-Agent**: 검색/추천 에이전트 분리
2. **Memory**: 대화 히스토리 장기 저장
3. **Advanced RAG**: Re-ranking, HyDE
4. **LangSmith**: 프로덕션 모니터링

## 📄 라이선스

기존 menu_bot 프로젝트와 동일

## 🤝 기여

이슈 및 PR 환영합니다!
