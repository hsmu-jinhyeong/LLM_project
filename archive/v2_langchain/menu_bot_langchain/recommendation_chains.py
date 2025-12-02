"""LangChain LCEL-based recommendation chains."""
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import logging

logger = logging.getLogger("recommendation_chains")


def get_current_time() -> str:
    """Get formatted current time in Korean."""
    return datetime.now().strftime("%p %I시 %M분").replace("AM", "오전").replace("PM", "오후")


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into context string.
    
    Args:
        docs: List of Document objects.
    
    Returns:
        Formatted context string.
    """
    if not docs:
        return "검색된 레시피가 없습니다."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get('title', '제목 없음')
        content = doc.page_content[:160]
        formatted.append(f"{i}. 제목: {title}\n   내용: {content}...")
    
    return "\n---\n".join(formatted)


# System prompt template for recommendation
RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 신뢰 기반 한국어 레시피 추천 전문가입니다.
검색된 레시피에 없는 재료나 조리법을 임의로 발명하지 마십시오.

<CONTEXT>
현재 시간: {current_time}
사용자 감성: {sentiment_description} (점수: {sentiment_score:.2f})

검색된 레시피:
{context}
</CONTEXT>

출력 형식 (JSON):
{{
    "recommendations": [
        {{
            "title": "메뉴명",
            "reason": "추천 이유 (180자 이하)",
            "match_factors": ["감성", "재료", "시간"]
        }}
    ],
    "sentiment": "{sentiment_description}",
    "timestamp": "{current_time}"
}}

규칙:
- 최대 2개 추천
- 다양성 확보 (중복되지 않는 주재료/조리법)
- reason 첫 문장에 감성 반영
- "단백질" 언급 시 단백질 근거 명시
- 시간 제약 언급 시 조리 시간 표시
- "데이트"/"분위기" 언급 시 플레이팅/감성 요소 반영
- "모임"/"파티" 등 단체 식사 시 공유형 메뉴 우선
- 레시피 없으면 일반 안전 식단 1개만 추천
- JSON만 출력, 다른 텍스트 금지
"""),
    ("human", "{user_input}"),
])


def create_recommendation_chain(retriever, llm: ChatOpenAI | None = None):
    """Create LangChain recommendation chain using LCEL.
    
    Args:
        retriever: LangChain retriever instance (e.g., FAISSRecipeRetriever).
        llm: ChatOpenAI instance (default: gpt-4o).
    
    Returns:
        Runnable chain that takes user_input and sentiment_data, returns recommendation JSON.
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Define chain using LCEL
    chain = (
        RunnableParallel(
            {
                "context": lambda x: format_docs(retriever.get_relevant_documents(x["user_input"])),
                "user_input": lambda x: x["user_input"],
                "sentiment_description": lambda x: x.get("sentiment_data", {}).get("description", "중립적인 기분"),
                "sentiment_score": lambda x: x.get("sentiment_data", {}).get("score", 0.5),
                "current_time": lambda x: get_current_time(),
            }
        )
        | RECOMMENDATION_PROMPT
        | llm
        | JsonOutputParser()
    )
    
    logger.info("✅ Created LangChain recommendation chain (LCEL)")
    return chain


def create_simple_rag_chain(retriever, llm: ChatOpenAI | None = None):
    """Create a simplified RAG chain without sentiment analysis.
    
    Args:
        retriever: LangChain retriever instance.
        llm: ChatOpenAI instance (default: gpt-4o).
    
    Returns:
        Runnable chain for basic RAG recommendations.
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    simple_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 한국어 레시피 추천 전문가입니다.
아래 검색된 레시피를 바탕으로 사용자에게 적합한 메뉴를 1~2개 추천하세요.

검색된 레시피:
{context}

추천 형식: 메뉴명과 간단한 이유를 한 문장으로 작성하세요."""),
        ("human", "{question}"),
    ])
    
    chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | simple_prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("✅ Created simple RAG chain")
    return chain
