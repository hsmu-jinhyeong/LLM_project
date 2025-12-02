"""LangChain tools for recipe search and sentiment analysis."""
from __future__ import annotations
from typing import Type, Optional
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("recipe_tools")


# -------------------- Tool Schemas --------------------

class RecipeSearchInput(BaseModel):
    """Input schema for recipe search tool."""
    query: str = Field(description="검색할 쿼리 텍스트 (예: 닭고기 요리, 매운 음식)")
    top_k: int = Field(default=3, description="반환할 레시피 개수")


class SentimentAnalysisInput(BaseModel):
    """Input schema for sentiment analysis tool."""
    text: str = Field(description="감성 분석할 텍스트")


# -------------------- Recipe Search Tool --------------------

class RecipeSearchTool(BaseTool):
    """Tool for searching recipes using vector similarity.
    
    This tool integrates the FAISS-based retriever as a LangChain tool
    that can be used in agents or chains with function calling.
    """
    
    name: str = "search_recipes"
    description: str = "벡터 검색 기반으로 레시피를 조회합니다. 사용자의 요구사항에 맞는 레시피를 찾을 때 사용하세요."
    args_schema: Type[BaseModel] = RecipeSearchInput
    retriever: Optional[object] = None  # FAISSRecipeRetriever instance
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(
        self,
        query: str,
        top_k: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute recipe search.
        
        Args:
            query: Search query.
            top_k: Number of results.
            run_manager: Callback manager (optional).
        
        Returns:
            Formatted search results as string.
        """
        if self.retriever is None:
            return "오류: 검색 엔진이 초기화되지 않았습니다."
        
        try:
            # Update retriever's top_k temporarily
            original_k = self.retriever.top_k
            self.retriever.top_k = top_k
            
            docs = self.retriever.get_relevant_documents(query)
            
            # Restore original top_k
            self.retriever.top_k = original_k
            
            if not docs:
                return "검색 결과가 없습니다."
            
            results = []
            for i, doc in enumerate(docs, 1):
                title = doc.metadata.get('title', '제목 없음')
                content = doc.page_content[:120]
                results.append(f"{i}. {title}\n   {content}...")
            
            logger.info(f"RecipeSearchTool retrieved {len(docs)} results for: {query[:50]}")
            return "\n\n".join(results)
        
        except Exception as e:
            logger.error(f"RecipeSearchTool error: {e}")
            return f"검색 중 오류 발생: {str(e)}"


# -------------------- Sentiment Analysis Tool --------------------

class SentimentAnalysisTool(BaseTool):
    """Tool for analyzing user sentiment using transformers pipeline.
    
    Wraps the original sentiment_module functionality as a LangChain tool.
    """
    
    name: str = "get_user_sentiment"
    description: str = "사용자 텍스트의 감성을 분석합니다 (긍정/부정/중립). 사용자의 기분이나 감정 상태를 파악할 때 사용하세요."
    args_schema: Type[BaseModel] = SentimentAnalysisInput
    
    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Execute sentiment analysis.
        
        Args:
            text: Text to analyze.
            run_manager: Callback manager (optional).
        
        Returns:
            Sentiment analysis result dict.
        """
        try:
            # Import sentiment module from local package
            from .sentiment_module import get_user_sentiment
            result = get_user_sentiment(text)
            logger.info(f"SentimentAnalysisTool result: {result.get('description')}")
            return result
        except ImportError:
            logger.warning("sentiment_module not available, using fallback")
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "description": "중립적인 기분"
            }
        except Exception as e:
            logger.error(f"SentimentAnalysisTool error: {e}")
            return {
                "label": "ERROR",
                "score": 0.0,
                "description": "분석 오류"
            }


def create_tools(retriever):
    """Create a list of tools for agent use.
    
    Args:
        retriever: FAISSRecipeRetriever instance.
    
    Returns:
        List of LangChain tools.
    """
    tools = [
        RecipeSearchTool(retriever=retriever),
        SentimentAnalysisTool(),
    ]
    logger.info(f"✅ Created {len(tools)} LangChain tools")
    return tools
