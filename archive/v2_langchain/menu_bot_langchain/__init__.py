"""Menu Recommendation Bot with LangChain & LangGraph integration.

This package provides an enhanced version of the menu bot using:
- LangChain for RAG pipeline and chain composition
- LangGraph for stateful recommendation workflow
- Custom retrievers and tools integration
"""
from .data_loader import load_recipe_data, extract_essential_info
from .embedding_utils import get_embedding, get_embeddings_batch, generate_embeddings
from .user_profile import UserProfile, filter_results_by_profile
from .sentiment_module import get_user_sentiment
from .faiss_index import create_faiss_index, load_faiss_index
from .recipe_retriever import FAISSRecipeRetriever, create_recipe_retriever
from .recommendation_chains import create_recommendation_chain, create_simple_rag_chain
from .recipe_tools import RecipeSearchTool, SentimentAnalysisTool, create_tools
from .recommendation_workflow import create_recommendation_graph, RecommendationState, run_recommendation_workflow

__version__ = "2.0.0"
__all__ = [
    # Data & Utilities
    "load_recipe_data",
    "extract_essential_info",
    "get_embedding",
    "get_embeddings_batch",
    "generate_embeddings",
    "UserProfile",
    "filter_results_by_profile",
    "get_user_sentiment",
    "create_faiss_index",
    "load_faiss_index",
    # LangChain Components
    "FAISSRecipeRetriever",
    "create_recipe_retriever",
    "create_recommendation_chain",
    "create_simple_rag_chain",
    "RecipeSearchTool",
    "SentimentAnalysisTool",
    "create_tools",
    # LangGraph Components
    "create_recommendation_graph",
    "RecommendationState",
    "run_recommendation_workflow",
]
