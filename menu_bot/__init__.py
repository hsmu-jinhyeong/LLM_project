"""Menu Recommendation Bot package.

Provides data loading, token analysis, embedding generation, FAISS vector search,
recipe retrieval, and GPT-based recommendation functions.
"""
from .data_loader import load_recipe_data, prepare_embedding_data, extract_essential_info
from .token_utils import count_tokens, analyze_token_statistics
from .embedding_utils import get_embedding, get_embeddings_batch, generate_embeddings
from .faiss_index import create_faiss_index, load_faiss_index
from .search import search_with_faiss, search_recipes, display_search_results
from .user_profile import UserProfile, filter_results_by_profile
from .recommendation_engine import (
    get_current_time,
    format_recipe_context,
    create_recommendation_prompt,
    recommend_with_rag,
    recommend_with_function_calling,
    tools,
)
