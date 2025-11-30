import numpy as np
import logging
from typing import List, Dict, Optional
from .embedding_utils import get_embedding
from .user_profile import UserProfile, filter_results_by_profile

logger = logging.getLogger("search")

# -------------------- Vector Search --------------------

def search_with_faiss(query_text: str, index, data, k: int = 3, profile: Optional[UserProfile] = None) -> List[Dict[str, str]]:
    """Search similar recipes using FAISS index.

    Args:
        query_text: User query.
        index: FAISS IndexFlatL2 instance.
        data: DataFrame containing 'RCP_TTL' and 'ESSENTIAL_CONTENT'.
        k: Top K results.
    Returns:
        List of recipe dicts with title and content.
    """
    query_embedding = np.array([get_embedding(query_text)], dtype='float32')
    distances, indices = index.search(query_embedding, k)
    results: List[Dict[str, str]] = []
    for idx in indices[0]:
        recipe = data.iloc[idx]
        results.append({
            'title': recipe.get('RCP_TTL', ''),
            'content': recipe.get('ESSENTIAL_CONTENT', '')
        })
    # Profile filtering
    if profile:
        results = filter_results_by_profile(results, profile)
    if not results:
        logger.info("Search returned 0 results after filtering.")
    return results

def display_search_results(results: List[Dict[str, str]], query: str) -> None:
    """Pretty-print search results.

    Args:
        results: List of recipe dictionaries.
        query: Original query string.
    """
    print(f"\nQuery: '{query}' | Results: {len(results)}\n")
    for i, r in enumerate(results, 1):
        preview = r['content'][:100].replace('\n', ' ')
        print(f"{i}. {r['title']}\n   {preview}...")

def search_recipes(query_text: str, index, data, top_k: int = 3, profile: Optional[UserProfile] = None) -> List[Dict[str, str]]:
    """Wrapper function for FAISS recipe search.

    Args:
        query_text: User query.
        index: FAISS index.
        data: DataFrame with embeddings alignment to index.
        top_k: Number of results.
    Returns:
        List of recipe dicts.
    """
    return search_with_faiss(query_text, index, data, k=top_k, profile=profile)
