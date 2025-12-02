import os
import time
import random
import logging
from functools import lru_cache
import numpy as np
from openai import OpenAI
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv

# -------------------- Embedding Utilities --------------------

_client = None
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("embedding_utils")

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        _client = OpenAI(api_key=api_key)
    return _client

@lru_cache(maxsize=1024)
def _cached_embedding(text: str, model: str) -> List[float]:
    client = _get_client()
    clean = (text or '').replace('\n', ' ').strip()
    for attempt in range(4):
        try:
            response = client.embeddings.create(input=[clean], model=model)
            return response.data[0].embedding
        except Exception as e:
            if attempt == 3:
                logger.error(f"Embedding failed permanently: {e}")
                raise
            backoff = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"Embedding retry {attempt+1} after error: {e} (sleep {backoff:.1f}s)")
            time.sleep(backoff)

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Fetch embedding vector from OpenAI with retry + caching.

    Long inputs (>4000 chars) will be truncated to preserve token budget.
    """
    if text and len(text) > 4000:
        text = text[:4000] + " ...(truncated)"
    return _cached_embedding(text, model)

def get_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Fetch embeddings for multiple texts in a single API call (batch processing).
    
    Args:
        texts: List of text strings to embed.
        model: OpenAI embedding model.
    
    Returns:
        List of embedding vectors.
    """
    client = _get_client()
    # Truncate long texts
    clean_texts = [(t[:4000] + " ...(truncated)" if t and len(t) > 4000 else t) for t in texts]
    clean_texts = [(t or '').replace('\n', ' ').strip() for t in clean_texts]
    
    for attempt in range(4):
        try:
            response = client.embeddings.create(input=clean_texts, model=model)
            return [data.embedding for data in response.data]
        except Exception as e:
            if attempt == 3:
                logger.error(f"Batch embedding failed permanently: {e}")
                raise
            backoff = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"Batch embedding retry {attempt+1}: {e} (sleep {backoff:.1f}s)")
            time.sleep(backoff)
    return []  # Should never reach here

def generate_embeddings(df: pd.DataFrame, content_column: str = 'ESSENTIAL_CONTENT', sample_size: int | None = 100, batch_size: int = 32) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate embeddings for DataFrame rows with batch API & progress logging.

    Args:
        df: Source DataFrame.
        content_column: Column containing text to embed.
        sample_size: If provided, only embed first N rows.
        batch_size: Number of texts per batch API call.
    Returns:
        (DataFrame with embedding column, numpy array of embeddings)
    """
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found for embedding")
    work_df = df.head(sample_size) if sample_size else df.copy()
    total = len(work_df)
    texts = work_df[content_column].tolist()
    all_embeddings: List[List[float]] = []
    
    logger.info(f"Starting batch embedding generation for {total} rows (batch_size={batch_size})...")
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_embs = get_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embs)
        except Exception as e:
            logger.error(f"Batch {i//batch_size} failed: {e}. Using zeros.")
            all_embeddings.extend([[]] * len(batch_texts))
        
        if (i // batch_size) % 5 == 0:
            logger.info(f"Embedding progress: {min(i+batch_size, total)}/{total} ({min(i+batch_size, total)/total*100:.1f}%)")
    
    work_df = work_df.copy()
    work_df['embedding'] = all_embeddings
    valid = [e for e in all_embeddings if e]
    emb_matrix = np.array(valid) if valid else np.zeros((0, 0))
    logger.info(f"✅ Embeddings generated: {emb_matrix.shape}")
    return work_df, emb_matrix
