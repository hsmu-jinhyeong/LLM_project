"""LangChain-based retriever for recipe search using FAISS vector store."""
from __future__ import annotations
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import logging

logger = logging.getLogger("recipe_retriever")


class FAISSRecipeRetriever(BaseRetriever):
    """Custom LangChain retriever wrapping FAISS index for recipe search.
    
    This retriever integrates with LangChain's retrieval interface while
    maintaining compatibility with the original FAISS index structure.
    """
    
    vectorstore: LangChainFAISS
    top_k: int = 3
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """Retrieve relevant recipe documents.
        
        Args:
            query: User query string.
            run_manager: Callback manager (optional).
        
        Returns:
            List of Document objects with recipe content.
        """
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
        return docs


def create_recipe_retriever(
    df: pd.DataFrame,
    embeddings_array: np.ndarray | None = None,
    embedding_column: str = 'embedding',
    content_column: str = 'ESSENTIAL_CONTENT',
    title_column: str = 'RCP_TTL',
    top_k: int = 3,
) -> FAISSRecipeRetriever:
    """Create a LangChain-compatible FAISS retriever from recipe DataFrame.
    
    Args:
        df: Recipe DataFrame with embeddings.
        embeddings_array: Optional pre-computed embeddings (N, D).
        embedding_column: Column name containing embedding lists.
        content_column: Column name for recipe content.
        title_column: Column name for recipe title.
        top_k: Number of results to retrieve.
    
    Returns:
        FAISSRecipeRetriever instance.
    """
    # Extract embeddings
    if embeddings_array is not None:
        embeddings = embeddings_array.astype('float32')
    elif embedding_column in df.columns:
        embeddings = np.array(df[embedding_column].tolist(), dtype='float32')
    else:
        raise ValueError(f"No embeddings found. Provide embeddings_array or {embedding_column} column.")
    
    # Create documents
    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=row.get(content_column, ''),
            metadata={
                'title': row.get(title_column, ''),
                'recipe_id': row.get('RCP_SNO', idx),
                'category': row.get('CKG_MTRL_ACTO_NM', ''),
            }
        )
        documents.append(doc)
    
    # Build FAISS index using LangChain wrapper
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Create docstore mapping
    index_to_id = {i: str(i) for i in range(len(documents))}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    
    # Initialize OpenAI embeddings for query embedding
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create LangChain FAISS vectorstore
    vectorstore = LangChainFAISS(
        embedding_function=openai_embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_id,
    )
    
    logger.info(f"✅ Created FAISS retriever with {len(documents)} documents (dim={dimension})")
    
    return FAISSRecipeRetriever(vectorstore=vectorstore, top_k=top_k)
