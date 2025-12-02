import faiss
import numpy as np

# -------------------- FAISS Index Utilities --------------------

def create_faiss_index(embeddings: np.ndarray, save_path: str | None = None) -> faiss.IndexFlatL2:
    """Create and optionally save a FAISS FlatL2 index.

    Args:
        embeddings: Numpy array shape (N, D) float32 or convertible.
        save_path: Optional path to save index.
    Returns:
        FAISS index instance.
    """
    if embeddings.dtype != 'float32':
        embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS index ready | vectors: {index.ntotal} | dim: {dimension}")
    if save_path:
        faiss.write_index(index, save_path)
        print(f"💾 Saved index → {save_path}")
    return index

def load_faiss_index(load_path: str) -> faiss.IndexFlatL2:
    """Load a previously saved FAISS index.

    Args:
        load_path: Path to index file.
    Returns:
        Loaded FAISS index.
    """
    index = faiss.read_index(load_path)
    print(f"✅ Loaded index | vectors: {index.ntotal}")
    return index
