"""Full dataset embedding generation and index persistence script.

This script:
- Loads the entire recipe CSV (23,192 rows)
- Generates embeddings for all recipes with batch optimization
- Creates FAISS index
- Saves index to disk for reuse
- Provides progress tracking and ETA estimation

Usage:
    python scripts/build_full_index.py [--batch-size 32] [--output data/recipe_full.index]

NOTE: Requires OPENAI_API_KEY environment variable.
Estimated time: 3-10 minutes with batch processing, ~$0.06 API cost.
"""
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "menu_bot_phase1"))

from data_loader import load_recipe_data, extract_essential_info
from faiss_index import create_faiss_index
from embedding_utils import get_embeddings_batch

# Default paths
DEFAULT_DATA_PATH = str(project_root / "data" / "TB_RECIPE_SEARCH_241226.csv")
DEFAULT_INDEX_PATH = str(project_root / "data" / "recipe_full.index")
DEFAULT_BATCH_SIZE = 32

load_dotenv(project_root / ".env")


def generate_embeddings_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings with batch processing and progress tracking.
    
    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts per API call (OpenAI supports batching).
    
    Returns:
        Numpy array of embeddings (N, 1536).
    """
    total = len(texts)
    embeddings = []
    start_time = time.time()
    
    print(f"🚀 Starting embedding generation for {total:,} recipes...")
    print(f"   Batch size: {batch_size} | Model: text-embedding-3-small")
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_start = time.time()
        
        # Generate embeddings using TRUE batch API (single API call per batch)
        try:
            batch_embeddings = get_embeddings_batch(batch_texts)
            if len(batch_embeddings) != len(batch_texts):
                print(f"⚠️  Batch {i//batch_size}: Expected {len(batch_texts)} embeddings, got {len(batch_embeddings)}")
                # Pad with zeros if mismatch
                while len(batch_embeddings) < len(batch_texts):
                    batch_embeddings.append([0.0] * 1536)
        except Exception as e:
            print(f"⚠️  Batch {i//batch_size} failed: {e}. Using zero vectors.")
            batch_embeddings = [[0.0] * 1536] * len(batch_texts)
        
        embeddings.extend(batch_embeddings)
        batch_time = time.time() - batch_start
        
        # Progress tracking
        processed = min(i + batch_size, total)
        progress_pct = (processed / total) * 100
        elapsed = time.time() - start_time
        
        if processed > 0:
            avg_per_item = elapsed / processed
            remaining = total - processed
            eta_seconds = avg_per_item * remaining
            eta_minutes = eta_seconds / 60
            
            print(f"   [{processed:,}/{total:,}] {progress_pct:.1f}% | "
                  f"Batch: {batch_time:.1f}s | ETA: {eta_minutes:.1f}m")
    
    total_time = time.time() - start_time
    print(f"✅ Embedding generation complete in {total_time/60:.1f} minutes")
    
    return np.array(embeddings, dtype='float32')


def filter_empty_embeddings(embeddings: np.ndarray, df) -> tuple[np.ndarray, any]:
    """Remove rows with zero/empty embeddings.
    
    Args:
        embeddings: Full embedding matrix.
        df: Corresponding DataFrame.
    
    Returns:
        (filtered_embeddings, filtered_df)
    """
    # Check for zero vectors (failed embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    valid_mask = norms > 0.0
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"⚠️  Filtering {invalid_count} empty/failed embeddings")
    
    return embeddings[valid_mask], df[valid_mask].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Generate full recipe embeddings and persist FAISS index")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to recipe CSV")
    parser.add_argument("--output", default=DEFAULT_INDEX_PATH, help="Output index file path")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--no-filter", action="store_true", help="Skip empty embedding filtering")
    
    args = parser.parse_args()
    
    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it in .env file or export OPENAI_API_KEY=your-key")
        sys.exit(1)
    
    # Validate data file
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Full Recipe Index Builder")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Step 1: Load data
    print("[1/5] Loading recipe data...")
    df = load_recipe_data(str(data_path))
    df = extract_essential_info(df)
    print(f"      Loaded {len(df):,} recipes")
    
    # Step 2: Extract texts
    print("[2/5] Extracting embedding texts...")
    texts = df['ESSENTIAL_CONTENT'].tolist()
    print(f"      Prepared {len(texts):,} texts")
    
    # Step 3: Generate embeddings
    print("[3/5] Generating embeddings (this may take several minutes)...")
    embeddings = generate_embeddings_batch(texts, batch_size=args.batch_size)
    print(f"      Generated embeddings shape: {embeddings.shape}")
    
    # Step 4: Filter empty embeddings
    if not args.no_filter:
        print("[4/5] Filtering empty embeddings...")
        embeddings, df = filter_empty_embeddings(embeddings, df)
        print(f"      Valid embeddings: {embeddings.shape[0]:,}")
    else:
        print("[4/5] Skipping empty embedding filter (--no-filter)")
    
    # Step 5: Create and save index
    print("[5/5] Creating FAISS index and saving to disk...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    index = create_faiss_index(embeddings, save_path=str(output_path))
    
    # Save filtered DataFrame for later use
    df_output = output_path.with_suffix('.parquet')
    df.to_parquet(df_output, index=False)
    print(f"💾 Saved filtered DataFrame → {df_output}")
    
    # Summary
    print()
    print("=" * 60)
    print("✅ Index build complete!")
    print("=" * 60)
    print(f"Index file: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Data file: {df_output}")
    print(f"Vectors: {index.ntotal:,}")
    print(f"Dimensions: {index.d}")
    print()
    print("To use this index in your application:")
    print(f"  from menu_bot import load_faiss_index")
    print(f"  index = load_faiss_index('{output_path}')")
    print(f"  df = pd.read_parquet('{df_output}')")


if __name__ == "__main__":
    main()
