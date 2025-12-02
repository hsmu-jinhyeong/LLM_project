"""Generate embeddings for full recipe dataset and save to parquet.

Run this script once to create recipe_full_with_embeddings.parquet
which can be loaded instantly without API calls.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from menu_bot_langchain.data_loader import load_recipe_data, extract_essential_info
from menu_bot_langchain.embedding_utils import generate_embeddings

def main():
    print("=" * 60)
    print("🚀 Full Recipe Embedding Generation")
    print("=" * 60)
    
    # Load full data
    parquet_path = PROJECT_ROOT / "data" / "recipe_full.parquet"
    
    if parquet_path.exists():
        print(f"\n📂 Loading from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"\n📂 Loading from CSV...")
        csv_path = PROJECT_ROOT / "data" / "TB_RECIPE_SEARCH_241226.csv"
        df = load_recipe_data(str(csv_path))
    
    print(f"✅ Loaded {len(df):,} recipes")
    
    # Ensure ESSENTIAL_CONTENT exists
    if 'ESSENTIAL_CONTENT' not in df.columns:
        print("\n🔧 Extracting essential content...")
        df = extract_essential_info(df)
    
    # Generate embeddings
    print(f"\n🤖 Generating embeddings for {len(df):,} recipes...")
    print("⚠️  This will take time and use OpenAI API credits!")
    print("💡 Tip: Use smaller batches if you hit rate limits\n")
    
    response = input(f"Continue with {len(df):,} recipes? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return
    
    # Generate in batches
    batch_size = 100  # Adjust based on your rate limits
    all_embeddings = []
    
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        print(f"\n📦 Batch {i//batch_size + 1}: Rows {i} - {end_idx}")
        
        batch_df = df.iloc[i:end_idx].copy()
        _, emb_matrix = generate_embeddings(
            batch_df,
            sample_size=len(batch_df),
            batch_size=32
        )
        all_embeddings.append(emb_matrix)
        
        print(f"✅ Progress: {end_idx}/{len(df)} ({end_idx/len(df)*100:.1f}%)")
    
    # Combine all embeddings
    print("\n🔗 Combining embeddings...")
    full_embeddings = np.vstack(all_embeddings)
    
    # Add to dataframe
    df['embedding'] = list(full_embeddings)
    
    # Save
    output_path = PROJECT_ROOT / "data" / "recipe_full_with_embeddings.parquet"
    print(f"\n💾 Saving to: {output_path}")
    df.to_parquet(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ SUCCESS! Created: recipe_full_with_embeddings.parquet")
    print(f"📊 Size: {len(df):,} recipes with embeddings")
    print(f"📏 Embedding dimension: {full_embeddings.shape[1]}")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("1. Update FULL_DATA_PATH in app.py to use this file")
    print("2. Enjoy instant loading without API calls!")

if __name__ == "__main__":
    main()
