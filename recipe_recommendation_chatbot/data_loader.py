"""Data loading utilities (compatible with original menu_bot)."""
import pandas as pd

def load_recipe_data(file_path: str) -> pd.DataFrame:
    """Load recipe CSV and build unified text column.

    Args:
        file_path: Path to CSV file.
    Returns:
        DataFrame with RECIPE_CONTENT column.
    """
    df = pd.read_csv(file_path)
    df['RECIPE_CONTENT'] = (
        df['RCP_TTL'].fillna('') + '. ' +
        df['CKG_MTRL_ACTO_NM'].fillna('') + ' 재료. ' +
        df['CKG_IPDC'].fillna('') + ' 상세 설명. ' +
        '재료 목록: ' + df['CKG_MTRL_CN'].fillna('')
    )
    return df

def create_essential_content(row: pd.Series) -> str:
    """Extract essential info (title + category + truncated ingredients).

    Args:
        row: Row of DataFrame.
    Returns:
        Essential info string.
    """
    title = str(row.get('RCP_TTL', '') or '')
    category = str(row.get('CKG_MTRL_ACTO_NM', '') or '')
    ingredients_raw = row.get('CKG_MTRL_CN', '')
    ingredients = str(ingredients_raw if pd.notna(ingredients_raw) else '')[:200]
    return f"{title}. {category} 요리. 주재료: {ingredients}"

def extract_essential_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add essential content column.

    Args:
        df: Recipe DataFrame with RECIPE_CONTENT.
    Returns:
        DataFrame with ESSENTIAL_CONTENT.
    """
    df['ESSENTIAL_CONTENT'] = df.apply(create_essential_content, axis=1)
    return df
