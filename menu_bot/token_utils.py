import pandas as pd
try:
    import tiktoken  # optional dependency
except Exception:
    tiktoken = None

# -------------------- Token Utilities --------------------

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens for given text using tiktoken.

    Args:
        text: Input text.
        model: Reference model for encoding (respected).
    Returns:
        Number of tokens.
    """
    if tiktoken is None:
        raise ModuleNotFoundError("tiktoken is not installed. Install it or avoid token counting.")
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text or ""))

def analyze_token_statistics(df: pd.DataFrame, content_column: str = 'RECIPE_CONTENT', model: str = "gpt-3.5-turbo") -> pd.DataFrame:
    """Compute token stats and attach token_count column.

    Args:
        df: DataFrame.
        content_column: Column name whose tokens to count.
        model: Model whose tokenizer to use.
    Returns:
        DataFrame with token_count column.
    """
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not in DataFrame")
    if tiktoken is None:
        raise ModuleNotFoundError("tiktoken is not installed. Install it to analyze token statistics.")
    df['token_count'] = df[content_column].apply(lambda x: count_tokens(x, model=model))
    print("📊 Token Stats:")
    print(f"Avg: {df['token_count'].mean():.0f}")
    print(f"Max: {df['token_count'].max()}")
    print(f"Min: {df['token_count'].min()}")
    over_limit = (df['token_count'] > 8191).sum()
    if over_limit:
        print(f"⚠️ Over 8,191 tokens: {over_limit}")
    return df
