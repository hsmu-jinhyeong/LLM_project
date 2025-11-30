"""Automatic evaluation script for recommendations using synthetic Q&A dataset.

Metrics:
- keyword_hit_rate: fraction of gold keywords appearing in model response
- recipe_count (heuristic): number of lines starting with bullet or numbering
- sentiment_usage: presence of sentiment-related Korean words when expected

NOTE: This script assumes OpenAI API access; replace model calls with mocks for offline tests.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

# Import search pipeline pieces
from menu_bot import (
    load_recipe_data,
    extract_essential_info,
    generate_embeddings,
    create_faiss_index,
)
from menu_bot.search import search_recipes as core_search
from menu_bot.sentiment_module import get_user_sentiment  # type: ignore

DATA_PATH = "data/TB_RECIPE_SEARCH_241226.csv"
SYN_PATH = "data/synthetic_qna.jsonl"
MODEL = "gpt-4o"

client = None
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

def _client():
    global client
    if client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=key)
    return client

def load_synthetic_cases(path: str | Path) -> List[Dict[str, Any]]:
    cases = []
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Synthetic dataset not found: {path_obj}. Returning empty list.")
        return cases
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    logger.info(f"Loaded {len(cases)} synthetic cases from {path_obj}")
    return cases

def build_search(sample_size: int = 200):
    df = load_recipe_data(DATA_PATH)
    df = extract_essential_info(df)
    sample_df, emb_matrix = generate_embeddings(df, sample_size=sample_size)
    index = create_faiss_index(emb_matrix)
    def search_fn(query: str, top_k: int = 3):
        return core_search(query, index, sample_df, top_k=top_k)
    return search_fn

SENTIMENT_WORDS = ["기분", "감성", "긍정", "부정", "중립"]

# -------------------- Evaluation --------------------

def _check_time_constraint(case: Dict[str, Any], response_text: str) -> bool:
    limit = case.get('constraints', {}).get('cook_time_max')
    if not limit:
        return True
    # naive heuristic: look for number <= limit followed by 분
    import re
    times = re.findall(r"(\d{1,3})\s*분", response_text)
    return any(int(t) <= int(limit) for t in times) or ("빠른" in response_text or "간단" in response_text)

def _check_allergy(case: Dict[str, Any], response_text: str) -> bool:
    allergy = case.get('constraints', {}).get('allergy') or case.get('constraints', {}).get('allergies')
    if not allergy:
        return True
    return allergy not in response_text

def _check_diet(case: Dict[str, Any], response_text: str) -> bool:
    diet = case.get('constraints', {}).get('diet')
    if not diet:
        return True
    if diet == 'vegan':
        forbidden = ['고기','쇠고기','돼지고기','닭','생선','해산물','치즈','버터','달걀','우유']
        return not any(f in response_text for f in forbidden)
    if diet == 'low_salt':
        salty = ['소금','간장','액젓','젓갈']
        return not any(s in response_text for s in salty)
    return True

def _check_preferred_flavor(case: Dict[str, Any], response_text: str) -> bool:
    flavor = case.get('constraints', {}).get('flavor')
    if not flavor:
        return True
    return flavor in response_text or {
        'sweet': '달콤',
        'spicy': '매운',
        'light': '담백',
    }.get(flavor, '') in response_text

def evaluate_case(case: Dict[str, Any], search_fn) -> Dict[str, Any]:
    user_input = case['user_input']
    expected_tools = case.get('expected_tools', [])
    sentiment_data = get_user_sentiment(user_input) if 'get_user_sentiment' in expected_tools else None
    search_query = user_input
    if sentiment_data:
        search_query += f" ({sentiment_data['description']})"
    retrieved = search_fn(search_query, top_k=3)
    # Build prompt quickly
    context_block = "\n".join([f"제목: {r['title']} 내용: {r['content'][:120]}" for r in retrieved])
    sentiment_line = f"사용자 감성: {sentiment_data['description']}" if sentiment_data else ""
    system_prompt = f"당신은 맞춤 메뉴 추천 전문가입니다. {sentiment_line}\n관련 레시피:\n{context_block}\n사용자 요청에 맞는 메뉴를 1~2개 추천하고 이유를 설명하세요."
    response = _client().chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_input}])
    text = response.choices[0].message.content

    # keyword hits
    gold = case.get('gold_keywords', [])
    hits = sum(1 for kw in gold if kw in text)
    keyword_hit_rate = hits / max(1, len(gold))

    # recipe count heuristic
    recipe_lines = [l for l in text.splitlines() if re.match(r"^\d+\.|^-", l.strip())]
    recipe_count = len(recipe_lines) or (1 if any(word in text for word in ["추천", "요리", "메뉴"]) else 0)

    # sentiment usage check
    sentiment_used = any(w in text for w in SENTIMENT_WORDS) if sentiment_data else True

    return {
        "intent": case['intent'],
        "keyword_hit_rate": keyword_hit_rate,
        "recipe_count": recipe_count,
        "sentiment_used": sentiment_used,
        "time_constraint_ok": _check_time_constraint(case, text),
        "allergy_ok": _check_allergy(case, text),
        "diet_ok": _check_diet(case, text),
        "preferred_flavor_ok": _check_preferred_flavor(case, text),
        "raw_response": text[:500],
    }

def run_evaluation():
    cases = load_synthetic_cases(SYN_PATH)
    search_fn = build_search(sample_size=150)
    results = [evaluate_case(c, search_fn) for c in cases]
    # Aggregate
    avg_hit = sum(r['keyword_hit_rate'] for r in results) / len(results)
    avg_recipes = sum(r['recipe_count'] for r in results) / len(results)
    sentiment_pass = sum(1 for r in results if r['sentiment_used']) / len(results)
    time_ok_rate = sum(1 for r in results if r['time_constraint_ok']) / len(results)
    allergy_ok_rate = sum(1 for r in results if r['allergy_ok']) / len(results)
    diet_ok_rate = sum(1 for r in results if r['diet_ok']) / len(results)
    flavor_ok_rate = sum(1 for r in results if r['preferred_flavor_ok']) / len(results)
    print("\n=== Evaluation Summary ===")
    print(f"Cases: {len(results)}")
    print(f"Avg keyword hit rate: {avg_hit:.2f}")
    print(f"Avg recipe count: {avg_recipes:.2f}")
    print(f"Sentiment usage rate: {sentiment_pass:.2f}")
    print(f"Time constraint satisfaction: {time_ok_rate:.2f}")
    print(f"Allergy avoidance rate: {allergy_ok_rate:.2f}")
    print(f"Diet compliance rate: {diet_ok_rate:.2f}")
    print(f"Preferred flavor alignment: {flavor_ok_rate:.2f}")
    print("\nPer-case (truncated):")
    for r in results:
        print(f"- {r['intent']}: hit={r['keyword_hit_rate']:.2f}, recipes={r['recipe_count']}, sentiment={r['sentiment_used']}, time={r['time_constraint_ok']}, allergy={r['allergy_ok']}, diet={r['diet_ok']}, flavor={r['preferred_flavor_ok']}")

if __name__ == "__main__":
    run_evaluation()
