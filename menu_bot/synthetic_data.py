"""Synthetic Q&A dataset generation for menu recommendation bot.
Creates JSONL file with user scenarios for evaluation.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class SyntheticCase:
    user_input: str
    intent: str
    constraints: Dict[str, Any]
    expected_tools: List[str]
    evaluation_criteria: Dict[str, Any]
    gold_keywords: List[str]
    metadata: Dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# Predefined scenarios
SCENARIOS: List[SyntheticCase] = [
    SyntheticCase(
        user_input="오늘 속이 안 좋아서 자극적이지 않고 따뜻한 게 먹고 싶어",
        intent="comfort_food",
        constraints={"spice_level": "mild", "cook_time_max": 30},
        expected_tools=["get_user_sentiment", "search_recipes"],
        evaluation_criteria={"uses_sentiment": True, "recipe_count": 1, "reason_length_max": 300},
        gold_keywords=["죽", "미음", "부드", "자극"],
        metadata={"scenario": "stomach discomfort", "priority": "high"},
    ),
    SyntheticCase(
        user_input="출근 전에 10분 안에 만들 간단한 아침 없을까?",
        intent="quick_breakfast",
        constraints={"cook_time_max": 10, "meal_type": "아침"},
        expected_tools=["search_recipes"],
        evaluation_criteria={"time_constraint_respected": True, "recipe_count": 2},
        gold_keywords=["빵", "시리얼", "토스트", "달걀"],
        metadata={"scenario": "morning rush"},
    ),
    SyntheticCase(
        user_input="기분이 다운돼서 좀 기분 좋아지는 달콤한 디저트 추천해줘",
        intent="mood_uplift",
        constraints={"flavor": "sweet"},
        expected_tools=["get_user_sentiment", "search_recipes"],
        evaluation_criteria={"flavor_alignment": True},
        gold_keywords=["케이크", "초코", "쿠키", "디저트"],
        metadata={"scenario": "low mood"},
    ),
    SyntheticCase(
        user_input="운동 끝났는데 단백질 많은 메뉴 뭐가 좋을까?",
        intent="post_workout",
        constraints={"protein_focus": True},
        expected_tools=["search_recipes"],
        evaluation_criteria={"protein_focus": True},
        gold_keywords=["닭", "두부", "달걀", "단백"],
        metadata={"scenario": "fitness"},
    ),
]

def write_jsonl(path: str | Path, cases: List[SyntheticCase] = SCENARIOS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.to_jsonl() + "\n")
    print(f"✅ Wrote synthetic dataset: {path} ({len(cases)} cases)")
    return path

if __name__ == "__main__":
    write_jsonl("data/synthetic_qna.jsonl")
