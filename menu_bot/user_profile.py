"""User profile and filtering utilities for personalized menu recommendations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class UserProfile:
    allergies: List[str] = field(default_factory=list)  # ingredients to avoid
    disliked_flavors: List[str] = field(default_factory=list)  # e.g., 매운, 달콤한
    preferred_flavors: List[str] = field(default_factory=list)  # boost ranking
    diet: str | None = None  # e.g., 'vegan', 'low_salt'

    def to_dict(self) -> Dict:
        return {
            'allergies': self.allergies,
            'disliked_flavors': self.disliked_flavors,
            'preferred_flavors': self.preferred_flavors,
            'diet': self.diet,
        }

FLAVOR_KEYWORDS = {
    'spicy': ['매운', '고추', '청양'],
    'sweet': ['달콤', '꿀', '설탕', '초코'],
    'savory': ['감칠', '간장', '버터'],
    'light': ['담백', '부드러운'],
}

DIET_EXCLUDE = {
    'vegan': ['고기', '쇠고기', '돼지고기', '닭고기', '달걀', '치즈', '버터', '우유', '생선', '해산물'],
    'low_salt': ['소금', '간장', '액젓', '젓갈'],
}

def _contains_any(text: str, tokens: List[str]) -> bool:
    return any(t for t in tokens if t and t in text)

def filter_results_by_profile(results: List[Dict[str, str]], profile: UserProfile) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for r in results:
        content = (r.get('content') or '')
        # Allergy exclusion
        if _contains_any(content, profile.allergies):
            continue
        # Diet exclusion
        if profile.diet and profile.diet in DIET_EXCLUDE:
            if _contains_any(content, DIET_EXCLUDE[profile.diet]):
                continue
        # Disliked flavor exclusion
        if _contains_any(content, profile.disliked_flavors):
            continue
        filtered.append(r)
    # Simple flavor preference boosting: reorder if preferred flavor tokens appear
    def preference_score(rec):
        content = rec.get('content', '')
        score = 0
        for pf in profile.preferred_flavors:
            if pf in content:
                score += 1
        return -score  # negative for ascending sort to put higher score first
    filtered.sort(key=preference_score)
    return filtered
