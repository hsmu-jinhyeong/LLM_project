"""
빠른 검증 스크립트: 프로필 + 프롬프트 통합 테스트

사이드바 없이 프롬프트만으로 비건 제약이 작동하는지 확인
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "menu_bot_phase1"))

from recommendation_workflow import run_recommendation_workflow, create_recommendation_graph
from recipe_retriever import FAISSRecipeRetriever

# 1. Retriever 초기화
DATA_PATH = project_root / "data" / "recipe_full_with_embeddings.parquet"
retriever = FAISSRecipeRetriever(str(DATA_PATH))

# 2. Graph 생성
graph = create_recommendation_graph(retriever)

# 3. 테스트 케이스
test_cases = [
    {
        "name": "비건 (프롬프트 only)",
        "input": "비건 식단인데 단백질 보충할 수 있는 요리 추천해줘",
        "profile": {}  # 사이드바 비활성화
    },
    {
        "name": "비건 (사이드바 only)",
        "input": "단백질 보충할 수 있는 요리 추천해줘",
        "profile": {"diet": "vegan", "allergies": [], "preferred_flavors": [], "disliked_flavors": []}
    },
    {
        "name": "알레르기 (프롬프트)",
        "input": "달걀 알레르기가 있는데 단백질 요리 추천해줘",
        "profile": {}
    },
    {
        "name": "복합 조건",
        "input": "저염식으로 간단하게 만들 수 있는 한식 추천해줘",
        "profile": {"diet": "vegan", "allergies": ["땅콩"], "preferred_flavors": ["매운맛"], "disliked_flavors": []}
    }
]

# 4. 실행
for idx, test in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"Test {idx}: {test['name']}")
    print(f"{'='*60}")
    print(f"입력: {test['input']}")
    print(f"프로필: {test['profile']}")
    print()
    
    try:
        result = run_recommendation_workflow(graph, test['input'], test['profile'])
        
        print(f"✅ 추천 개수: {len(result.get('recommendations', []))}")
        for rec in result.get('recommendations', []):
            print(f"  - {rec.get('title', 'N/A')}")
            print(f"    이유: {rec.get('reason', 'N/A')[:80]}...")
            
        # 금지 재료 체크
        forbidden = []
        if test['profile'].get('diet') == 'vegan' or '비건' in test['input']:
            forbidden = ['달걀', '고기', '우유', '치즈', '버터', '생선']
        
        if forbidden:
            print("\n  🔍 금지 재료 체크:")
            for rec in result.get('recommendations', []):
                title = rec.get('title', '')
                found = [f for f in forbidden if f in title]
                if found:
                    print(f"    ❌ {title}: {', '.join(found)} 포함!")
                else:
                    print(f"    ✅ {title}: OK")
    
    except Exception as e:
        print(f"❌ 에러: {e}")

print(f"\n{'='*60}")
print("테스트 완료!")
print(f"{'='*60}")
