"""코드 무결성 및 호환성 검사 스크립트"""
import sys
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("menu_bot_phase1 코드 무결성 검사")
print("=" * 70)

# 검사할 모듈 목록
modules_to_test = [
    "data_loader",
    "embedding_utils",
    "faiss_index",
    "sentiment_module",
    "user_profile",
    "recipe_retriever",
    "recipe_tools",
    "recommendation_chains",
    "recommendation_workflow",
    "memory_manager",
    "agent_system",
]

errors = []
warnings = []
passed = []

for module_name in modules_to_test:
    print(f"\n[검사] {module_name}.py")
    try:
        # Import 테스트
        module = __import__(module_name)
        print(f"  ✅ Import 성공")
        
        # 주요 함수/클래스 존재 확인
        if module_name == "data_loader":
            assert hasattr(module, 'load_recipe_data')
            assert hasattr(module, 'extract_essential_info')
            print(f"  ✅ 필수 함수 확인")
            
        elif module_name == "embedding_utils":
            assert hasattr(module, 'get_embedding')
            assert hasattr(module, 'generate_embeddings')
            print(f"  ✅ 필수 함수 확인")
            
        elif module_name == "recipe_retriever":
            assert hasattr(module, 'FAISSRecipeRetriever')
            assert hasattr(module, 'create_recipe_retriever')
            print(f"  ✅ 필수 클래스/함수 확인")
            
        elif module_name == "recipe_tools":
            assert hasattr(module, 'RecipeSearchTool')
            assert hasattr(module, 'SentimentAnalysisTool')
            assert hasattr(module, 'create_tools')
            print(f"  ✅ 필수 클래스/함수 확인")
            
        elif module_name == "recommendation_chains":
            assert hasattr(module, 'create_recommendation_chain')
            assert hasattr(module, 'create_simple_rag_chain')
            print(f"  ✅ 필수 함수 확인")
            
        elif module_name == "recommendation_workflow":
            assert hasattr(module, 'create_recommendation_graph')
            assert hasattr(module, 'run_recommendation_workflow')
            assert hasattr(module, 'RecommendationState')
            print(f"  ✅ 필수 함수/클래스 확인")
            
        elif module_name == "memory_manager":
            assert hasattr(module, 'get_session_id')
            assert hasattr(module, 'create_memory')
            assert hasattr(module, 'add_user_message')
            assert hasattr(module, 'add_ai_message')
            print(f"  ✅ 필수 함수 확인")
            
        elif module_name == "agent_system":
            assert hasattr(module, 'create_recommendation_agent')
            assert hasattr(module, 'run_agent_recommendation')
            assert hasattr(module, 'AGENTS_AVAILABLE')
            if module.AGENTS_AVAILABLE:
                print(f"  ✅ langchain.agents 사용 가능")
            else:
                warnings.append(f"{module_name}: langchain.agents 미설치")
                print(f"  ⚠️  langchain.agents 미설치 (선택사항)")
            print(f"  ✅ 필수 함수 확인")
        
        passed.append(module_name)
        
    except ImportError as e:
        errors.append(f"{module_name}: Import 실패 - {e}")
        print(f"  ❌ Import 실패: {e}")
    except AssertionError as e:
        errors.append(f"{module_name}: 필수 함수/클래스 누락")
        print(f"  ❌ 필수 함수/클래스 누락")
    except Exception as e:
        errors.append(f"{module_name}: 예상치 못한 오류 - {e}")
        print(f"  ❌ 오류: {e}")

# 요약
print("\n" + "=" * 70)
print("검사 요약")
print("=" * 70)
print(f"✅ 통과: {len(passed)}/{len(modules_to_test)}")
print(f"⚠️  경고: {len(warnings)}")
print(f"❌ 오류: {len(errors)}")

if warnings:
    print("\n경고 목록:")
    for w in warnings:
        print(f"  - {w}")

if errors:
    print("\n오류 목록:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n🎉 모든 모듈이 정상적으로 작동합니다!")
    print("\n다음 단계:")
    print("  1. streamlit run app.py --server.port=8502")
    print("  2. Agent 모드 사용 시: pip install langchain")
    sys.exit(0)
