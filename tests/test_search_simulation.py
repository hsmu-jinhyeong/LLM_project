"""
실제 검색 시뮬레이션: 비건 단백질 검색이 작동하는지 확인
"""
import sys
import os
from pathlib import Path

# 환경 변수 로드
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# API 키 확인
if not os.getenv('OPENAI_API_KEY'):
    print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다!")
    print("   .env 파일을 확인하거나 환경 변수를 설정하세요.")
    sys.exit(1)

sys.path.insert(0, str(project_root / "menu_bot_phase1"))

from recipe_retriever import create_recipe_retriever
import pandas as pd
import numpy as np

# 1. 데이터 로드
DATA_PATH = project_root / "data" / "recipe_full_with_embeddings.parquet"
print(f"데이터 로딩 시도: {DATA_PATH}")

try:
    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Parquet 로드 성공: {len(df)}개 레시피\n")
except Exception as e:
    print(f"❌ Parquet 로드 실패: {e}")
    print("대체 방법으로 CSV 사용...\n")
    df = pd.read_csv("data/TB_RECIPE_SEARCH_241226.csv", encoding='utf-8')
    print(f"✅ CSV 로드 성공: {len(df)}개 레시피\n")

# 2. 임베딩 확인
if 'embedding' in df.columns:
    print(f"✅ 임베딩 컬럼 존재")
    embeddings = np.array(df['embedding'].tolist(), dtype='float32')
    print(f"   임베딩 shape: {embeddings.shape}")
else:
    print(f"❌ 임베딩 컬럼 없음! 컬럼: {df.columns.tolist()[:10]}...")
    print("   임베딩 없이 검색 불가능")
    sys.exit(1)

# 3. Retriever 생성
print(f"\n{'='*60}")
print("Retriever 생성 중...")
print(f"{'='*60}")

try:
    retriever = create_recipe_retriever(
        df=df,
        embeddings_array=embeddings,
        top_k=10
    )
    print(f"✅ Retriever 생성 성공\n")
except Exception as e:
    print(f"❌ Retriever 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 검색 테스트
test_queries = [
    "비건 식단인데 단백질 보충할 수 있는 요리 추천해줘",
    "두부 요리",
    "콩나물 레시피",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"검색어: {query}")
    print(f"{'='*60}")
    
    try:
        # 비건 키워드 감지
        is_vegan_query = any(k in query.lower() for k in ['비건', 'vegan', '채식'])
        
        # 쿼리 개선: 비건 + 단백질이면 식물성 단백질 키워드 추가
        enhanced_query = query
        if is_vegan_query and any(k in query.lower() for k in ['단백질', '영양']):
            enhanced_query = f"{query} 두부 콩 견과류"
            print(f"🔍 쿼리 강화: {enhanced_query}\n")
        
        search_k = 50 if is_vegan_query else 10
        
        # vectorstore 직접 사용 (workflow와 동일)
        docs = retriever.vectorstore.similarity_search(enhanced_query, k=search_k)
        
        print(f"✅ 검색 성공: {len(docs)}개 결과 (k={search_k})\n")
        
        # 비건 필터링
        vegan_exclude = [
            # 육류
            '고기', '쇠고기', '돼지고기', '닭고기', '닭', '양고기', '오리고기', '삼격살', '목살', '항정살', '등심', '안심', '갈비', '차돌', '사태', '양지', '우삼격',
            # 달걀/유제품
            '달걀', '계란', '치즈', '버터', '우유', '크림', '요구르트', '생크림', '노른자', '흰자',
            # 해산물 (전체)
            '생선', '해산물', '다슬기', '굴', '조개', '새우', '게', '오징어', '낙지', '문어', '주꾸미',
            '고등어', '갈치', '꽁치', '참치', '연어', '광어', '우럭', '조기', '멸치', '북어', '명태', '대구', '동태',
            '조갯살', '홍합', '바지락', '가리비', '전복', '소라', '해물', '어묵', '오덱', '골뱅이',
            # 특수 동물성
            '선지', '곱창', '막창', '명란', '창란', '알탕', '젓갈', '까나리', '액젓',
            # 가공육
            '베이컨', '소시지', '햄', '스팸', '육포', '베컨'
        ]
        
        filtered = []
        for doc in docs:
            title = doc.metadata.get('title', '')
            content = doc.page_content
            
            if not any(excluded in title + content for excluded in vegan_exclude):
                filtered.append(doc)
        
        print(f"필터링 후: {len(filtered)}개 (제거: {len(docs) - len(filtered)}개)\n")
        
        if filtered:
            print("비건 레시피:")
            for i, doc in enumerate(filtered[:5], 1):
                print(f"  {i}. {doc.metadata.get('title', 'N/A')}")
        else:
            print("❌ 필터링 후 레시피 없음!")
            print("\n제거된 레시피 (비건 아님):")
            for i, doc in enumerate(docs[:5], 1):
                title = doc.metadata.get('title', 'N/A')
                found = [e for e in vegan_exclude if e in title + doc.page_content]
                print(f"  {i}. {title}")
                if found:
                    print(f"     금지재료: {', '.join(found[:3])}...")
        
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("테스트 완료")
print(f"{'='*60}")
