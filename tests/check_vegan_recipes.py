"""
비건 레시피 존재 여부 확인 스크립트
"""
import pandas as pd
from pathlib import Path

# 데이터 로드 (CSV 사용)
project_root = Path(__file__).parent.parent
DATA_PATH = project_root / "data" / "TB_RECIPE_SEARCH_241226.csv"
print(f"데이터 파일: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except:
    df = pd.read_csv(DATA_PATH, encoding='cp949')

print(f"총 레시피 개수: {len(df)}")
print(f"컬럼: {df.columns.tolist()}\n")

# 비건 제외 재료 (확장 - workflow와 동일)
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

# 단백질 관련 키워드
protein_keywords = ['단백질', '두부', '콩', '된장', '청국장', '비지', '렌즈콩', '병아리콩', '퀴노아', '템페', '세이탄']

print("=" * 80)
print("1. 비건 가능 레시피 찾기 (동물성 재료 제외)")
print("=" * 80)

# 제목 + 재료 내용에서 제외 재료 체크
df['is_vegan'] = df.apply(
    lambda row: not any(
        excluded in str(row.get('RCP_TTL', '')) + str(row.get('CKG_MTRL_CN', '')) + str(row.get('CKG_IPDC', ''))
        for excluded in vegan_exclude
    ),
    axis=1
)

vegan_recipes = df[df['is_vegan']]
print(f"\n비건 가능 레시피: {len(vegan_recipes)}개 / {len(df)}개 ({len(vegan_recipes)/len(df)*100:.1f}%)")

print("\n[샘플 비건 레시피 10개]")
for idx, row in vegan_recipes.head(10).iterrows():
    print(f"  - {row.get('RCP_TTL', 'N/A')}")

print("\n" + "=" * 80)
print("2. 비건 + 단백질 레시피 찾기")
print("=" * 80)

# 단백질 키워드 포함
vegan_protein = vegan_recipes[
    vegan_recipes.apply(
        lambda row: any(
            keyword in str(row.get('RCP_TTL', '')) + str(row.get('CKG_MTRL_CN', '')) + str(row.get('CKG_IPDC', ''))
            for keyword in protein_keywords
        ),
        axis=1
    )
]

print(f"\n비건 + 단백질 레시피: {len(vegan_protein)}개")

if len(vegan_protein) > 0:
    print("\n[비건 단백질 레시피 - 문제 식단 점검]")
    problematic = []
    
    for idx, row in vegan_protein.head(50).iterrows():
        title = row.get('RCP_TTL', 'N/A')
        materials = str(row.get('CKG_MTRL_CN', ''))
        
        # 금지 재료 체크
        found_forbidden = [v for v in vegan_exclude if v in title + materials]
        
        if found_forbidden:
            problematic.append({
                'title': title,
                'forbidden': found_forbidden,
                'materials': materials[:100]
            })
    
    if problematic:
        print(f"\n❌ 문제 발견! {len(problematic)}개 레시피에 금지 재료 포함:")
        for p in problematic[:10]:
            print(f"\n  제목: {p['title']}")
            print(f"  금지재료: {', '.join(p['forbidden'])}")
            print(f"  재료: {p['materials']}...")
    else:
        print(f"\n✅ 검사 완료: 상위 50개 레시피 모두 비건 기준 충족")
    
    print("\n[정상 비건 단백질 레시피 샘플 20개]")
    clean_count = 0
    for idx, row in vegan_protein.iterrows():
        if clean_count >= 20:
            break
        
        title = row.get('RCP_TTL', 'N/A')
        materials = str(row.get('CKG_MTRL_CN', ''))
        
        # 금지 재료 없는지 재확인
        if not any(v in title + materials for v in vegan_exclude):
            print(f"  ✅ {title}")
            print(f"     재료: {materials[:80]}...")
            clean_count += 1
    
    if clean_count == 0:
        print("  ❌ 정상 비건 레시피를 찾을 수 없습니다!")
else:
    print("\n❌ 비건 + 단백질 레시피가 없습니다!")

print("\n" + "=" * 80)
print("3. 두부 레시피 찾기 (가장 일반적인 비건 단백질)")
print("=" * 80)

tofu_recipes = vegan_recipes[
    vegan_recipes.apply(
        lambda row: '두부' in str(row.get('RCP_TTL', '')) + str(row.get('CKG_MTRL_CN', '')),
        axis=1
    )
]

print(f"\n비건 두부 레시피: {len(tofu_recipes)}개")

if len(tofu_recipes) > 0:
    print("\n[두부 레시피 샘플]")
    for idx, row in tofu_recipes.head(10).iterrows():
        print(f"  - {row.get('RCP_TTL', 'N/A')}")

print("\n" + "=" * 80)
print("4. 콩 레시피 찾기")
print("=" * 80)

bean_recipes = vegan_recipes[
    vegan_recipes.apply(
        lambda row: any(
            keyword in str(row.get('RCP_TTL', '')) + str(row.get('CKG_MTRL_CN', ''))
            for keyword in ['콩', '된장', '청국장', '비지']
        ),
        axis=1
    )
]

print(f"\n비건 콩 레시피: {len(bean_recipes)}개")

if len(bean_recipes) > 0:
    print("\n[콩 레시피 샘플]")
    for idx, row in bean_recipes.head(10).iterrows():
        print(f"  - {row.get('RCP_TTL', 'N/A')}")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

if len(vegan_protein) > 0:
    print(f"✅ 질문 가능: 비건 단백질 레시피 {len(vegan_protein)}개 존재")
    print(f"   - 두부: {len(tofu_recipes)}개")
    print(f"   - 콩/된장/청국장: {len(bean_recipes)}개")
elif len(tofu_recipes) > 0 or len(bean_recipes) > 0:
    print(f"⚠️ 부분 가능: 직접적인 '단백질' 키워드는 없지만 비건 단백질 재료 사용")
    print(f"   - 두부: {len(tofu_recipes)}개")
    print(f"   - 콩/된장/청국장: {len(bean_recipes)}개")
else:
    print("❌ 질문 불가능: 비건 단백질 레시피가 데이터에 없음")
