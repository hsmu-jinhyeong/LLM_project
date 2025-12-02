# 프로젝트 구조 상세

## 📁 디렉토리 구조

```
LLM_project/
├── 📄 README.md                    # 프로젝트 메인 문서
├── 📄 LICENSE                      # 라이선스
├── 📄 .env                         # 환경 변수 (OPENAI_API_KEY)
├── 📄 .gitignore                   # Git 무시 파일
│
├── 📂 menu_bot_phase1/             # ⭐ 메인 애플리케이션
│   ├── app.py                     # Streamlit 웹 앱
│   ├── recommendation_workflow.py  # LangGraph 워크플로우
│   ├── recipe_retriever.py        # FAISS 검색 엔진
│   ├── recipe_tools.py            # LangChain 도구
│   ├── memory_manager.py          # 대화 메모리
│   ├── agent_system.py            # Multi-Agent 시스템
│   ├── user_profile.py            # 사용자 프로필
│   ├── sentiment_module.py        # 감성 분석
│   ├── data_loader.py             # 데이터 로더
│   ├── embedding_utils.py         # 임베딩 유틸
│   ├── faiss_index.py             # FAISS 인덱스
│   ├── recommendation_chains.py   # LangChain 체인
│   ├── requirements.txt           # Python 패키지
│   ├── run_app.bat               # 실행 스크립트 (Windows)
│   ├── README.md                 # Phase 1 상세 가이드
│   ├── ARCHITECTURE.txt          # 기술 아키텍처
│   ├── STREAMLIT_GUIDE.md        # UI 사용법
│   └── scripts/                  # 유틸리티 스크립트
│       ├── build_full_embeddings.py
│       ├── run_examples.py
│       └── validate_code.py
│
├── 📂 data/                        # 데이터 파일
│   ├── TB_RECIPE_SEARCH_241226.csv          # 원본 레시피 (23,192개)
│   ├── recipe_full_with_embeddings.parquet  # 임베딩 포함 데이터
│   └── recipe_full.index                    # FAISS 인덱스 (선택)
│
├── 📂 tests/                       # 테스트 스크립트
│   ├── check_vegan_recipes.py     # 비건 레시피 검증
│   ├── test_search_simulation.py  # 검색 시뮬레이션
│   ├── test_profile_integration.py # 프로필 통합 테스트
│   └── TEST_SCENARIOS.md          # 테스트 시나리오
│
├── 📂 docs/                        # 프로젝트 문서
│   ├── PHASE1_SUMMARY.md          # Phase 1 완료 보고서
│   ├── final_report.md            # 최종 연구 보고서
│   ├── mid-report.md              # 중간 보고서
│   └── reference_project.md       # 참고 프로젝트
│
├── 📂 scripts/                     # 빌드/유틸리티 스크립트
│   └── build_full_index.py        # 임베딩 생성 스크립트
│
└── 📂 archive/                     # 이전 버전 (보관용)
    ├── v1_original/               # 최초 버전
    └── v2_langchain/              # LangChain 초기 버전
        └── menu_bot_langchain/

```

## 🎯 주요 파일 설명

### 실행 파일
- **`menu_bot_phase1/app.py`** (698줄)
  - Streamlit 웹 인터페이스
  - 사용자 프로필 관리
  - LangGraph 워크플로우 실행
  - 비용 모니터링 UI

- **`menu_bot_phase1/run_app.bat`**
  - Windows 환경에서 앱 실행
  - `streamlit run app.py --server.port=8502`

### 핵심 로직
- **`recommendation_workflow.py`** (420줄)
  - LangGraph 워크플로우 정의
  - 검색 → 감성 분석 → 추천 생성
  - 프로필 + 프롬프트 통합 시스템
  - 비건 필터링 (64개 금지 재료)
  - 쿼리 자동 강화

- **`recipe_retriever.py`** (257줄)
  - FAISS 벡터 검색
  - LangChain Retriever 통합
  - 임베딩 생성 및 로드

- **`agent_system.py`** (209줄)
  - ReAct 패턴 Multi-Agent
  - RecipeSearchTool, SentimentAnalysisTool
  - 안전장치 (max_iterations=3, timeout=30s)

### 데이터 처리
- **`data_loader.py`** (156줄)
  - CSV/Parquet 로드
  - 필수 정보 추출
  - 데이터 전처리

- **`embedding_utils.py`** (117줄)
  - OpenAI Embeddings API
  - 배치 처리 최적화
  - text-embedding-3-small 모델

### 사용자 기능
- **`user_profile.py`** (167줄)
  - 식단 제약 (vegan, vegetarian, low_sodium)
  - 알레르기 관리
  - 선호도 필터링

- **`sentiment_module.py`** (112줄)
  - Transformers 감성 분석
  - 사용자 의도 파악

## 🔧 빌드 스크립트

### `scripts/build_full_index.py`
임베딩 생성 및 FAISS 인덱스 저장

```bash
# 사용법
python scripts/build_full_index.py \
  --batch-size 32 \
  --output data/recipe_full.index
```

**기능**:
- 23,192개 레시피 임베딩 생성
- FAISS 인덱스 저장
- 진행률 및 ETA 표시
- 비용: 약 $0.06

## 🧪 테스트 파일

### `tests/check_vegan_recipes.py`
비건 레시피 데이터 검증

```bash
python tests\check_vegan_recipes.py
```

**출력**:
- 비건 가능 레시피: 3,039개 (13.1%)
- 비건 단백질 레시피: 536개
- 금지 재료 체크

### `tests/test_search_simulation.py`
실제 검색 동작 시뮬레이션

```bash
python tests\test_search_simulation.py
```

**출력**:
- 검색 결과 개수
- 필터링 후 개수
- 비건 레시피 샘플

### `tests/test_profile_integration.py`
프로필 + 프롬프트 통합 테스트

```bash
python tests\test_profile_integration.py
```

**테스트 케이스**:
1. 비건 (프롬프트 only)
2. 비건 (사이드바 only)
3. 알레르기 (프롬프트)
4. 복합 조건

## 📊 데이터 파일

### `data/TB_RECIPE_SEARCH_241226.csv`
- **크기**: 23,192 레시피
- **출처**: 공공데이터포털
- **컬럼**: RCP_TTL, CKG_MTRL_CN, CKG_IPDC 등

### `data/recipe_full_with_embeddings.parquet`
- **크기**: 23,192 레시피 + 임베딩
- **임베딩**: 1536차원 (text-embedding-3-small)
- **포맷**: Parquet (빠른 로딩)

## 📖 문서 파일

### `docs/PHASE1_SUMMARY.md`
Phase 1 프로젝트 완료 보고서
- 구현 내역
- 비용 분석
- 테스트 결과
- 프로젝트 평가

### `docs/final_report.md`
최종 연구 보고서
- 연구 목적
- 기술 스택
- 구현 과정
- 결과 및 결론

### `tests/TEST_SCENARIOS.md`
테스트 시나리오 상세
- 8가지 테스트 케이스
- 기대 결과
- 로그 확인 방법

## 🗂️ Archive 폴더

### `archive/v1_original/`
최초 구현 버전 (보관용)

### `archive/v2_langchain/`
LangChain 초기 버전 (보관용)

---

**업데이트**: 2025년 12월 2일  
**정리 완료**: 디렉토리 구조 재구성, 테스트/문서 분리
