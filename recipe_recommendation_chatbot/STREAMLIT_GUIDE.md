# Streamlit App 실행 가이드 (LangChain 버전)

## 🚀 빠른 실행

### 방법 1: Windows 배치 파일 (가장 쉬움)
```cmd
cd menu_bot_langchain
run_app.bat
```

### 방법 2: Python 스크립트
```cmd
cd menu_bot_langchain
python run_app.py
```

### 방법 3: 직접 Streamlit 실행
```cmd
cd menu_bot_langchain
streamlit run app_langchain.py
```

---

## 📋 사전 준비

### 1. 의존성 설치
```cmd
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일에 OpenAI API 키 추가:
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

### 3. 데이터 준비
다음 파일이 있어야 합니다:
```
data/TB_RECIPE_SEARCH_241226.csv
```

**선택사항**: 전체 인덱스 사용 (더 빠른 검색)
```
data/recipe_full.index
data/recipe_full.parquet
```

전체 인덱스 생성 방법:
```cmd
cd ..
python build_full_index.py
```

---

## 🎯 앱 기능

### 1. 추천 모드 (3가지)

#### 🚀 LangGraph Workflow (권장)
- 상태 관리 기반 워크플로우
- 감성 분석 → 검색 → 추천 생성
- 조건부 라우팅
- 전체 프로세스 추적 가능

#### ⚡ Simple RAG
- 빠른 기본 추천
- 벡터 검색 + LLM 응답
- 감성 분석 없음

#### 🎯 Advanced Chain
- LCEL 체인 기반
- 감성 분석 통합
- JSON 형식 구조화된 응답

### 2. 사용자 프로필 설정
- **알레르기**: 피해야 할 재료
- **선호 맛**: 선호하는 맛 키워드
- **비선호 맛**: 피하고 싶은 맛
- **식단 제약**: 비건, 저염 등

### 3. 인덱스 설정
- **전체 인덱스**: 23,000+ 레시피 (느리지만 정확)
- **샘플 인덱스**: 50-500 레시피 (빠름)

---

## 💡 사용 예시

### 예시 질문들:

1. **단백질 중심**
   ```
   운동 끝나고 단백질 많은 메뉴 추천해줘
   ```

2. **감성 기반**
   ```
   기분이 우울해서 따뜻하고 위로되는 음식 먹고 싶어
   ```

3. **시간 제약**
   ```
   출근 전에 10분 안에 만들 수 있는 간단한 아침 메뉴
   ```

4. **상황 기반**
   ```
   비 오는 날 먹기 좋은 따뜻한 요리 추천
   ```

5. **식단 제약**
   ```
   비건 식단으로 단백질 보충할 수 있는 메뉴
   ```

---

## 🔧 기술 스택

### Frontend
- **Streamlit**: 웹 UI 프레임워크
- **Python 3.8+**: 백엔드

### Backend (LangChain/LangGraph)
- **LangChain Retriever**: FAISS 벡터 검색
- **LCEL Chain**: 선언적 파이프라인
- **LangGraph**: 상태 기반 워크플로우
- **OpenAI GPT-4**: 추천 생성

### Data
- **FAISS**: 벡터 인덱스
- **Pandas**: 데이터 처리
- **Transformers**: 감성 분석

---

## 📊 앱 vs 기존 app.py 비교

| 기능 | 기존 app.py | app_langchain.py |
|------|-------------|------------------|
| 추천 엔진 | 함수 기반 | LangChain/LangGraph |
| 상태 관리 | ❌ | ✅ TypedDict |
| 모드 선택 | 자동 | 수동 (3가지) |
| 프로필 필터링 | 검색 후 | 검색 전 (Retriever 래핑) |
| 대화 히스토리 | ❌ | ✅ 세션 저장 |
| 디버그 정보 | 제한적 | ✅ 상세 JSON |
| 에러 처리 | 기본 | ✅ 강화 |

---

## 🐛 문제 해결

### Streamlit이 설치 안됨
```cmd
pip install streamlit
```

### OpenAI API 오류
1. `.env` 파일에 `OPENAI_API_KEY` 확인
2. API 키 유효성 확인
3. API 사용량 한도 확인

### 데이터 파일 없음
```
FileNotFoundError: data/TB_RECIPE_SEARCH_241226.csv
```
→ 데이터 파일을 `data/` 폴더에 배치

### 임베딩 생성 느림
- 샘플 크기를 줄이세요 (50-100)
- 또는 전체 인덱스를 미리 생성:
  ```cmd
  python build_full_index.py
  ```

### 메모리 부족
- 샘플 크기 감소
- 배치 크기 감소 (embedding_utils.py)

---

## 🎨 UI 커스터마이징

### 포트 변경
```cmd
streamlit run app_langchain.py --server.port 8502
```

### 테마 변경
`.streamlit/config.toml` 생성:
```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## 📈 성능 팁

### 1. 전체 인덱스 사용
```cmd
python build_full_index.py
```
→ 검색 속도 10배 향상

### 2. 캐싱 활용
- `@st.cache_resource` 자동 적용
- 재실행 시 인덱스 재로드 불필요

### 3. 배치 크기 조정
`generate_embeddings(..., batch_size=32)` → 속도 향상

---

## 🔐 보안 주의사항

1. **API 키 보호**
   - `.env` 파일을 `.gitignore`에 추가
   - 공개 저장소에 업로드 금지

2. **사용량 모니터링**
   - OpenAI API 사용량 확인
   - 비용 한도 설정

---

## 📞 추가 도움말

### 관련 문서
- `README.md`: 패키지 개요
- `ARCHITECTURE.txt`: 아키텍처 상세
- `LANGCHAIN_MIGRATION_REPORT.md`: 변경사항

### 예제 코드
- `run_examples.py`: CLI 예제
- `app_langchain.py`: Streamlit 앱

---

## ✅ 실행 체크리스트

- [ ] 의존성 설치 완료
- [ ] `.env` 파일에 API 키 설정
- [ ] 데이터 파일 준비 (`data/TB_RECIPE_SEARCH_241226.csv`)
- [ ] Streamlit 설치 확인 (`streamlit --version`)
- [ ] 앱 실행 (`run_app.bat` 또는 `python run_app.py`)
- [ ] 브라우저에서 `http://localhost:8501` 접속
- [ ] 테스트 질문 입력 및 추천 확인

---

**Happy Recommending!** 🍽️
