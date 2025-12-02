import logging
import time

# 로깅 설정
logger = logging.getLogger("sentiment_module")
logger.setLevel(logging.INFO)

# Lazy loading을 위해 전역 변수로 파이프라인 캐싱
_sentiment_pipeline = None

# Streamlit 환경에서 캐싱 지원
try:
    import streamlit as st
    USE_STREAMLIT_CACHE = True
except ImportError:
    USE_STREAMLIT_CACHE = False

def _get_sentiment_pipeline():
    """실제 파이프라인 로딩 구현 (캐싱 가능하도록 분리)"""
    logger.info("[SENTIMENT] Initializing sentiment analysis pipeline (first call)...")
    start_time = time.time()
    
    try:
        logger.info("[SENTIMENT] Importing transformers library...")
        from transformers import pipeline
        
        logger.info("[SENTIMENT] Loading DistilBERT model (130MB)...")
        # 기본 영어 모델 명시적 지정 (130MB, 빠름)
        pipe = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[SENTIMENT] Model loaded successfully in {elapsed:.2f}s")
        return pipe
        
    except Exception as e:
        logger.error(f"[SENTIMENT] Error loading sentiment model: {e}")
        return None

def _get_sentiment_pipeline():
    """감성 분석 파이프라인을 지연 로딩하는 헬퍼 함수.
    
    Streamlit 환경에서는 st.cache_resource를 사용하여 세션 간 공유.
    비-Streamlit 환경에서는 전역 변수로 캐싱.
    """
    global _sentiment_pipeline
    
    if USE_STREAMLIT_CACHE:
        # Streamlit 캐싱 활용
        if '_sentiment_pipeline_cached' not in st.session_state:
            # 첫 로드 시에만 실행
            if _sentiment_pipeline is None:
                _sentiment_pipeline = _load_pipeline_impl()
            st.session_state['_sentiment_pipeline_cached'] = True
        return _sentiment_pipeline
    else:
        # 전역 변수 캐싱 (기존 방식)
        if _sentiment_pipeline is None:
            _sentiment_pipeline = _load_pipeline_impl()
        else:
            logger.debug("[SENTIMENT] Using cached pipeline")
        return _sentiment_pipeline

def get_user_sentiment(text):
    """
    사용자 텍스트를 입력받아 감성 분석 결과를 반환하는 함수.
    한국어 및 다국어 지원.
    
    첫 호출 시 transformers 모델을 로드하므로 약간의 지연이 있을 수 있습니다.
    """
    logger.info(f"[SENTIMENT] Analyzing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    start_time = time.time()
    
    sentiment_pipeline = _get_sentiment_pipeline()
    
    if not sentiment_pipeline:
        logger.warning("[SENTIMENT] Pipeline unavailable, returning NEUTRAL")
        return {"label": "NEUTRAL", "score": 0.0, "description": "중립적인 기분"}
        
    try:
        # 텍스트에 대해 감성 분석 수행
        result = sentiment_pipeline(text)[0]
        
        # 결과를 파악하기 쉬운 형태로 변환
        label = result['label']  # 예: 'positive', 'negative', 'neutral', 'POSITIVE', 'NEGATIVE'
        score = result['score']  # 감성 점수 (확률)

        # 감성 라벨을 한국어로 변환 (대소문자 무관)
        label_upper = label.upper()
        if 'POSITIVE' in label_upper or 'POS' in label_upper:
            sentiment_text = "긍정적인 기분"
        elif 'NEGATIVE' in label_upper or 'NEG' in label_upper:
            sentiment_text = "부정적인 기분"
        else:
            sentiment_text = "중립적인 기분"

        elapsed = time.time() - start_time
        logger.info(f"[SENTIMENT] Result: {sentiment_text} (score: {score:.3f}) in {elapsed:.3f}s")
        
        return {
            "label": label,
            "score": score,
            "description": sentiment_text
        }
    except Exception as e:
        logger.error(f"[SENTIMENT] Analysis failed: {e}")
        return {"label": "NEUTRAL", "score": 0.5, "description": "중립적인 기분"}

# 테스트 예시
# analysis = get_user_sentiment("오늘 날씨가 너무 좋아서 기분이 정말 최고야!")
# print(analysis)