from transformers import pipeline

# 감성 분석 파이프라인 초기화
# 'sentiment-analysis' 태스크를 수행하는 사전 학습된 모델을 로드합니다.
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
except Exception as e:
    # 모델 다운로드 실패 시 대체 로직 또는 에러 처리
    print(f"Error loading sentiment model: {e}")
    sentiment_pipeline = None

def get_user_sentiment(text):
    """
    사용자 텍스트를 입력받아 감성 분석 결과를 반환하는 함수.
    """
    if not sentiment_pipeline:
        return {"label": "NEUTRAL", "score": 0.0} # 모델 로드 실패 시 기본값
        
    try:
        # 텍스트에 대해 감성 분석 수행
        result = sentiment_pipeline(text)[0]
        
        # 결과를 파악하기 쉬운 형태로 변환
        label = result['label']  # 예: 'POSITIVE', 'NEGATIVE'
        score = result['score']  # 감성 점수 (확률)

        # 감성 라벨을 한국어로 변환하거나 추천 시스템에 맞게 조정
        if label == 'POSITIVE':
            sentiment_text = "긍정적인 기분"
        elif label == 'NEGATIVE':
            sentiment_text = "부정적인 기분"
        else:
            sentiment_text = "중립적인 기분"

        return {
            "label": label,
            "score": score,
            "description": sentiment_text
        }
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return {"label": "ERROR", "score": 0.0, "description": "분석 오류"}

# 테스트 예시
# analysis = get_user_sentiment("오늘 날씨가 너무 좋아서 기분이 정말 최고야!")
# print(analysis)