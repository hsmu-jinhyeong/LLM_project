import openai
import datetime
import os
# OpenAI API 키 설정 (보안을 위해 환경 변수로 설정 권장)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 섹션 1에서 정의한 감성 분석 함수가 필요합니다.
from sentiment_module import get_user_sentiment 

def get_menu_recommendation(user_input):
    """
    사용자 입력에 감성 분석을 적용하여 메뉴를 추천하는 챗봇 응답을 받습니다.
    """
    # 1. 감성 분석 수행
    sentiment_data = get_user_sentiment(user_input)
    current_time = datetime.datetime.now().strftime("%H시 %M분") # 현재 시간 파악 (datetime 라이브러리 필요)
    
    # 2. 감성 분석 결과를 프롬프트에 삽입하여 컨텍스트 강화
    # System Prompt는 챗봇의 역할과 제약 조건을 정의합니다.
    system_prompt = f"""
    당신은 사용자 맞춤형 메뉴 추천 전문가입니다. 
    사용자의 현재 상황, 시간대, 감성 분석 결과를 **가장 중요하게 고려**하여 메뉴를 추천하세요.
    - 현재 시간: {current_time}
    - 사용자의 감성: {sentiment_data['description']} (점수: {sentiment_data['score']:.2f})
    추천 시에는 이유를 간결하게 설명하고, 추천 메뉴를 1~2개 제시하세요.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # 더 나은 추론 능력을 위해 최신 모델 권장
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7 # 창의적인 추천을 위해 적절한 값 설정
        )
        
        # 3. 챗봇 응답 반환
        return response.choices[0].message.content
        
    except openai.APIError as e:
        return f"OpenAI API 호출 중 오류가 발생했습니다: {e}"

# 사용 예시
user_text = "아, 오늘 업무가 너무 많아서 스트레스 받아. 점심 뭐 먹지?"
recommendation = get_menu_recommendation(user_text)
print(recommendation)