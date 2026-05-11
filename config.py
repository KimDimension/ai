"""
AI 서버 설정

⚠️ 모델명 주의 ⚠️
- 반드시 gemini-2.5-flash 사용
- gemini-2.0-flash / gemini-2.0-flash-001 은 이 프로젝트(skuniv-training-2)에서 404 발생 — 사용 불가
- 모델명 변경 시 deploy.yml의 --set-env-vars 도 함께 수정할 것
"""
import vertexai
from pydantic_settings import BaseSettings


class AISettings(BaseSettings):
    # Vertex AI (ADC 인증 — 서비스 계정 키 불필요)
    GCP_PROJECT_ID: str
    GCP_LOCATION: str = "us-central1"
    # ⚠️ gemini-2.0-flash 계열은 skuniv-training-2 프로젝트에서 접근 불가 (404)
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Backend DB 연결 (RAG용) — 반드시 .env에 설정 (기본값 없음, 하드코딩 금지)
    DATABASE_URL: str

    class Config:
        env_file = ".env"
        extra = "allow"


settings = AISettings()

# Vertex AI 초기화 (ADC 자동 인증)
vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
