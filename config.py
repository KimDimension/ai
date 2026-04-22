"""
AI 서버 설정
"""
import os
from pydantic_settings import BaseSettings


class AISettings(BaseSettings):
    # Gemini API — 반드시 .env 파일에 설정 (코드에 키 하드코딩 금지)
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Backend DB 연결 (RAG용)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://capd_user:capd_pass@db:5432/capd")

    # Backend API URL (ai 서버 → backend 서버로 결과 저장)
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://backend:8000")

    # 문진 설정
    MAX_TURNS: int = 5          # AI 질문 최대 횟수
    URGENT_KEYWORDS: list = [   # 긴급 신호 키워드 (환자 답변에서 감지)
        "복통", "배가 너무 아파", "열이 나", "고열", "숨이 너무", "응급",
        "극심한", "참을 수 없", "119", "병원 가야"
    ]

    class Config:
        env_file = ".env"
        extra = "allow"


settings = AISettings()
