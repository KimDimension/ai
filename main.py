"""
CAPD AI 서버 — FastAPI (포트 8001)
backend 서버와 HTTP로 통신
"""
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.agents.question_agent import start_conversation, next_turn
from ai.agents.summary_agent import generate_summary_and_triage
from ai.agents.ai_question_agent import generate_ai_questions
from ai.rag.retriever import search_kdigo_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CAPD AI API",
    description="대화형 문진 / 위험도 트리아지 / 종합 요약 AI 서버",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 스키마 ─────────────────────────────────────────────────────

class StartConversationRequest(BaseModel):
    record_data: dict          # 환자 투석 기록
    common_qa: list[dict] = [] # 공통 질문 답변 목록
    kdigo_context: str = ""    # RAG 컨텍스트 (선택)


class NextTurnRequest(BaseModel):
    record_data: dict
    common_qa: list[dict] = []
    history: list[dict]        # [{"role":"ai"|"user","content":str}]
    patient_answer: str        # 방금 환자 답변
    turn_number: int           # 현재 턴 번호 (1-based)
    kdigo_context: str = ""


class SummaryRequest(BaseModel):
    record_data: dict
    common_qa: list[dict] = []
    conversation_messages: list[dict]  # 전체 대화 내용


class AIMessage(BaseModel):
    type: str      # "question" | "urgent" | "done"
    content: str   # 환자에게 보낼 메시지
    reason: str    # AI 판단 근거 (로그/디버깅용)


class SummaryResponse(BaseModel):
    risk_level: str   # "normal" | "caution" | "urgent"
    ai_summary: str   # 의사용 요약
    emr_soap: str     # S/O/A/P EMR


class AIQuestionsRequest(BaseModel):
    """기존 정적 설문용 AI 질문 생성 (surveys.py에서 호출)"""
    record_data: dict
    rejected_keys: list[str] = []   # 제외할 패턴 키 목록


class AIQuestionsResponse(BaseModel):
    questions: list[dict]           # [{"question_text": str, "reason": str}]


# ── 엔드포인트 ──────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "capd-ai"}


@app.post("/conversation/start", response_model=AIMessage)
def conversation_start(body: StartConversationRequest):
    """
    문진 시작 — 첫 번째 AI 질문 생성
    공통 질문 답변 완료 후 프론트에서 호출
    """
    result = start_conversation(
        record_data=body.record_data,
        common_qa=body.common_qa,
        kdigo_context=body.kdigo_context,
    )
    return AIMessage(**result)


@app.post("/conversation/next", response_model=AIMessage)
def conversation_next(body: NextTurnRequest):
    """
    다음 질문 생성 — 환자가 답변할 때마다 호출
    type이 "done" 또는 "urgent"이면 문진 종료
    """
    result = next_turn(
        record_data=body.record_data,
        common_qa=body.common_qa,
        history=body.history,
        patient_answer=body.patient_answer,
        turn_number=body.turn_number,
        kdigo_context=body.kdigo_context,
    )
    return AIMessage(**result)


@app.post("/summary", response_model=SummaryResponse)
def create_summary(body: SummaryRequest):
    """
    문진 종료 후 위험도 + 요약 + EMR 생성
    backend 서버가 대화 저장 후 호출
    """
    result = generate_summary_and_triage(
        record_data=body.record_data,
        common_qa=body.common_qa,
        conversation_messages=body.conversation_messages,
    )
    return SummaryResponse(**result)


@app.post("/ai-questions/generate", response_model=AIQuestionsResponse)
def generate_questions(body: AIQuestionsRequest):
    """
    기존 정적 설문 AI 질문 생성
    (구 backend/services/ai_service.py + rag_service.py 역할)
    backend/surveys.py 백그라운드 태스크에서 호출
    """
    kdigo_context = search_kdigo_context(body.record_data)
    questions = generate_ai_questions(
        record_data=body.record_data,
        rejected_keys=body.rejected_keys,
        kdigo_context=kdigo_context,
    )
    return AIQuestionsResponse(questions=questions)
