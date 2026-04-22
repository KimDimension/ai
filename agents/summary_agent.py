"""
위험도 트리아지 + AI 종합 요약 + EMR 작성 에이전트
- 문진 완료 후 전체 대화 + 기록을 보고 위험도 판단
- 의사용 AI 요약문 생성
- S/O/A/P EMR 작성
"""
import json
import logging

import google.generativeai as genai

from ai.config import settings

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)


def generate_summary_and_triage(
    record_data: dict,
    common_qa: list[dict],
    conversation_messages: list[dict],
) -> dict:
    """
    문진 종료 후 위험도 + 요약 + EMR 생성

    Args:
        record_data:             오늘 투석 기록
        common_qa:               공통 질문 답변 목록
        conversation_messages:   대화 메시지 목록 [{"role":"ai"|"user","content":str}]

    Returns:
        {
            "risk_level":  "normal" | "caution" | "urgent",
            "ai_summary":  "의사용 요약 (2~4문장)",
            "emr_soap":    "S: ...\nO: ...\nA: ...\nP: ...",
        }
    """
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

        # 공통질문 정리
        common_text = ""
        if common_qa:
            lines = []
            for item in common_qa:
                answer = item.get("choice", "미응답")
                if item.get("text_answer"):
                    answer += f" / {item['text_answer']}"
                lines.append(f"- {item['question_text']}: {answer}")
            common_text = "\n".join(lines)

        # 대화 내용 정리
        conv_text = ""
        if conversation_messages:
            lines = []
            for msg in conversation_messages:
                prefix = "AI" if msg["role"] == "ai" else "환자"
                lines.append(f"{prefix}: {msg['content']}")
            conv_text = "\n".join(lines)

        prompt = f"""당신은 CAPD(복막투석) 전문 의료 AI입니다.
아래 환자 데이터를 바탕으로 위험도 분류, 의사용 요약, EMR(SOAP)을 작성하세요.

[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

[공통 질문 답변]
{common_text or "없음"}

[AI 대화형 문진 내용]
{conv_text or "문진 없음"}

[작성 기준]

위험도 분류:
- urgent(긴급): 복막염 의심(탁한 투석액+복통+발열), 극심한 증상, 즉각 처치 필요
- caution(주의): 체중 2kg 이상 증가, 혈압 160 이상 또는 90 미만, 혈당 250 이상, 소변량 감소 등
- normal(정상): 이상 소견 없음

요약 작성 지침:
- 의사가 한눈에 파악할 수 있도록 2~4문장으로 핵심만
- 이상 수치와 환자 호소 증상 중심으로 작성
- 한국어로 작성

EMR SOAP 작성:
- S(Subjective): 환자가 직접 호소한 증상 (문진 내용 기반)
- O(Objective): 객관적 수치 (기록 데이터 기반)
- A(Assessment): AI의 소견 및 위험도
- P(Plan): 권장 조치 사항

[응답 형식 — 반드시 JSON으로만 응답]
{{
  "risk_level": "normal" | "caution" | "urgent",
  "ai_summary": "의사용 요약 텍스트",
  "emr_soap": "S: ...\\nO: ...\\nA: ...\\nP: ..."
}}"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=800,
            ),
        )

        return _parse_summary_response(response.text)

    except Exception as e:
        logger.error(f"요약/트리아지 생성 실패: {e}")
        return _fallback_triage(record_data)


def _parse_summary_response(text: str) -> dict:
    """응답 파싱"""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)

        # risk_level 유효성 검사
        if data.get("risk_level") not in ("normal", "caution", "urgent"):
            data["risk_level"] = "caution"

        return {
            "risk_level": data.get("risk_level", "caution"),
            "ai_summary": data.get("ai_summary", "요약 생성 실패"),
            "emr_soap":   data.get("emr_soap", "EMR 생성 실패"),
        }

    except json.JSONDecodeError:
        logger.warning(f"요약 JSON 파싱 실패: {text[:100]}")
        return {
            "risk_level": "caution",
            "ai_summary": text.strip()[:500],
            "emr_soap":   "",
        }


def _fallback_triage(record_data: dict) -> dict:
    """Gemini 실패 시 규칙 기반 위험도 판단"""
    risk = "normal"

    bp = record_data.get("blood_pressure", "")
    try:
        systolic = int(bp.split("/")[0])
        if systolic >= 160 or systolic < 90:
            risk = "caution"
    except Exception:
        pass

    if record_data.get("turbid_peritoneal"):
        risk = "urgent"

    glucose = record_data.get("fasting_blood_glucose") or 0
    if glucose >= 250:
        risk = "caution"

    return {
        "risk_level": risk,
        "ai_summary": "AI 요약 생성에 실패했습니다. 의사가 직접 기록을 확인해 주세요.",
        "emr_soap":   "",
    }
