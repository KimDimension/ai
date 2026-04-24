"""
위험도 트리아지 + AI 종합 요약 + EMR 작성 에이전트
- 기록 수치 + 공통 질문 응답 + AI 구조화 설문 응답을 종합 분석
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
    ai_survey_responses: list[dict] = None,
    conversation_messages: list[dict] = None,  # 하위 호환성 유지 (무시됨)
) -> dict:
    """
    설문 완료 후 위험도 + 요약 + EMR 생성

    Args:
        record_data:          오늘 투석 기록
        common_qa:            공통 질문 답변 목록
                              [{"question_text": str, "choice": "yes"|"no"|None, "text_answer": str}]
        ai_survey_responses:  AI 추천 질문 응답 목록
                              [{"question_text": str, "question_type": str, "answer": str}]
        conversation_messages: 하위 호환성용 (무시됨)

    Returns:
        {
            "risk_level":  "normal" | "caution" | "urgent",
            "ai_summary":  "의사용 요약 (2~4문장)",
            "emr_soap":    "S: ...\\nO: ...\\nA: ...\\nP: ...",
        }
    """
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

        # 공통질문 응답 정리
        common_text = "없음"
        if common_qa:
            lines = []
            for item in common_qa:
                answer = item.get("choice", "미응답") or "미응답"
                if item.get("text_answer"):
                    answer += f" / 추가: {item['text_answer']}"
                lines.append(f"- {item['question_text']}: {answer}")
            common_text = "\n".join(lines)

        # AI 설문 응답 정리
        ai_survey_text = "없음"
        if ai_survey_responses:
            lines = []
            for item in ai_survey_responses:
                q_type = item.get("question_type", "yes_no")
                answer = item.get("answer", "미응답") or "미응답"

                type_label = {
                    "yes_no":        "예/아니오",
                    "single_select": "단일 선택",
                    "multi_select":  "다중 선택",
                    "short_text":    "단답",
                }.get(q_type, q_type)

                lines.append(f"- [{type_label}] {item['question_text']}: {answer}")
            ai_survey_text = "\n".join(lines)

        prompt = f"""당신은 CAPD(복막투석) 전문 의료 AI입니다.
아래 환자 데이터를 바탕으로 위험도 분류, 의사용 요약, EMR(SOAP)을 작성하세요.

[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

[공통 질문 답변]
{common_text}

[AI 추천 질문 응답]
{ai_survey_text}

[작성 기준]

위험도 분류:
- urgent(긴급): 복막염 의심(탁한 투석액+복통+발열), 극심한 증상, 즉각 처치 필요
- caution(주의): 체중 2kg 이상 증가, 혈압 160 이상 또는 90 미만, 혈당 250 이상, 소변량 감소 등
- normal(정상): 이상 소견 없음

요약 작성 지침:
- 의사가 한눈에 파악할 수 있도록 2~4문장으로 핵심만 작성
- 이상 수치와 환자 호소 증상 중심으로 작성
- AI 설문 응답에서 주목할 내용이 있으면 포함
- 한국어로 작성

EMR SOAP 작성:
- S(Subjective): 환자가 직접 호소한 증상 (설문 응답 기반)
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
                response_mime_type="application/json",
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
