"""
정적 설문용 AI 추천 질문 생성 에이전트
- 환자 투석 기록 이상 수치 기반으로 예/아니오 질문 1개 생성
- 의사 공통 질문 아래 'AI 추천 질문' 섹션에 표시됨
- RAG(KDIGO 검색) 컨텍스트 주입 지원
- 고령 환자 대상: 쉬운 표현, 짧은 문장, 불안 유발 표현 금지
"""
import json
import logging
import re

import google.generativeai as genai

from ai.config import settings
from ai.tools.record_analyzer import summarize_anomalies_text

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)


def generate_ai_questions(
    record_data: dict,
    rejected_keys: list[str] = None,
    kdigo_context: str = "",
) -> list[dict]:
    """
    환자 투석 기록 기반 AI 추천 질문 생성 (예/아니오 설문용)

    Args:
        record_data:    환자의 오늘 투석 기록 dict
        rejected_keys:  제외할 질문 패턴 키 목록
        kdigo_context:  RAG로 검색한 KDIGO 관련 문단

    Returns:
        [{"question_text": "..."}] 형태 리스트
        오류 시 빈 리스트 반환
    """
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

        rejected_str = ", ".join(rejected_keys) if rejected_keys else "없음"
        anomaly_text = summarize_anomalies_text(record_data)

        kdigo_block = ""
        if kdigo_context:
            kdigo_block = f"""
[KDIGO 관련 지침]
{kdigo_context}
"""

        prompt = f"""당신은 CAPD(복막투석) 환자를 담당하는 의료팀의 AI 보조 도구입니다.
아래 오늘의 투석 기록과 이상 수치 분석을 바탕으로, 의사에게 전달할 추가 정보를 수집하기 위한 질문 1개를 생성하세요.
{kdigo_block}
[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

[이상 수치 분석]
{anomaly_text}

[이미 제외된 패턴]
{rejected_str}

[규칙]
- 이상 수치나 주의가 필요한 항목에 집중하세요
- KDIGO 지침이 있으면 해당 근거를 바탕으로 질문하세요
- 환자가 예/아니오로 답할 수 있는 질문을 만드세요
- 대부분 고령 환자임을 감안하여 쉽고 짧은 한국어 표현을 사용하세요 (의학 전문용어 금지)
- "심각한", "위험한", "응급" 등 불안감을 줄 수 있는 표현은 사용하지 마세요
- 이 질문은 진단이 아니라 의사에게 상태를 전달하기 위한 정보 수집임을 염두에 두세요
- 제외된 패턴과 유사한 질문은 만들지 마세요
- 질문은 30자 이내로 간결하게 작성하세요

아래 JSON 형식으로만 응답하세요:
{{"question_text": "질문 내용"}}"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )

        text = response.text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # JSON이 잘린 경우 question_text만 regex로 추출
            match = re.search(r'"question_text"\s*:\s*"([^"]+)"', text)
            if match:
                return [{"question_text": match.group(1)}]
            return []

        if isinstance(data, dict) and "question_text" in data:
            return [{"question_text": data["question_text"]}]
        elif isinstance(data, list):
            return [{"question_text": q["question_text"]} for q in data if "question_text" in q]

        return []

    except json.JSONDecodeError as e:
        logger.warning(f"AI 질문 JSON 파싱 실패: {e}")
        return []
    except Exception as e:
        logger.warning(f"AI 질문 생성 실패: {e}")
        return []
