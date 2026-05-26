"""
ai/agents/common.py — Gemini 호출 공통 유틸

ai_question_agent.py / summary_agent.py에서 반복되던 패턴 통합:
  - get_gemini_model(): GenerativeModel 생성
  - generate_with_retry(): 재시도 로직 포함 generate_content 호출
  - parse_json_response(): JSON 파싱 (partial recovery + regex fallback)
"""
import json
import logging
import re
from typing import Any

from vertexai.generative_models import GenerationConfig, GenerativeModel

from ai.config import settings  # noqa: F401 — vertexai.init() 호출 포함

logger = logging.getLogger(__name__)


def get_gemini_model() -> GenerativeModel:
    """설정된 GEMINI_MODEL로 GenerativeModel 인스턴스 반환."""
    return GenerativeModel(model_name=settings.GEMINI_MODEL)


def generate_with_retry(
    model: GenerativeModel,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_output_tokens: int = 4096,
    max_retries: int = 2,
    retry_temperature_delta: float = 0.2,
) -> str:
    """
    Gemini generate_content 호출 + 실패 시 재시도.

    Args:
        model: GenerativeModel 인스턴스
        prompt: 프롬프트 문자열
        temperature: 초기 temperature
        max_output_tokens: 최대 출력 토큰
        max_retries: 최대 재시도 횟수 (초기 시도 제외)
        retry_temperature_delta: 재시도 시 temperature 증가량

    Returns:
        응답 텍스트 (str). 모든 시도 실패 시 ValueError 발생.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        current_temp = min(temperature + retry_temperature_delta * attempt, 1.0)
        try:
            resp = model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=current_temp,
                    max_output_tokens=max_output_tokens,
                ),
            )
            return resp.text
        except Exception as e:
            last_error = e
            logger.warning(f"generate_content 실패 (attempt {attempt + 1}/{max_retries + 1}): {e}")

    raise ValueError(f"Gemini 호출 {max_retries + 1}회 모두 실패: {last_error}")


def parse_json_response(text: str, array: bool = False) -> Any:
    """
    LLM 응답 텍스트에서 JSON 파싱.
    1차: 직접 json.loads
    2차: ```json ... ``` 코드블록 추출
    3차: 첫 번째 { } 또는 [ ] 블록 추출 (partial recovery)

    Args:
        text: LLM 응답 원문
        array: True면 리스트, False면 dict를 기대

    Returns:
        파싱된 Python 객체. 실패 시 [] 또는 {} 반환.
    """
    default = [] if array else {}

    # 1차: 직접 파싱
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2차: 코드블록 제거
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3차: 첫 번째 구조 추출
    pattern = r"\[.*?\]" if array else r"\{.*?\}"
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("JSON 파싱 전체 실패, 기본값 반환")
    return default
