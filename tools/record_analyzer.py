"""
환자 기록 이상 수치 분석 도구
- 규칙 기반으로 주의가 필요한 항목 감지
- ai_questions 생성 및 summary_agent에서 활용
"""
from typing import Optional


def analyze_anomalies(record_data: dict) -> list[dict]:
    """
    기록에서 이상 수치 항목을 감지하여 반환

    Returns:
        [{"field": str, "value": ..., "reason": str, "severity": "urgent"|"caution"}]
    """
    results = []

    # 혈압
    bp = record_data.get("blood_pressure") or ""
    try:
        parts = bp.split("/")
        systolic  = int(parts[0])
        diastolic = int(parts[1])
        if systolic >= 160 or diastolic >= 100:
            results.append({
                "field":    "blood_pressure",
                "value":    bp,
                "reason":   f"혈압이 높습니다 ({bp} mmHg). 수분 섭취 및 투석 처방 재검토가 필요할 수 있습니다.",
                "severity": "caution",
            })
        elif systolic < 90:
            results.append({
                "field":    "blood_pressure",
                "value":    bp,
                "reason":   f"혈압이 낮습니다 ({bp} mmHg). 탈수 또는 과도한 투석 제거를 확인하세요.",
                "severity": "caution",
            })
    except Exception:
        pass

    # 혼탁한 투석액 → 복막염 의심 (긴급)
    if record_data.get("turbid_peritoneal"):
        results.append({
            "field":    "turbid_peritoneal",
            "value":    True,
            "reason":   "투석액이 혼탁합니다. 복막염 가능성이 있어 즉각적인 확인이 필요합니다.",
            "severity": "urgent",
        })

    # 혈당
    glucose = record_data.get("fasting_blood_glucose")
    if glucose:
        if glucose > 250:
            results.append({
                "field":    "fasting_blood_glucose",
                "value":    glucose,
                "reason":   f"공복혈당이 높습니다 ({glucose} mg/dL). 당 농도 처방 및 인슐린 조절이 필요할 수 있습니다.",
                "severity": "caution",
            })

    # 체중 (이전 기록 없이 절대값만 판단 — 기준 부재 시 75kg 이상 플래그)
    weight = record_data.get("weight")
    if weight and weight > 80:
        results.append({
            "field":    "weight",
            "value":    weight,
            "reason":   f"체중이 {weight}kg으로 높습니다. 부종 여부 및 수분 균형을 확인하세요.",
            "severity": "caution",
        })

    # 소변량
    urine = record_data.get("urine_count")
    if urine is not None and urine == 0:
        results.append({
            "field":    "urine_count",
            "value":    urine,
            "reason":   "소변이 없습니다. 잔여 신기능 소실 여부를 확인하세요.",
            "severity": "caution",
        })

    return results


def has_urgent_anomaly(record_data: dict) -> bool:
    """긴급 이상 수치 존재 여부"""
    return any(a["severity"] == "urgent" for a in analyze_anomalies(record_data))


def summarize_anomalies_text(record_data: dict) -> str:
    """이상 항목 요약 텍스트 (프롬프트 주입용)"""
    anomalies = analyze_anomalies(record_data)
    if not anomalies:
        return "이상 수치 없음"
    lines = [f"- [{a['severity'].upper()}] {a['reason']}" for a in anomalies]
    return "\n".join(lines)
