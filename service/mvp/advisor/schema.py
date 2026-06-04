"""LLM 출력 스키마 — 모든 주장에 evidence_ref를 강제한다.

이 스키마를 통과하지 못한 응답은 hard_gate에서 거부된다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """LLM이 만든 하나의 주장 — 반드시 근거 키를 인용해야 한다."""

    text: str = Field(min_length=1, max_length=200)
    evidence_ref: str = Field(min_length=1, description="EvidencePack의 키 경로 (예: signals.item_recent14_pop_log1p)")


class AdvisorResponse(BaseModel):
    """LLM 응답 — UI에 보이는 단위. 사용자/운영자 모두 같은 스키마."""

    user_id: str
    item_id: str
    question_key: str
    summary: str = Field(min_length=1, max_length=300)
    claims: list[Claim] = Field(min_length=1, max_length=6)
    decision_hint: str = Field(default="", max_length=200)
    confidence_raw: float = Field(ge=0.0, le=1.0)

    # trust_gate가 채워주는 필드 (LLM은 채우지 않음, 채워도 무시)
    confidence_calibrated: float | None = None  # Calibration 결과
    self_check_score: float | None = None  # SelfCheckGPT 일치도 (1=일관, 0=흔들림)
    hard_gate_passed: bool = True  # Hard-gate 통과 여부
    flags: list[str] = Field(default_factory=list)  # 경고/메모 (⚠ 등)
