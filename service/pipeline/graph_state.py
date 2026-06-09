"""graph_state.py — LangGraph 상태 스키마.

shopping 경로 (Evidence Pack + hard_gate):
    intent_router → alias_resolver → profile_loader → pack_loader
    → solar_explainer → hard_gate_check → END

general 경로 (Evidence 없음, 미검증 배지):
    intent_router → general_chat → END
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


Intent = Literal["shopping", "general", "user_id"]


class GraphState(TypedDict, total=False):
    # 입력
    message: str
    user_alias: Optional[str]
    selected_item_id: Optional[str]       # 카드 선택 시 — 없으면 submission top-1
    question_key: str                     # advisor.prompts.QUICK_QUESTIONS 키

    # intent
    intent: Optional[Intent]

    # shopping 경로
    sections: Optional[dict[str, Any]]    # orchestrator 5섹션 결과
    pack_json: Optional[str]              # EvidencePack JSON (state는 직렬화 가능해야 함)
    advisor_response: Optional[dict]      # AdvisorResponse.model_dump()
    hard_gate_errors: Optional[list[str]]

    # 출력
    trust_badge: Optional[Literal["verified", "rejected", "unverified"]]
    response_text: Optional[str]