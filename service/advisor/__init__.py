"""advisor — Evidence Pack + 질문 → 구조화 LLM 응답.

기본 인터페이스: `get_client()`.
- UPSTAGE_API_KEY가 있고 ADVISOR_FORCE_MOCK!=1 이면 실제 SolarProClient.
- 그 외에는 MockClient (결정적, 비용 0, 본체 없이 데모 가능).

UI·trust_gate·eval은 모두 `get_client()`만 호출하므로 실/모크 전환이 투명하다.
"""

from __future__ import annotations

import os

from .schema import AdvisorResponse, Claim
from .prompts import QUICK_QUESTIONS, SYSTEM_PROMPT, build_user_message


def get_client(force_mock: bool | None = None):
    """환경에 따라 실제 클라이언트 또는 mock을 반환."""
    if force_mock is None:
        force_mock = os.getenv("ADVISOR_FORCE_MOCK") == "1" or not os.getenv("UPSTAGE_API_KEY")
    if force_mock:
        from .mock_client import MockClient
        return MockClient()
    from .client import SolarProClient
    return SolarProClient()


__all__ = [
    "AdvisorResponse", "Claim",
    "QUICK_QUESTIONS", "SYSTEM_PROMPT", "build_user_message",
    "get_client",
]
