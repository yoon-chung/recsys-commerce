"""Hard-gate — 결정적 검사. LLM 응답이 사용자에게 가기 전 코드로 닫는 마지막 안전망.

검사 항목:
1. user_id가 pack과 일치
2. item_id가 pack.recommendations 안에 있음 (범위 밖이면 환각)
3. 모든 claim.evidence_ref가 pack.evidence_keys() 화이트리스트 안에 있음
4. 각 evidence_ref가 가리키는 실제 값이 비어있지/false가 아님 (근거 없는 인용 금지)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from advisor.schema import AdvisorResponse
    from evidence_pack.schema import EvidencePack, Recommendation


def get_evidence_value(pack: "EvidencePack", rec: "Recommendation", ref: str) -> Any:
    """evidence_ref(예: signals.item_pop_log1p)가 실제로 가리키는 값."""
    if "." not in ref:
        return None
    prefix, attr = ref.split(".", 1)
    if prefix == "user_context":
        return getattr(pack.user_context, attr, None)
    if prefix == "signals":
        return getattr(rec.signals, attr, None)
    if prefix == "recommendation":
        return getattr(rec, attr, None)
    return None


def _is_supported(value: Any) -> bool:
    """근거가 실제로 뒷받침되는가? False/0/빈컬렉션이면 미지원."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value) > 0
    return True


_EN_NEGATION = re.compile(r"\b(not|no|never)\b", re.IGNORECASE)


def _looks_negated(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("없", "아니")) or bool(_EN_NEGATION.search(text))


def _contradicts_value(text: str, value: Any) -> bool:
    # MVP 수준의 결정적 모순 검사: bool true 근거를 부정형 문장으로 설명하면 차단.
    if isinstance(value, bool) and value and _looks_negated(text):
        return True
    return False


def check_response(pack: "EvidencePack", response: "AdvisorResponse") -> list[str]:
    """검사 실패 목록을 반환. 빈 리스트면 통과."""
    failures: list[str] = []

    # 1. user_id 일치
    if response.user_id != pack.user_id:
        failures.append(f"user_id mismatch: {response.user_id} != {pack.user_id}")

    # 2. item_id가 pack 안에 있는가
    target = next((r for r in pack.recommendations if r.item_id == response.item_id), None)
    if target is None:
        failures.append(f"item_id not in recommendations: {response.item_id}")
        return failures  # target 없으면 이후 검사 불가

    # 3. evidence_ref 화이트리스트 + 4. 값이 실제로 뒷받침되는가
    allowed = pack.evidence_keys()
    for i, c in enumerate(response.claims):
        if c.evidence_ref not in allowed:
            failures.append(f"claim[{i}] evidence_ref not in whitelist: {c.evidence_ref}")
            continue
        value = get_evidence_value(pack, target, c.evidence_ref)
        if not _is_supported(value):
            failures.append(
                f"claim[{i}] evidence is empty/false at {c.evidence_ref}: value={value!r}"
            )
            continue
        if _contradicts_value(c.text, value):
            failures.append(
                f"claim[{i}] text contradicts {c.evidence_ref}: value={value!r}, text={c.text!r}"
            )

    return failures


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from advisor import get_client
    from advisor.schema import AdvisorResponse, Claim  # noqa: F811
    from evidence_pack import MockAdapter, build_evidence_pack, iter_evidence_pack

    mock = MockAdapter()
    build_evidence_pack(mock, mock, "data/_hg_test.jsonl", max_users=1)
    pack = next(iter_evidence_pack("data/_hg_test.jsonl"))
    rec0 = pack.recommendations[0]

    # 통과 케이스 - mock client는 항상 valid 응답을 만듦
    client = get_client(force_mock=True)
    good = client.generate(pack, rec0.item_id, "why")
    good_failures = check_response(pack, good)
    print(f"GOOD: passed={not good_failures} failures={good_failures}")

    # 실패 케이스 - 일부러 깨진 응답
    bad = AdvisorResponse(
        user_id=pack.user_id, item_id="non-existent-item",
        question_key="why", summary="가짜",
        claims=[Claim(text="없는 근거", evidence_ref="signals.fake_field")],
        decision_hint="", confidence_raw=0.99,
    )
    bad_failures = check_response(pack, bad)
    print(f"BAD: passed={not bad_failures} failures={bad_failures}")
