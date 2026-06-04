"""SelfCheckGPT — N샘플 일관성으로 환각 탐지 (reference-free).

원 논문: Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
for Generative Large Language Models" (arXiv:2303.08896, 2023).

핵심 아이디어: 모델이 사실을 알면 다양한 샘플링에서도 일관된 답을 내고, 지어내면 흔들린다.
→ N개 샘플을 비교해 일관성 점수를 매긴다. 라벨/정답 불필요.

여기서 우리는 단순화된 변형을 쓴다:
- 같은 (pack, item_id, question)으로 N개 샘플 생성
- 각 sample의 evidence_ref 집합을 추출
- 첫 sample(target)의 각 ref가 다른 sample들에 얼마나 자주 등장하는지 측정
- ref별 일관성 평균 = 전체 self_check_score (1.0=완전 일관, 0.0=완전 불일치)
- 임계치 이하인 ref를 가진 claim에 ⚠ flag
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from advisor.schema import AdvisorResponse


def self_check_consistency(samples: list["AdvisorResponse"]) -> dict:
    """일관성 점수와 ref별 세부.

    Returns:
        {"score": float, "per_ref": dict[ref, float], "n_samples": int}
    """
    if not samples:
        return {"score": 0.0, "per_ref": {}, "n_samples": 0}
    if len(samples) == 1:
        return {"score": 1.0, "per_ref": {}, "n_samples": 1}

    target_refs = {c.evidence_ref for c in samples[0].claims}
    if not target_refs:
        return {"score": 0.0, "per_ref": {}, "n_samples": len(samples)}

    others = [{c.evidence_ref for c in s.claims} for s in samples[1:]]
    per_ref: dict[str, float] = {}
    for ref in target_refs:
        n_in = sum(1 for o in others if ref in o)
        per_ref[ref] = n_in / len(others) if others else 1.0

    score = sum(per_ref.values()) / len(per_ref)
    return {"score": score, "per_ref": per_ref, "n_samples": len(samples)}


def apply_self_check(
    samples: list["AdvisorResponse"],
    threshold: float = 0.5,
) -> "AdvisorResponse":
    """첫 sample을 target으로 잡고 self_check_score와 ⚠ flag를 채워 반환."""
    if not samples:
        raise ValueError("samples is empty")
    target = samples[0]
    result = self_check_consistency(samples)
    target.self_check_score = round(result["score"], 3)

    flags = list(target.flags)
    for c in target.claims:
        p = result["per_ref"].get(c.evidence_ref, 1.0)
        if p < threshold:
            flags.append(f"⚠ wobbly: {c.evidence_ref} ({p:.0%} 일관)")
    target.flags = flags
    return target


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from advisor import get_client
    from evidence_pack import MockAdapter, build_evidence_pack, iter_evidence_pack

    mock = MockAdapter()
    build_evidence_pack(mock, mock, "data/_sc_test.jsonl", max_users=2)
    packs = list(iter_evidence_pack("data/_sc_test.jsonl"))
    client = get_client(force_mock=True)

    for pack in packs:
        for rec in pack.recommendations[:3]:
            samples = client.generate_samples(pack, rec.item_id, "should_buy", n=4)
            result = self_check_consistency(samples)
            target = apply_self_check(samples)
            print(
                f"{pack.user_id} item={rec.item_id[-4:]} "
                f"score={target.self_check_score} "
                f"flags={target.flags} "
                f"per_ref={result['per_ref']}"
            )
