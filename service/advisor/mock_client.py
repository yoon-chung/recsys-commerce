"""Mock LLM 클라이언트 — 결정적, 비용 0, 본체 없이 데모 가능.

설계: Evidence Pack의 신호를 직접 읽어 그럴듯한 응답을 만든다. 따라서:
- 신호가 강하면 confidence↑, claims↑ — 실 LLM과 같은 방향
- sample_idx마다 미세하게 다른 claims/confidence를 내, SelfCheckGPT가 일관성을 측정해볼 수 있게 함
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .schema import AdvisorResponse, Claim

if TYPE_CHECKING:
    from evidence_pack.schema import EvidencePack, Recommendation


def _find_target(pack: "EvidencePack", item_id: str) -> "Recommendation":
    for r in pack.recommendations:
        if r.item_id == item_id:
            return r
    return pack.recommendations[0]


def _build_claims(
    pack: "EvidencePack",
    target: "Recommendation",
    sample_idx: int,
    question_key: str = "why",
) -> list[Claim]:
    """결정적이지만 sample_idx에 따라 살짝 흔들리는 claim 집합."""
    sigs = target.signals
    ctx = pack.user_context
    claims: list[Claim] = []

    if question_key == "revisit":
        if sigs.cart_without_purchase:
            claims.append(Claim(
                text="장바구니에 담아두셨던 상품입니다",
                evidence_ref="signals.cart_without_purchase",
            ))
        if sigs.last_interaction_days is not None and sigs.last_interaction_days <= 7:
            claims.append(Claim(
                text="최근 관심 가지셨던 상품입니다",
                evidence_ref="signals.last_interaction_days",
            ))
        elif sigs.last_event_type:
            claims.append(Claim(
                text=f"마지막 상호작용은 {sigs.last_event_type}였습니다",
                evidence_ref="signals.last_event_type",
            ))
        if sigs.revisit_event_count is not None and sigs.revisit_event_count >= 3:
            claims.append(Claim(
                text="여러 번 확인하신 상품입니다",
                evidence_ref="signals.revisit_event_count",
            ))
        if sigs.revisit_score is not None:
            claims.append(Claim(
                text="과거 행동 기준 재방문 점수가 있습니다",
                evidence_ref="signals.revisit_score",
            ))
        if not claims:
            claims.append(Claim(text="재방문 이력 기반 후보입니다", evidence_ref="recommendation.rank"))
        return claims[:4]

    # 강한 신호 — 모든 샘플에 들어감 (consistent)
    if sigs.item_recent14_pop_log1p > 3.0:
        claims.append(Claim(
            text="최근 14일 인기가 높습니다",
            evidence_ref="signals.item_recent14_pop_log1p",
        ))

    if sigs.model_hit_count >= 2:
        models_str = "·".join(sigs.models_recommending[:2])
        claims.append(Claim(
            text=f"{sigs.model_hit_count}개 모델({models_str})이 함께 추천했습니다",
            evidence_ref="signals.model_hit_count",
        ))

    if sigs.carted_before:
        claims.append(Claim(
            text="이전에 장바구니에 담은 적이 있습니다",
            evidence_ref="signals.carted_before",
        ))
    elif sigs.seen_before:
        claims.append(Claim(
            text="이전에 본 적이 있는 상품입니다",
            evidence_ref="signals.seen_before",
        ))

    if sigs.price_in_user_band:
        claims.append(Claim(
            text="평소 보던 가격대 안에 있습니다",
            evidence_ref="signals.price_in_user_band",
        ))

    if sigs.brand_affinity:
        claims.append(Claim(
            text=f"평소 관심 브랜드({target.brand})와 맞습니다",
            evidence_ref="signals.brand_affinity",
        ))
    elif target.brand and target.brand in (ctx.carted_brands or []):
        claims.append(Claim(
            text=f"평소 담아두던 브랜드({target.brand})입니다",
            evidence_ref="user_context.carted_brands",
        ))

    if sigs.category_l2_affinity:
        claims.append(Claim(
            text=f"자주 보던 카테고리({target.category_l2 or target.category_code})와 맞습니다",
            evidence_ref="signals.category_l2_affinity",
        ))

    # 약한 신호 — sample_idx에 따라 들어갔다 빠짐 (inconsistent, SelfCheck가 잡아냄)
    wobble = (sample_idx + _hash_int(target.item_id)) % 3
    if wobble == 0 and 1.5 < sigs.item_pop_log1p <= 3.0 and not any(c.evidence_ref == "signals.item_pop_log1p" for c in claims):
        claims.append(Claim(
            text="전체 인기가 어느 정도 있습니다",
            evidence_ref="signals.item_pop_log1p",
        ))

    if not claims:  # 최후 안전망 — 비어 있으면 pydantic 검증 실패
        claims.append(Claim(
            text=f"앙상블 {target.rank}위로 추천되었습니다",
            evidence_ref="recommendation.rank",
        ))
    return claims[:4]


def _summary_for(question_key: str, target: "Recommendation", n_claims: int) -> str:
    brand = target.brand or "이 상품"
    cat = target.category_code or "이 카테고리"
    summaries = {
        "why": f"{brand}의 {cat} 상품이며, {n_claims}가지 근거가 일치합니다.",
        "should_buy": (
            "근거가 충분해 담아두기를 권합니다." if n_claims >= 2 else "근거가 약하니 조금 더 살펴보세요."
        ),
        "cheaper": f"현재 가격은 {target.price}이며, 더 저렴한 대안 비교에는 추가 정보가 필요합니다.",
        "fit_preference": (
            "취향 신호가 일치합니다." if n_claims >= 2 else "취향 일치 신호는 약합니다."
        ),
        "revisit": (
            "다시 볼 만한 과거 관심 신호가 있습니다." if n_claims >= 2 else "재방문 근거는 제한적입니다."
        ),
        "category_trend": f"{cat} 카테고리에서 최근 조회 추세와 전환 근거를 확인했습니다.",
        "promotion_candidate": (
            "조회·전환·취향 신호가 있어 프로모션 후보로 적합합니다." if n_claims >= 2
            else "프로모션 후보로는 근거가 약합니다."
        ),
        "discontinue_candidate": (
            "추천 합의가 약해 단종 검토 여지가 있습니다." if n_claims < 2 else "단종 후보로는 부적합합니다."
        ),
    }
    return summaries.get(question_key, f"{brand} — {n_claims}가지 근거 확인.")


def _decision_hint(confidence: float, question_key: str) -> str:
    if question_key == "should_buy":
        return "담아두고 가격 변동 확인" if confidence > 0.65 else "추가 조사 권장"
    if question_key == "revisit":
        return "장바구니 상태와 가격을 다시 확인" if confidence > 0.65 else "비슷한 상품과 비교"
    if question_key == "promotion_candidate":
        return "소규모 A/B 실험 권장" if confidence > 0.65 else "보류"
    if confidence > 0.7:
        return "현재 근거로 충분"
    return "추가 정보 필요"


def _hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:6], 16)


class MockClient:
    """결정적 mock — sample_idx에 따라 미세하게 변동되어 SelfCheck 데모 가능."""

    model = "mock-solar"

    def generate(
        self,
        pack: "EvidencePack",
        item_id: str,
        question_key: str,
        temperature: float = 0.4,
        sample_idx: int = 0,
    ) -> AdvisorResponse:
        target = _find_target(pack, item_id)
        claims = _build_claims(pack, target, sample_idx, question_key)
        n_claims = len(claims)

        # 신호 강도 기반 confidence — 같은 입력엔 같은 confidence (sample_idx도 반영)
        base = min(0.95, 0.35 + 0.13 * n_claims)
        if question_key == "revisit" and target.signals.cart_without_purchase:
            base = min(0.95, base + 0.12)
        salt_noise = (((sample_idx + _hash_int(target.item_id)) % 5) - 2) * 0.03
        conf = max(0.05, min(0.99, base + salt_noise))

        return AdvisorResponse(
            user_id=pack.user_id,
            item_id=target.item_id,
            question_key=question_key,
            summary=_summary_for(question_key, target, n_claims),
            claims=claims,
            decision_hint=_decision_hint(conf, question_key),
            confidence_raw=round(conf, 3),
        )

    def generate_samples(
        self,
        pack: "EvidencePack",
        item_id: str,
        question_key: str,
        n: int = 3,
        temperature: float = 0.7,
    ) -> list[AdvisorResponse]:
        return [self.generate(pack, item_id, question_key, temperature, sample_idx=i) for i in range(n)]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evidence_pack import MockAdapter, build_evidence_pack, iter_evidence_pack

    mock = MockAdapter(top_k=5)
    build_evidence_pack(mock, mock, "data/_mockclient_test.jsonl", max_users=1)
    pack = next(iter_evidence_pack("data/_mockclient_test.jsonl"))
    c = MockClient()
    resp = c.generate(pack, pack.recommendations[0].item_id, "why")
    print(resp.model_dump_json(indent=2))
    print("\n--- 3 samples (SelfCheck용) ---")
    for r in c.generate_samples(pack, pack.recommendations[0].item_id, "should_buy", n=3):
        print(f"conf={r.confidence_raw} claims={[c.evidence_ref for c in r.claims]}")
