"""Streamlit UI 컴포넌트 — 추천 카드의 근거 칩."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evidence_pack.schema import Recommendation


def evidence_chips(rec: "Recommendation") -> list[str]:
    """추천 카드 하단에 표시할 칩(이모지+라벨) 리스트."""
    chips: list[str] = []
    s = rec.signals
    if s.item_recent14_pop_log1p > 3.0:
        chips.append("📈 인기↑")
    if s.model_hit_count >= 2:
        chips.append(f"🤝 모델합의 {s.model_hit_count}")
    if s.cart_boosted:
        chips.append("🛒 cart-boost")
    if s.revisit_score is not None:
        chips.append("↩ 재방문")
    if s.carted_before:
        chips.append("🛒 담은적있음")
    elif s.seen_before:
        chips.append("👁 본적있음")
    if s.purchased_before:
        chips.append("✅ 구매이력")
    if s.price_in_user_band:
        chips.append("💵 가격대맞음")
    if s.brand_affinity:
        chips.append("🏷 브랜드선호")
    if s.category_l2_affinity:
        chips.append("🧭 카테고리선호")
    return chips