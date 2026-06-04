"""Evidence Pack 스키마 — LLM이 근거로 삼는 "단 하나의 진실 소스".

설계 원칙:
- LLM의 모든 주장은 이 스키마의 키 하나에 대응되어야 한다 (hard-gate 검증).
- 본체 추천기에 의존하지 않는다 — 보조 신호(인기도/히스토리/메타)는 train.parquet에서 자체 계산.
- 본체가 풍부한 신호(per-model rank, cart_boosted)를 주면 활용, 안 주면 기본값으로.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """사용자 단위 요약 — 추천 카드 전체의 컨텍스트."""

    total_events: int = 0
    recent14_events: int = 0
    top_categories: list[str] = Field(default_factory=list)
    top_brands: list[str] = Field(default_factory=list)
    carted_brands: list[str] = Field(default_factory=list)
    purchased_brands: list[str] = Field(default_factory=list)
    price_median: float | None = None
    price_p25: float | None = None
    price_p75: float | None = None
    price_iqr: float | None = None


class ItemSignals(BaseModel):
    """추천 1건에 대한 근거 신호 — LLM 주장의 `evidence_ref`가 가리키는 곳."""

    # 인기도 (train.parquet에서 자체 계산)
    item_pop_log1p: float = 0.0
    item_recent14_pop_log1p: float = 0.0

    # 모델 합의 (본체 어댑터에서 옵션으로 들어옴)
    models_recommending: list[str] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)  # 모델명 -> rank
    model_hit_count: int = 0
    ensemble_rank: int = 0

    # 본체 후처리 신호 (옵션)
    cart_boosted: bool = False

    # 사용자 과거 상호작용 (train.parquet에서 자체 계산)
    seen_before: bool = False
    carted_before: bool = False
    purchased_before: bool = False

    # EDA 기반 선호/전환 신호
    brand_affinity: bool = False
    category_l2_affinity: bool = False
    price_in_user_band: bool = False
    price_distance_to_user_median: float | None = None
    item_view_count: int = 0
    item_cart_count: int = 0
    item_purchase_count: int = 0
    item_v2c_smoothed: float = 0.0
    item_v2p_smoothed: float = 0.0
    item_recent14_view_count: int = 0
    item_prev14_view_count: int = 0
    item_recent_trend_ratio: float = 0.0

    # revisit 추천 유형 신호 — revisit 후보에만 채워짐
    revisit_score: float | None = None
    last_event_type: str | None = None
    last_interaction_days: int | None = None
    revisit_event_count: int | None = None
    cart_without_purchase: bool = False


class Recommendation(BaseModel):
    """추천 1건 = 아이템 메타 + 신호."""

    rank: int  # 1-indexed 최종 순위
    item_id: str
    # 표시용 별칭 — UI·prompt 표시 전용. evidence_keys()에 포함되지 않으며 LLM 주장 근거로 사용 불가.
    item_alias: str | None = None
    category_code: str | None = None
    category_l2: str | None = None
    brand: str | None = None
    price: float | None = None
    signals: ItemSignals


class EvidencePack(BaseModel):
    """사용자 1명에 대한 근거 묶음 = 사용자 컨텍스트 + 추천 top-K."""

    user_id: str
    # 표시용 별칭 — UI·prompt 표시 전용. evidence_keys()에 포함되지 않으며 LLM 주장 근거로 사용 불가.
    user_alias: str | None = None
    user_context: UserContext
    recommendations: list[Recommendation]

    def evidence_keys(self) -> set[str]:
        """LLM 주장의 evidence_ref로 허용되는 키 집합 (hard-gate 화이트리스트)."""
        keys: set[str] = set()
        for f in UserContext.model_fields:
            keys.add(f"user_context.{f}")
        for f in ItemSignals.model_fields:
            keys.add(f"signals.{f}")
        for f in ("rank", "item_id", "category_code", "category_l2", "brand", "price"):
            keys.add(f"recommendation.{f}")
        return keys


if __name__ == "__main__":
    # 스키마 데모
    pack = EvidencePack(
        user_id="demo-user",
        user_context=UserContext(total_events=120, recent14_events=18),
        recommendations=[
            Recommendation(
                rank=1, item_id="demo-item", category_code="apparel.shoes",
                category_l2="apparel.shoes", brand="kapika", price=72.0,
                signals=ItemSignals(item_recent14_pop_log1p=4.1, model_hit_count=2),
            )
        ],
    )
    print(pack.model_dump_json(indent=2))
    print("\nallowed evidence_ref keys:", sorted(pack.evidence_keys()))
