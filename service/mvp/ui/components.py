"""Streamlit UI 재사용 컴포넌트 — 추천 카드, 신뢰도 배지, 근거 칩.

streamlit 없이도 `trust_badge` / `evidence_chips` / `aggregate_packs_by`는 import·호출 가능.
streamlit이 필요한 함수(`recommendation_card`, `response_card`)는 호출 시점에 import한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import streamlit as st  # noqa: F401
    from advisor.schema import AdvisorResponse
    from evidence_pack.schema import EvidencePack, Recommendation


def _st():
    """Streamlit lazy import — 데모 실행 시점에만 필요."""
    import streamlit as st  # noqa: WPS433
    return st


# --- 신뢰도 배지 -----------------------------------------------------------

def trust_badge(confidence: float | None, label: str = "신뢰도") -> str:
    """0~1 confidence를 색 + 퍼센트 마크다운으로 변환."""
    if confidence is None:
        return f"⚪ {label} —"
    pct = int(round(confidence * 100))
    if confidence >= 0.7:
        emoji = "🟢"
    elif confidence >= 0.4:
        emoji = "🟡"
    else:
        emoji = "🔴"
    return f"{emoji} **{label} {pct}%**"


# --- 근거 칩 -------------------------------------------------------------

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


# --- 추천 카드 -----------------------------------------------------------

def recommendation_card(rec: "Recommendation", calibrated_conf: float | None = None) -> None:
    """추천 1건 카드 렌더. selectbox/button과 함께 호출되는 게 일반적."""
    st = _st()
    with st.container(border=True):
        head = f"#{rec.rank} · **{rec.brand or '—'}** · {rec.category_code or '—'}"
        price = f"${rec.price:.2f}" if rec.price is not None else "—"
        st.markdown(f"{head}  \n💵 {price}")
        chips = evidence_chips(rec)
        if chips:
            st.markdown(" ".join(chips))
        if calibrated_conf is not None:
            st.markdown(trust_badge(calibrated_conf))
        st.caption(f"`{rec.item_id}`")


# --- 응답 카드 (claims + flag + 신뢰도) ----------------------------------

def response_card(resp: "AdvisorResponse") -> None:
    """LLM 응답을 chat_message 안에 출력."""
    st = _st()
    conf = resp.confidence_calibrated if resp.confidence_calibrated is not None else resp.confidence_raw

    # 헤드라인
    pass_emoji = "✓" if resp.hard_gate_passed else "✗"
    sc = resp.self_check_score
    sc_str = f"일관성 {int(sc * 100)}%" if sc is not None else "일관성 —"
    st.markdown(f"**{resp.summary}**  \n{trust_badge(conf)}  ·  Hard-gate {pass_emoji}  ·  {sc_str}")

    # 주장
    if resp.claims:
        with st.expander("근거(claims)", expanded=True):
            for c in resp.claims:
                st.markdown(f"- {c.text}  \n  ↳ `{c.evidence_ref}`")

    # 플래그
    if resp.flags:
        for f in resp.flags:
            st.warning(f)

    # 결정 힌트
    if resp.decision_hint:
        st.info(f"💡 {resp.decision_hint}")


# --- 집계 메트릭 (운영자 탭용) -------------------------------------------

def aggregate_packs_by(packs: list["EvidencePack"], key: str = "category_l2") -> list[dict]:
    """카테고리/브랜드별 추천 노출, 전환, 최근 조회 트렌드를 집계."""
    agg: dict[str, dict] = {}
    for p in packs:
        for r in p.recommendations:
            k = getattr(r, key) or "(unknown)"
            d = agg.setdefault(
                k,
                {
                    "n_recs": 0,
                    "n_items": 0,
                    "sum_hit": 0.0,
                    "sum_recent_pop": 0.0,
                    "sum_v2c": 0.0,
                    "sum_v2p": 0.0,
                    "views": 0,
                    "carts": 0,
                    "purchases": 0,
                    "recent_views": 0,
                    "prev_views": 0,
                    "_items": set(),
                },
            )
            d["n_recs"] += 1
            d["sum_hit"] += r.signals.model_hit_count
            d["sum_recent_pop"] += r.signals.item_recent14_pop_log1p
            if r.item_id in d["_items"]:
                continue
            d["_items"].add(r.item_id)
            d["n_items"] += 1
            d["views"] += r.signals.item_view_count
            d["carts"] += r.signals.item_cart_count
            d["purchases"] += r.signals.item_purchase_count
            d["recent_views"] += r.signals.item_recent14_view_count
            d["prev_views"] += r.signals.item_prev14_view_count
            d["sum_v2c"] += r.signals.item_v2c_smoothed
            d["sum_v2p"] += r.signals.item_v2p_smoothed

    rows = []
    for k, d in agg.items():
        n_recs = d["n_recs"]
        n_items = d["n_items"]
        views = d["views"]
        trend = (d["recent_views"] + 1.0) / (d["prev_views"] + 1.0)
        rows.append(
            {
                key: k,
                "n_recs": n_recs,
                "n_items": n_items,
                "views": views,
                "carts": d["carts"],
                "purchases": d["purchases"],
                "raw_v2c": d["carts"] / views if views else 0.0,
                "raw_v2p": d["purchases"] / views if views else 0.0,
                "avg_v2c_smoothed": d["sum_v2c"] / n_items if n_items else 0.0,
                "avg_v2p_smoothed": d["sum_v2p"] / n_items if n_items else 0.0,
                "recent_views": d["recent_views"],
                "prev_views": d["prev_views"],
                "recent_trend_ratio": trend,
                "avg_recent_pop_log1p": d["sum_recent_pop"] / n_recs if n_recs else 0.0,
                "avg_hit_count": d["sum_hit"] / n_recs if n_recs else 0.0,
            }
        )
    rows.sort(key=lambda r: (r["recent_trend_ratio"], r["avg_v2p_smoothed"], r["recent_views"]), reverse=True)
    return rows

