"""ui/app.py — Streamlit MVP: 사이드바 유저 선택 + 5섹션 + LangGraph 채팅.

실행:
    streamlit run service/mvp/ui/app.py

사전 조건 (service/mvp/data/ 안에 존재):
    id_aliases.json · user_recommendations.json · item_catalog.json
    recency_pool.json · evidence_pack.jsonl · user_profiles.db

API 키:
    .env 의 UPSTAGE_API_KEY 있으면 Solar Pro 호출, 없으면 mock으로 폴백.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advisor import QUICK_QUESTIONS
from evidence_pack.schema import EvidencePack
from langgraph.checkpoint.memory import MemorySaver
from pipeline.graph import build_app
from ui.components import evidence_chips, trust_badge

DATA_DIR = ROOT / "data"


# ── 리소스 로딩 (Streamlit 캐시) ────────────────────────────────────────────

@st.cache_resource(show_spinner="LangGraph + Evidence Pack 로딩 중...")
def _bootstrap_app():
    memory = MemorySaver()
    app, resources = build_app(data_dir=DATA_DIR, checkpointer=memory)
    return app, resources


@st.cache_data(show_spinner=False)
def _load_id_aliases() -> dict:
    with (DATA_DIR / "id_aliases.json").open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_item_images() -> dict:
    """item_alias → image URL 매핑 (있을 때만 표시)."""
    path = DATA_DIR / "item_images.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_profile(db_path: Path, user_alias: str) -> dict | None:
    con = sqlite3.connect(db_path)
    cur = con.execute("SELECT * FROM user_profiles WHERE user_alias=?", (user_alias,))
    row = cur.fetchone()
    cols = [d[0] for d in cur.description]
    con.close()
    if row is None:
        return None
    prof = dict(zip(cols, row))
    for k in ("top_categories", "top_brands", "carted_brands", "purchased_brands",
              "seen_items", "carted_items", "purchased_items", "event_log"):
        if k in prof:
            prof[k] = json.loads(prof[k] or "[]")
    return prof


# ── UI 빌딩 블록 ────────────────────────────────────────────────────────────

def _render_profile_card(profile: dict, user_alias: str) -> None:
    with st.container(border=True):
        st.markdown(f"### 👤 {user_alias}")
        c1, c2, c3 = st.columns(3)
        c1.metric("총 이벤트", profile.get("total_events", 0))
        c2.metric("최근 14일", profile.get("recent14_events", 0))
        c3.metric("seen items", len(profile.get("seen_items") or []))
        cats = profile.get("top_categories") or []
        brands = profile.get("top_brands") or []
        if cats:
            st.markdown(f"**관심 카테고리**: {', '.join(cats[:3])}")
        if brands:
            st.markdown(f"**관심 브랜드**: {', '.join(brands[:3])}")
        pm = profile.get("price_median")
        if pm is not None:
            st.markdown(f"**가격대 (중앙값)**: ${pm:.2f}")


def _simple_item_card(
    alias: str, catalog: dict, rank: int | None = None,
    image_map: dict | None = None,
) -> None:
    meta = catalog.get(alias, {})
    with st.container(border=True):
        if image_map and (img := image_map.get(alias)):
            st.image(img, use_container_width=True)
        head = f"#{rank} · " if rank else ""
        st.markdown(
            f"{head}**{meta.get('brand') or '—'}** · "
            f"{meta.get('category_l2') or '—'}"
        )
        price = meta.get("price")
        if price is not None:
            st.markdown(f"💵 ${price:.2f}")
        recent14 = meta.get("recent14_events", 0)
        if recent14 > 0:
            st.caption(f"📈 최근 14일 노출 {recent14:,}")
        st.caption(f"`{alias}`")


def _render_revisit_card(item: dict, catalog: dict, image_map: dict | None = None) -> None:
    alias = item["item_id"]
    meta = catalog.get(alias, {})
    with st.container(border=True):
        if image_map and (img := image_map.get(alias)):
            st.image(img, use_container_width=True)
        st.markdown(f"**{meta.get('brand') or '—'}** · {meta.get('category_l2') or '—'}")
        score = item.get("revisit_score", 0)
        last_type = item.get("last_event_type", "?")
        days = item.get("last_interaction_days", "?")
        st.markdown(f"↩ revisit_score **{score}** · last: {last_type} ({days}일 전)")
        st.caption(f"`{alias}`")


def _render_section(name: str, label: str, items: list, catalog: dict, image_map: dict) -> None:
    if not items:
        return
    with st.expander(f"{label} ({len(items)})", expanded=(name == "submission")):
        cols = st.columns(min(len(items), 3))
        for i, item in enumerate(items):
            col = cols[i % len(cols)]
            with col:
                if name == "revisit":
                    _render_revisit_card(item, catalog, image_map)
                else:
                    _simple_item_card(item, catalog, rank=i + 1, image_map=image_map)


def _render_submission_with_signals(pack: EvidencePack | None, image_map: dict) -> None:
    """submission 섹션은 EvidencePack의 Recommendation 객체로 chips까지 렌더."""
    if pack is None or not pack.recommendations:
        return
    with st.expander(f"🎯 모델 Top-{len(pack.recommendations[:5])}", expanded=True):
        cols = st.columns(min(len(pack.recommendations[:5]), 3))
        for i, rec in enumerate(pack.recommendations[:5]):
            with cols[i % len(cols)]:
                with st.container(border=True):
                    alias = rec.item_alias or rec.item_id
                    if (img := image_map.get(alias)):
                        st.image(img, use_container_width=True)
                    st.markdown(
                        f"#{rec.rank} · **{rec.brand or '—'}** · "
                        f"{(rec.category_code or '—')[:25]}"
                    )
                    if rec.price is not None:
                        st.markdown(f"💵 ${rec.price:.2f}")
                    chips = evidence_chips(rec)
                    if chips:
                        st.markdown(" ".join(chips[:4]))
                    st.caption(f"`{alias}`")


# ── 메인 앱 ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Commerce AI Advisor", page_icon="🛍", layout="wide")
    st.title("🛍 Commerce AI Advisor")
    st.caption("Evidence Pack + trust_gate · LangGraph multi-turn · Solar Pro narration")

    app, resources = _bootstrap_app()
    aliases = _load_id_aliases()
    image_map = _load_item_images()
    users_map = aliases.get("users", {})

    # ── 세션 상태 초기화 ────────────────────────────────────────────────
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── 사이드바: 유저 선택 ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 데모 유저")
        # evidence_pack에 있는 유저만 dropdown에 (검증 가능 범위)
        valid_aliases = sorted(resources.pack_index.keys())
        if not valid_aliases:
            st.error("evidence_pack.jsonl에 유저가 없습니다. Phase 0b를 실행하세요.")
            return
        selected = st.selectbox(
            "선택", valid_aliases[:50],
            help=f"evidence_pack에 있는 {len(valid_aliases)}명 중 처음 50명",
        )
        st.markdown("---")
        st.markdown(f"**API**: {'Solar Pro' if os.getenv('UPSTAGE_API_KEY') else 'Mock'}")
        st.markdown(f"**thread_id**: `{st.session_state.thread_id[:8]}...`")
        if st.button("대화 초기화"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()

    # ── 프로필 카드 ──────────────────────────────────────────────────────
    profile = _load_profile(resources.profile_db_path, selected)
    if profile:
        _render_profile_card(profile, selected)
    else:
        st.warning(f"{selected} 프로필이 DB에 없습니다 (sample 빌드 범위 외).")
        return

    # ── 5섹션 추천 ──────────────────────────────────────────────────────
    st.markdown("### 추천")
    # orchestrator 직접 호출 — UI 진입 시 즉시 표시
    from pipeline.orchestrator import recommend_all_sections
    sections = recommend_all_sections(
        selected, resources.recs, resources.catalog, resources.recency_pool,
        resources.profile_db_path, top_per_section=3,
    )

    pack = resources.pack_index.get(selected)
    _render_submission_with_signals(pack, image_map)
    _render_section("collaborative", "👥 비슷한 분들이 구매한 상품", sections.get("collaborative") or [], resources.catalog, image_map)
    _render_section("content", "🧭 내 취향에 맞는 상품", sections.get("content") or [], resources.catalog, image_map)
    _render_section("recency", "✨ 새로 나온 상품", sections.get("recency") or [], resources.catalog, image_map)
    _render_section("revisit", "↩ 다시 살펴볼 상품", sections.get("revisit") or [], resources.catalog, image_map)

    # ── 채팅 영역 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 질문하기")

    # 빠른 질문 버튼
    cols = st.columns(4)
    quick = list(QUICK_QUESTIONS.items())[:4]
    for col, (qkey, qtext) in zip(cols, quick):
        if col.button(qtext, key=f"q_{qkey}"):
            st.session_state._queued_msg = (qtext, qkey)

    # 채팅 히스토리
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("badge"):
                st.caption(f"trust: **{entry['badge']}**")

    # 채팅 입력
    user_input = st.chat_input(f"{selected}에게 질문...")
    queued = st.session_state.pop("_queued_msg", None)
    if queued:
        user_input, question_key = queued
    else:
        question_key = "why"

    if user_input:
        # user_alias는 state로 별도 전달 — message에 prepend하면 intent_router가
        # alias 패턴을 감지해 일반 질문도 shopping path로 hijack됨
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = app.invoke(
            {"message": user_input, "question_key": question_key, "user_alias": selected},
            config=cfg,
        )
        resp = result.get("response_text") or "(응답 없음)"
        badge = result.get("trust_badge") or "unverified"
        st.session_state.chat_history.append({"role": "assistant", "content": resp, "badge": badge})

        # advisor_response가 있으면 claims도 표시
        ar = result.get("advisor_response")
        if ar and ar.get("claims"):
            claims_md = "\n".join(
                f"- {c['text']} `({c['evidence_ref']})`" for c in ar["claims"][:4]
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**근거 (claims)**:\n{claims_md}",
            })
        st.rerun()


if __name__ == "__main__":
    main()