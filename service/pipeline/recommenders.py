"""recommenders.py — 5섹션 중 submission/content/recency 엔진 (Python 규칙 기반, LLM 미사용).

각 엔진은 item_alias 리스트를 반환한다. orchestrator.py가 dedup·취합해 최종 5섹션을 만든다.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────

def _load_profile(db_path: str | Path, user_alias: str) -> dict | None:
    con = sqlite3.connect(db_path)
    cur = con.execute(
        "SELECT * FROM user_profiles WHERE user_alias = ?", (user_alias,)
    )
    row = cur.fetchone()
    col_names = [d[0] for d in cur.description]
    con.close()
    if row is None:
        return None
    profile = dict(zip(col_names, row))
    for key in ("top_categories", "top_brands", "carted_brands", "purchased_brands",
                "seen_items", "carted_items", "purchased_items", "event_log"):
        if key in profile:
            profile[key] = json.loads(profile[key] or "[]")
    return profile


# ── 유형 1: Submission 기반 ───────────────────────────────────────────────────

def recommend_submission(
    user_alias: str,
    recs: dict[str, list[str]],
    top_n: int = 10,
) -> list[str]:
    """submission 기반 Top-N 반환."""
    return recs.get(user_alias, [])[:top_n]


# ── 유형 2: 프로필 기반 Content ───────────────────────────────────────────────

def recommend_profile_content(
    user_alias: str,
    db_path: str | Path,
    catalog: dict[str, dict],
    submission_top: list[str],
    top_n: int = 5,
) -> list[str]:
    """유저 프로필의 카테고리/브랜드/가격대와 매칭되는 unseen 아이템 추천."""
    profile = _load_profile(db_path, user_alias)
    if not profile:
        return []

    seen = set(profile["seen_items"])
    top_cats = set(profile["top_categories"])
    pref_brands = set(profile["top_brands"]) | set(profile["carted_brands"])
    median = profile.get("price_median")
    iqr = profile.get("price_iqr")

    sub_rank = {iid: i for i, iid in enumerate(submission_top)}

    def _score(item_alias: str, meta: dict) -> float:
        score = 0.0
        cat_l2 = (meta.get("category_l2") or "").split(".")[0:2]
        cat_l2_str = ".".join(cat_l2)
        if cat_l2_str and cat_l2_str in top_cats:
            score += 2.0
        if meta.get("brand") and meta["brand"] in pref_brands:
            score += 2.0
        price = meta.get("price")
        if price is not None and median is not None and iqr is not None and iqr > 0:
            if abs(float(price) - float(median)) <= 1.5 * float(iqr):
                score += 1.0
        if item_alias in sub_rank:
            score += 1.0 - sub_rank[item_alias] / max(len(submission_top), 1)
        return score

    candidates = [
        (alias, _score(alias, meta))
        for alias, meta in catalog.items()
        if alias not in seen and _score(alias, meta) > 0
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    # fallback: 점수 0이면 unseen + submission 순위만
    if not candidates:
        candidates = [
            (alias, -sub_rank.get(alias, len(submission_top)))
            for alias in catalog
            if alias not in seen
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

    return [alias for alias, _ in candidates[:top_n]]


# ── 유형 3: 신상품/Unseen × Recency ──────────────────────────────────────────

def recommend_unseen_recency(
    user_alias: str,
    db_path: str | Path,
    catalog: dict[str, dict],
    recency_items: list[str],
    submission_top: list[str],
    top_n: int = 5,
) -> list[str]:
    """아직 보지 않은 최신 상품 추천 (unseen × recency 교집합)."""
    profile = _load_profile(db_path, user_alias)
    if not profile:
        return recency_items[:top_n]

    seen = set(profile["seen_items"])
    top_cats = set(profile["top_categories"])
    sub_rank = {iid: i for i, iid in enumerate(submission_top)}

    unseen_recency = [iid for iid in recency_items if iid not in seen]

    def _score(item_alias: str) -> float:
        meta = catalog.get(item_alias, {})
        score = 0.0
        cat_l2 = ".".join((meta.get("category_l2") or "").split(".")[:2])
        if cat_l2 in top_cats:
            score += 2.0
        score -= sub_rank.get(item_alias, len(submission_top)) * 0.01
        return score

    candidates = sorted(unseen_recency, key=_score, reverse=True)

    # fallback: 카테고리 필터 없이 unseen × recency 순서
    if not candidates:
        candidates = unseen_recency

    return candidates[:top_n]
