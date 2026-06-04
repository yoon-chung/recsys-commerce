"""orchestrator.py — 5섹션 통합 + dedup.

각 user_alias에 대해 5섹션 후보를 만들어 dedup·필터 후 반환.

섹션 (LLM_MVP_UNIFIED_ROADMAP.md §3):
    0. 모델 Top-10  (submission)
    1. 비슷한 분들이 산  (CF, 선택 — FAISS 없으면 빈 리스트)
    2. 내 취향  (content)
    3. 새로 나온  (recency)
    4. 다시 볼 만한  (revisit, seen ≥ 3 미만이면 숨김)

Dedup 체인:
    submission(0) > collaborative(1) > content(2) > recency(3)
    revisit(4)은 seen 풀이라 disjoint → dedup 미적용
"""

from __future__ import annotations

import sqlite3
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .recommenders import (
    _load_profile,
    recommend_profile_content,
    recommend_submission,
    recommend_unseen_recency,
)


REVISIT_MIN_SEEN = 3

# CF 점수 가중치 (이웃 행동별)
CF_EVENT_WEIGHT = {"purchase": 3.0, "cart": 1.0, "view": 0.0}


def _revisit_from_log(event_log: list[dict], top_n: int = 3) -> list[dict]:
    """evidence_pack.builder.recommend_revisit 위임 (lazy import로 순환 회피)."""
    from evidence_pack.builder import recommend_revisit
    return recommend_revisit(event_log, top_n=top_n)


def _cf_recommend_from_neighbors(
    target_seen: set[str],
    neighbor_aliases: list[str],
    db_path: str | Path,
    top_n: int,
) -> list[str]:
    """이웃 유저들의 purchase/cart 이력 → 점수 합산 → 본인 unseen만 → Top-N."""
    if not neighbor_aliases:
        return []
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in neighbor_aliases)
    rows = con.execute(
        f"SELECT carted_items, purchased_items FROM user_profiles WHERE user_alias IN ({placeholders})",
        neighbor_aliases,
    ).fetchall()
    con.close()

    scores: dict[str, float] = defaultdict(float)
    for carted_json, purch_json in rows:
        carted = json.loads(carted_json or "[]")
        purchased = json.loads(purch_json or "[]")
        for it in purchased:
            if it not in target_seen:
                scores[it] += CF_EVENT_WEIGHT["purchase"]
        for it in carted:
            if it not in target_seen:
                scores[it] += CF_EVENT_WEIGHT["cart"]

    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def recommend_all_sections(
    user_alias: str,
    recs: dict[str, list[str]],
    catalog: dict[str, dict],
    recency_pool: list[str],
    db_path: str | Path,
    top_per_section: int = 5,
    cf_neighbors: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """5섹션 후보 + dedup된 결과 반환.

    반환:
        {
          "submission":     [item_alias, ...]   # top_per_section
          "collaborative":  [item_alias, ...]   # dedup 적용, FAISS 없으면 []
          "content":        [item_alias, ...]   # dedup 적용
          "recency":        [item_alias, ...]   # dedup 적용
          "revisit":        [{item_id, revisit_score, ...}]  # 독립, seen<3이면 []
        }
    """
    sections: dict[str, Any] = {
        "submission": [],
        "collaborative": [],
        "content": [],
        "recency": [],
        "revisit": [],
    }

    profile = _load_profile(db_path, user_alias)
    if profile is None:
        return sections

    # Section 0: submission Top-10 (단독, dedup 시작점)
    sub_full = recommend_submission(user_alias, recs, top_n=10)
    sections["submission"] = sub_full[:top_per_section]
    used: set[str] = set(sections["submission"])

    # Section 1: CF — FAISS user-user neighbors가 있을 때만
    # cf_neighbors[user_alias] = list[neighbor_user_alias] (FAISS Top-K)
    # 이웃들의 purchase/cart 이력 → 점수 합산 → 본인 unseen만
    if cf_neighbors is not None:
        target_seen = set(profile.get("seen_items") or [])
        neighbor_aliases = cf_neighbors.get(user_alias, [])
        cf_raw = _cf_recommend_from_neighbors(
            target_seen, neighbor_aliases, db_path,
            top_n=top_per_section + len(used),
        )
        cf_deduped = [a for a in cf_raw if a not in used][:top_per_section]
        sections["collaborative"] = cf_deduped
        used.update(cf_deduped)

    # Section 2: Content — submission_top 인자에는 full 10개 전달 (점수 계산용)
    content_raw = recommend_profile_content(
        user_alias, db_path, catalog, sub_full, top_n=top_per_section + len(used)
    )
    content_deduped = [a for a in content_raw if a not in used][:top_per_section]
    sections["content"] = content_deduped
    used.update(content_deduped)

    # Section 3: Recency
    recency_raw = recommend_unseen_recency(
        user_alias, db_path, catalog, recency_pool, sub_full,
        top_n=top_per_section + len(used),
    )
    recency_deduped = [a for a in recency_raw if a not in used][:top_per_section]
    sections["recency"] = recency_deduped
    used.update(recency_deduped)

    # Section 4: Revisit (seen 풀, 독립)
    event_log = profile.get("event_log") or []
    if len(profile.get("seen_items") or []) >= REVISIT_MIN_SEEN and event_log:
        sections["revisit"] = _revisit_from_log(event_log, top_n=top_per_section)

    return sections