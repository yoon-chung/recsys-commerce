"""Evidence Pack 빌더 — 어댑터 + 카탈로그 → JSONL 산출물.

CLI 사용 예:

    # 본체 없이 (mock 데이터로 끝까지)
    python -m evidence_pack.builder --adapter mock --out data/evidence_pack.jsonl

    # 본체 완성 후 (submission CSV + train.parquet)
    python -m evidence_pack.builder \\
        --adapter csv --submission-csv /path/to/submission.csv \\
        --catalog parquet --train-parquet /path/to/train.parquet \\
        --out data/evidence_pack.jsonl --max-users 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

from .adapter import (
    CatalogSource,
    CSVAdapter,
    MockAdapter,
    ParquetCatalogSource,
    RecsysAdapter,
    category_l2_of,
)
from .schema import EvidencePack, ItemSignals, Recommendation, UserContext


EVENT_WEIGHT = {"cart": 1.0, "view": 0.2, "purchase": 0.3}
HALF_LIFE_DAYS = 14


def recommend_revisit(event_history: list[dict], top_n: int = 3) -> list[dict]:
    """seen 이벤트 풀에서 TIFU 스타일 재방문 후보를 뽑는다."""
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    last_event: dict[str, dict] = {}
    recent_purchase_items = {
        str(ev["item_id"])
        for ev in event_history
        if ev.get("event_type") == "purchase" and int(ev.get("days_ago", 0)) <= 7
    }

    for ev in event_history:
        iid = str(ev["item_id"])
        if iid in recent_purchase_items:
            continue
        etype = str(ev["event_type"])
        days = int(ev.get("days_ago", 0))

        w = EVENT_WEIGHT.get(etype, 0.1)
        decay = 1.0 / (1.0 + days / HALF_LIFE_DAYS)
        scores[iid] = scores.get(iid, 0.0) + w * decay
        counts[iid] = counts.get(iid, 0) + 1

        if iid not in last_event or days < last_event[iid]["days"]:
            last_event[iid] = {"type": etype, "days": days}

    for iid in scores:
        scores[iid] *= 1.0 + 0.25 * min(counts[iid] - 1, 3)

    max_score = max(scores.values()) if scores else 1.0
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
    purchase_set = {str(ev["item_id"]) for ev in event_history if ev.get("event_type") == "purchase"}

    return [
        {
            "item_id": iid,
            "revisit_score": round(scores[iid] / max_score, 3),
            "last_event_type": last_event[iid]["type"],
            "last_interaction_days": last_event[iid]["days"],
            "revisit_event_count": counts[iid],
            "cart_without_purchase": (
                last_event[iid]["type"] == "cart" and iid not in purchase_set
            ),
        }
        for iid in ranked
    ]


def _price_fit(price: float | None, user_ctx: dict) -> tuple[bool, float | None]:
    median = user_ctx.get("price_median")
    iqr = user_ctx.get("price_iqr")
    if price is None or median is None:
        return False, None
    distance = abs(float(price) - float(median)) / max(abs(float(median)), 1.0)
    if iqr is None:
        return False, distance
    lo = float(median) - 1.5 * float(iqr)
    hi = float(median) + 1.5 * float(iqr)
    return lo <= float(price) <= hi, distance


def _build_one(
    user_id: str,
    rec: RecsysAdapter,
    cat: CatalogSource,
) -> EvidencePack | None:
    """사용자 1명에 대해 EvidencePack 1개를 만든다. top-K가 없으면 None."""
    top = rec.get_top_k(user_id)
    if not top:
        return None
    rich = rec.get_rich_signals(user_id) or {}
    catalog = cat.get_catalog()
    pop = cat.get_popularity()
    history = cat.get_user_history(user_id)
    event_history = cat.get_event_history(user_id) if hasattr(cat, "get_event_history") else []
    user_ctx = cat.get_user_context(user_id)

    recommendations: list[Recommendation] = []
    for rank, item_id in enumerate(top, 1):
        meta = catalog.get(item_id, {})
        p = pop.get(item_id, {})
        r = rich.get(item_id, {}) if rich else {}
        ranks: dict[str, int] = r.get("ranks", {}) or {}
        category_l2 = meta.get("category_l2") or category_l2_of(meta.get("category_code"))
        price = meta.get("price")
        price_in_band, price_distance = _price_fit(price, user_ctx)
        top_brands = set(user_ctx.get("top_brands", []) or [])
        affinity_brands = top_brands | set(user_ctx.get("carted_brands", []) or []) | set(user_ctx.get("purchased_brands", []) or [])
        top_categories = set(user_ctx.get("top_categories", []) or [])
        signals = ItemSignals(
            item_pop_log1p=float(p.get("item_pop_log1p", 0.0)),
            item_recent14_pop_log1p=float(p.get("item_recent14_pop_log1p", 0.0)),
            models_recommending=list(ranks.keys()),
            ranks=ranks,
            model_hit_count=len(ranks),
            ensemble_rank=int(r.get("ensemble_rank", rank)),
            cart_boosted=bool(r.get("cart_boosted", False)),
            seen_before=item_id in history.get("viewed", set()),
            carted_before=item_id in history.get("carted", set()),
            purchased_before=item_id in history.get("purchased", set()),
            brand_affinity=bool(meta.get("brand") and meta.get("brand") in affinity_brands),
            category_l2_affinity=bool(category_l2 and category_l2 in top_categories),
            price_in_user_band=price_in_band,
            price_distance_to_user_median=price_distance,
            item_view_count=int(p.get("item_view_count", 0)),
            item_cart_count=int(p.get("item_cart_count", 0)),
            item_purchase_count=int(p.get("item_purchase_count", 0)),
            item_v2c_smoothed=float(p.get("item_v2c_smoothed", 0.0)),
            item_v2p_smoothed=float(p.get("item_v2p_smoothed", 0.0)),
            item_recent14_view_count=int(p.get("item_recent14_view_count", 0)),
            item_prev14_view_count=int(p.get("item_prev14_view_count", 0)),
            item_recent_trend_ratio=float(p.get("item_recent_trend_ratio", 0.0)),
        )
        recommendations.append(
            Recommendation(
                rank=rank,
                item_id=item_id,
                category_code=meta.get("category_code"),
                category_l2=category_l2,
                brand=meta.get("brand"),
                price=price,
                signals=signals,
            )
        )

    existing_items = {r.item_id for r in recommendations}
    revisit_candidates = [
        r for r in recommend_revisit(event_history, top_n=len(top) + 3)
        if r["item_id"] not in existing_items
    ][:3]
    for offset, revisit in enumerate(revisit_candidates, 1):
        item_id = revisit["item_id"]
        rank = len(recommendations) + 1
        meta = catalog.get(item_id, {})
        p = pop.get(item_id, {})
        category_l2 = meta.get("category_l2") or category_l2_of(meta.get("category_code"))
        price = meta.get("price")
        price_in_band, price_distance = _price_fit(price, user_ctx)
        top_brands = set(user_ctx.get("top_brands", []) or [])
        affinity_brands = top_brands | set(user_ctx.get("carted_brands", []) or []) | set(user_ctx.get("purchased_brands", []) or [])
        top_categories = set(user_ctx.get("top_categories", []) or [])
        signals = ItemSignals(
            item_pop_log1p=float(p.get("item_pop_log1p", 0.0)),
            item_recent14_pop_log1p=float(p.get("item_recent14_pop_log1p", 0.0)),
            models_recommending=[],
            ranks={},
            model_hit_count=0,
            ensemble_rank=rank,
            cart_boosted=False,
            seen_before=item_id in history.get("viewed", set()),
            carted_before=item_id in history.get("carted", set()),
            purchased_before=item_id in history.get("purchased", set()),
            brand_affinity=bool(meta.get("brand") and meta.get("brand") in affinity_brands),
            category_l2_affinity=bool(category_l2 and category_l2 in top_categories),
            price_in_user_band=price_in_band,
            price_distance_to_user_median=price_distance,
            item_view_count=int(p.get("item_view_count", 0)),
            item_cart_count=int(p.get("item_cart_count", 0)),
            item_purchase_count=int(p.get("item_purchase_count", 0)),
            item_v2c_smoothed=float(p.get("item_v2c_smoothed", 0.0)),
            item_v2p_smoothed=float(p.get("item_v2p_smoothed", 0.0)),
            item_recent14_view_count=int(p.get("item_recent14_view_count", 0)),
            item_prev14_view_count=int(p.get("item_prev14_view_count", 0)),
            item_recent_trend_ratio=float(p.get("item_recent_trend_ratio", 0.0)),
            revisit_score=float(revisit["revisit_score"]),
            last_event_type=revisit["last_event_type"],
            last_interaction_days=int(revisit["last_interaction_days"]),
            revisit_event_count=int(revisit["revisit_event_count"]),
            cart_without_purchase=bool(revisit["cart_without_purchase"]),
        )
        recommendations.append(
            Recommendation(
                rank=rank,
                item_id=item_id,
                category_code=meta.get("category_code"),
                category_l2=category_l2,
                brand=meta.get("brand"),
                price=price,
                signals=signals,
            )
        )

    return EvidencePack(
        user_id=user_id,
        user_context=UserContext(**user_ctx),
        recommendations=recommendations,
    )


def build_evidence_pack(
    rec: RecsysAdapter,
    cat: CatalogSource,
    out_path: str | Path,
    max_users: int | None = None,
) -> int:
    """어댑터·카탈로그로 사용자별 EvidencePack을 JSONL로 저장. 작성한 행 수 반환."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    user_ids = rec.list_user_ids()
    if max_users:
        user_ids = user_ids[:max_users]

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for uid in user_ids:
            pack = _build_one(uid, rec, cat)
            if pack is None:
                continue
            f.write(pack.model_dump_json() + "\n")
            n_written += 1
    return n_written


def iter_evidence_pack(path: str | Path) -> Iterator[EvidencePack]:
    """저장된 JSONL을 EvidencePack 객체로 스트림 로드."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield EvidencePack.model_validate_json(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidence Pack 빌더")
    parser.add_argument("--adapter", choices=["mock", "csv"], default="mock",
                        help="추천 결과 어댑터 (기본: mock)")
    parser.add_argument("--submission-csv", default=None,
                        help="--adapter csv 일 때 필수. 본체가 만든 submission CSV 경로.")
    parser.add_argument("--catalog", choices=["mock", "parquet"], default=None,
                        help="카탈로그 소스. 기본: --adapter에 맞춰 자동 (mock→mock, csv→parquet 필요).")
    parser.add_argument("--train-parquet", default=None,
                        help="--catalog parquet 일 때 필수. 본체 학습용 train.parquet 경로.")
    parser.add_argument("--catalog-cache", default=None,
                        help="전체 parquet 집계 캐시 경로. --max-users 없는 전체 빌드에서만 저장/재사용.")
    parser.add_argument("--out", default="data/evidence_pack.jsonl",
                        help="JSONL 출력 경로 (기본: data/evidence_pack.jsonl)")
    parser.add_argument("--max-users", type=int, default=None,
                        help="처리할 최대 사용자 수 (데모용)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mock-n-users", type=int, default=50, help="(mock) 더미 사용자 수")
    parser.add_argument("--mock-n-items", type=int, default=200, help="(mock) 더미 아이템 수")
    parser.add_argument("--seed", type=int, default=42, help="(mock) seed")
    args = parser.parse_args(argv)

    # 추천 어댑터
    if args.adapter == "mock":
        mock = MockAdapter(
            n_users=args.mock_n_users,
            n_items=args.mock_n_items,
            top_k=args.top_k,
            seed=args.seed,
        )
        rec: RecsysAdapter = mock
        default_cat = "mock"
    else:
        if not args.submission_csv:
            print("ERROR: --adapter csv 일 때 --submission-csv 필요", file=sys.stderr)
            return 2
        rec = CSVAdapter(args.submission_csv, top_k=args.top_k, max_users=args.max_users)
        default_cat = "parquet"
        mock = None

    # 카탈로그 소스
    cat_choice = args.catalog or default_cat
    if cat_choice == "mock":
        if mock is None:
            print("ERROR: --catalog mock 은 --adapter mock 일 때만 가능", file=sys.stderr)
            return 2
        cat_src: CatalogSource = mock
    else:
        if not args.train_parquet:
            print("ERROR: --catalog parquet 일 때 --train-parquet 필요", file=sys.stderr)
            return 2
        selected_user_ids = rec.list_user_ids()
        if args.max_users:
            selected_user_ids = selected_user_ids[:args.max_users]
        selected_item_ids = {iid for uid in selected_user_ids for iid in rec.get_top_k(uid)}
        partial = args.max_users is not None
        cat_src = ParquetCatalogSource(
            args.train_parquet,
            user_ids=selected_user_ids if partial else None,
            item_ids=selected_item_ids if partial else None,
            cache_path=args.catalog_cache,
        )

    n = build_evidence_pack(rec, cat_src, args.out, max_users=args.max_users)
    print(f"Wrote {n} evidence packs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
