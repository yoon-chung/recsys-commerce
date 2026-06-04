"""build_catalog.py — recommenders.py가 필요한 catalog/recency/recs 산출물 빌드.

사용법:
    python -m pipeline.build_catalog \\
        --train data/train.parquet \\
        --submission ../../outputs_best_owy/submission_reranker_lgbm.csv \\
        --aliases data/id_aliases.json \\
        --out-dir data/

산출:
    item_catalog.json         — {item_alias: {category_code, category_l2, brand, price, ...}}
    recency_pool.json         — 최근 14일 등장 item_alias 리스트 (인기도 정렬)
    user_recommendations.json — {user_alias: [item_alias, ...]} (submission top-10)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_item_catalog(train_parquet: str | Path, items_map: dict) -> dict:
    import pandas as pd

    from .brand_remap import remap_brand_series

    cols = ["item_id", "category_code", "brand", "price", "event_time", "event_type"]
    df = pd.read_parquet(train_parquet, columns=cols)
    df["item_id"] = df["item_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["brand"] = remap_brand_series(df["brand"])

    max_time = df["event_time"].max()
    recent_cutoff = max_time - pd.Timedelta(days=14)

    grouped = (
        df.dropna(subset=["item_id"])
        .groupby("item_id")
        .agg(
            category_code=("category_code", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            brand=("brand", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            price=("price", "median"),
            total_events=("event_time", "count"),
            last_seen=("event_time", "max"),
            first_seen=("event_time", "min"),
        )
    )

    recent_counts = (
        df[df["event_time"] >= recent_cutoff].groupby("item_id").size().to_dict()
    )

    catalog: dict[str, dict] = {}
    for item_id, row in grouped.iterrows():
        alias = items_map.get(item_id)
        if not alias:
            continue
        cat = row["category_code"]
        category_l2 = ".".join(str(cat).split(".")[:2]) if cat else None
        catalog[alias] = {
            "item_id": item_id,
            "category_code": cat,
            "category_l2": category_l2,
            "brand": row["brand"],
            "price": float(row["price"]) if row["price"] == row["price"] else None,
            "total_events": int(row["total_events"]),
            "recent14_events": int(recent_counts.get(item_id, 0)),
            "last_seen": row["last_seen"].isoformat() if row["last_seen"] is not None else None,
            "first_seen": row["first_seen"].isoformat() if row["first_seen"] is not None else None,
        }
    return catalog


def build_recency_pool(catalog: dict, max_pool: int = 500) -> list[str]:
    """최근 14일 등장 횟수 기준 상위 max_pool개 alias."""
    scored = [
        (alias, meta["recent14_events"])
        for alias, meta in catalog.items()
        if meta["recent14_events"] > 0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [alias for alias, _ in scored[:max_pool]]


def build_user_recommendations(
    submission_csv: str | Path,
    users_map: dict,
    items_map: dict,
    top_k: int = 10,
) -> dict[str, list[str]]:
    import pandas as pd

    sub = pd.read_csv(submission_csv, dtype={"user_id": str, "item_id": str})
    recs: dict[str, list[str]] = {}
    for user_id, group in sub.groupby("user_id", sort=False):
        user_alias = users_map.get(user_id)
        if not user_alias:
            continue
        aliases = [items_map.get(iid) for iid in group["item_id"].tolist()[:top_k]]
        aliases = [a for a in aliases if a is not None]
        recs[user_alias] = aliases
    return recs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="item_catalog + recency_pool + user_recs 빌드")
    parser.add_argument("--train", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--aliases", default="data/id_aliases.json")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--recency-pool-size", type=int, default=500)
    args = parser.parse_args(argv)

    with open(args.aliases, encoding="utf-8") as f:
        aliases = json.load(f)

    users_map = aliases.get("users", {})
    items_map = aliases.get("items", {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1/3 item_catalog 빌드 중...")
    catalog = build_item_catalog(args.train, items_map)
    with (out_dir / "item_catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False)
    print(f"  → {len(catalog):,} items in item_catalog.json")

    print("2/3 recency_pool 빌드 중...")
    pool = build_recency_pool(catalog, max_pool=args.recency_pool_size)
    with (out_dir / "recency_pool.json").open("w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False)
    print(f"  → {len(pool)} items in recency_pool.json")

    print("3/3 user_recommendations 빌드 중...")
    recs = build_user_recommendations(args.submission, users_map, items_map, top_k=args.top_k)
    with (out_dir / "user_recommendations.json").open("w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    print(f"  → {len(recs):,} users in user_recommendations.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())