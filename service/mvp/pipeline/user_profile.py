"""user_profile.py — train.parquet에서 유저 프로필을 집계해 SQLite에 저장.

사용법:
    python -m src.mvp.user_profile \\
        --train data/train.parquet \\
        --aliases data/id_aliases.json \\
        --out data/user_profiles.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_alias       TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    total_events     INTEGER DEFAULT 0,
    recent14_events  INTEGER DEFAULT 0,
    top_categories   TEXT DEFAULT '[]',
    top_brands       TEXT DEFAULT '[]',
    carted_brands    TEXT DEFAULT '[]',
    purchased_brands TEXT DEFAULT '[]',
    price_median     REAL,
    price_p25        REAL,
    price_p75        REAL,
    price_iqr        REAL,
    seen_items       TEXT DEFAULT '[]',
    carted_items     TEXT DEFAULT '[]',
    purchased_items  TEXT DEFAULT '[]',
    event_log        TEXT DEFAULT '[]'
);
"""

INSERT_SQL = """
INSERT OR REPLACE INTO user_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def build_profiles(
    train_parquet: str | Path,
    aliases: dict,
    max_users: int | None = None,
) -> list[dict]:
    import math
    import pandas as pd

    from .brand_remap import remap_brand_series

    cols = ["user_id", "item_id", "event_type", "event_time", "category_code", "brand", "price"]
    df = pd.read_parquet(train_parquet, columns=cols)
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["brand"] = remap_brand_series(df["brand"])

    if max_users is not None:
        users_map = aliases.get("users", {})
        sorted_aliases = sorted(users_map.items(), key=lambda kv: kv[1])  # alias 순 (user_00001~)
        keep_uids = {uid for uid, _ in sorted_aliases[:max_users]}
        df = df[df["user_id"].isin(keep_uids)].copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["category_l2"] = df["category_code"].apply(
        lambda c: ".".join(str(c).split(".")[:2]) if pd.notna(c) else None
    )

    max_time = df["event_time"].max()
    cutoff = max_time - pd.Timedelta(days=14)

    users_map = aliases.get("users", {})
    items_map = aliases.get("items", {})

    profiles: list[dict] = []

    for user_id, group in df.groupby("user_id", sort=False):
        user_alias = users_map.get(user_id)
        if not user_alias:
            continue

        total_events = len(group)
        recent14_events = int((group["event_time"] >= cutoff).sum())

        view_g = group[group["event_type"] == "view"]
        cart_g = group[group["event_type"] == "cart"]
        purch_g = group[group["event_type"] == "purchase"]

        # 카테고리·브랜드 선호 (view 기반)
        top_categories = (
            view_g["category_l2"].dropna().value_counts().head(3).index.tolist()
        )
        top_brands = (
            view_g["brand"].dropna().value_counts().head(3).index.tolist()
        )
        carted_brands = cart_g["brand"].dropna().unique().tolist()
        purchased_brands = purch_g["brand"].dropna().unique().tolist()

        # 가격대 (view 기반)
        prices = view_g["price"].dropna().astype(float)
        if len(prices) >= 2:
            p25 = float(prices.quantile(0.25))
            p50 = float(prices.quantile(0.50))
            p75 = float(prices.quantile(0.75))
            iqr = p75 - p25
        elif len(prices) == 1:
            p25 = p50 = p75 = float(prices.iloc[0])
            iqr = 0.0
        else:
            p25 = p50 = p75 = iqr = None

        # seen_items: view+cart+purchase 통합
        seen_ids = group["item_id"].unique().tolist()
        cart_ids = cart_g["item_id"].unique().tolist()
        purch_ids = purch_g["item_id"].unique().tolist()

        seen_aliases = [items_map.get(i, i) for i in seen_ids]
        cart_aliases = [items_map.get(i, i) for i in cart_ids]
        purch_aliases = [items_map.get(i, i) for i in purch_ids]

        # event_log: revisit recommender 입력용 (item_alias + event_type + days_ago)
        event_log = [
            {
                "item_id": items_map.get(row.item_id, row.item_id),
                "event_type": row.event_type,
                "days_ago": int((max_time - row.event_time).days),
            }
            for row in group.itertuples()
            if pd.notna(row.event_time)
        ]

        profiles.append({
            "user_alias": user_alias,
            "user_id": user_id,
            "total_events": total_events,
            "recent14_events": recent14_events,
            "top_categories": top_categories,
            "top_brands": top_brands,
            "carted_brands": carted_brands,
            "purchased_brands": purchased_brands,
            "price_median": p50,
            "price_p25": p25,
            "price_p75": p75,
            "price_iqr": iqr,
            "seen_items": seen_aliases,
            "carted_items": cart_aliases,
            "purchased_items": purch_aliases,
            "event_log": event_log,
        })

    return profiles


def save_profiles(profiles: list[dict], out_path: str | Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(out_path)
    con.execute(CREATE_TABLE_SQL)

    def _j(v: list | None) -> str:
        return json.dumps(v or [], ensure_ascii=False)

    rows = [
        (
            p["user_alias"], p["user_id"],
            p.get("total_events", 0), p.get("recent14_events", 0),
            _j(p.get("top_categories")), _j(p.get("top_brands")),
            _j(p.get("carted_brands")), _j(p.get("purchased_brands")),
            p.get("price_median"), p.get("price_p25"),
            p.get("price_p75"), p.get("price_iqr"),
            _j(p.get("seen_items")), _j(p.get("carted_items")),
            _j(p.get("purchased_items")),
            _j(p.get("event_log")),
        )
        for p in profiles
    ]
    con.executemany(INSERT_SQL, rows)
    con.commit()
    con.close()
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="유저 프로필 집계 → SQLite 저장")
    parser.add_argument("--train", required=True)
    parser.add_argument("--aliases", default="data/id_aliases.json")
    parser.add_argument("--out", default="data/user_profiles.db")
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args(argv)

    with open(args.aliases, encoding="utf-8") as f:
        aliases = json.load(f)

    print("프로필 집계 중...")
    profiles = build_profiles(args.train, aliases, max_users=args.max_users)
    n = save_profiles(profiles, args.out)
    print(f"Wrote {n:,} user profiles → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
