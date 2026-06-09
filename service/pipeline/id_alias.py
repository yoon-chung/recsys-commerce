"""id_alias.py — user/item UUID ↔ alias 양방향 매핑 테이블 생성.

사용법:
    python -m pipeline.id_alias \\
        --train ../baseline/data/train.parquet \\
        --submission data/raw/submission_reranker_lgbm.csv \\
        --out data/id_aliases.json

출력 포맷 (data/id_aliases.json):
{
  "users":     {"<uuid>": "user_00001", ...},
  "users_inv": {"user_00001": "<uuid>", ...},
  "items":     {"<uuid>": "shoes.keds.kapika.mid_0001", ...},
  "items_inv": {"shoes.keds.kapika.mid_0001": "<uuid>", ...}
}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _brand_slug(brand: str | None) -> str:
    if not brand:
        return "unknown"
    slug = str(brand).lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "unknown"


def _price_bucket(price: float | None) -> str:
    if price is None or price != price:
        return "mid"
    if float(price) < 30:
        return "low"
    if float(price) <= 80:
        return "mid"
    return "high"


def _category_parts(category_code: str | None) -> tuple[str, str | None]:
    if not category_code:
        return "unknown", None
    parts = str(category_code).split(".")
    l2 = parts[1] if len(parts) >= 2 else parts[0]
    l3 = parts[2] if len(parts) >= 3 else None
    return l2, l3


def build_aliases(train_parquet: str | Path, submission_csv: str | Path) -> dict:
    import pandas as pd

    from .brand_remap import remap_brand_series

    # 1. train.parquet에서 item 메타 정보 수집
    train_cols = ["user_id", "item_id", "category_code", "brand", "price"]
    df = pd.read_parquet(train_parquet, columns=train_cols)
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["brand"] = remap_brand_series(df["brand"])

    # 2. submission CSV에서 user/item 추가 수집 (train에 없는 경우 대비)
    sub = pd.read_csv(submission_csv, dtype={"user_id": str, "item_id": str})

    all_user_ids = sorted(set(df["user_id"]) | set(sub["user_id"]))
    all_item_ids = set(df["item_id"]) | set(sub["item_id"])

    # 3. user alias 생성
    users: dict[str, str] = {}
    users_inv: dict[str, str] = {}
    for idx, uid in enumerate(all_user_ids, 1):
        alias = f"user_{idx:05d}"
        users[uid] = alias
        users_inv[alias] = uid

    # 4. item 메타 대표값 추출 (같은 item_id에 여러 행이 있으면 mode/mean)
    item_meta = (
        df.dropna(subset=["item_id"])
        .groupby("item_id")
        .agg(
            category_code=("category_code", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            brand=("brand", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            price=("price", "mean"),
        )
        .to_dict("index")
    )

    # 5. item Semantic ID 생성
    items: dict[str, str] = {}
    items_inv: dict[str, str] = {}
    bucket_counter: dict[str, int] = defaultdict(int)

    for iid in sorted(all_item_ids):
        meta = item_meta.get(iid, {})
        l2, l3 = _category_parts(meta.get("category_code"))
        bslug = _brand_slug(meta.get("brand"))
        pbucket = _price_bucket(meta.get("price"))
        key = f"{l2}.{l3}.{bslug}.{pbucket}" if l3 else f"{l2}.{bslug}.{pbucket}"
        bucket_counter[key] += 1
        alias = f"{key}_{bucket_counter[key]:04d}"
        items[iid] = alias
        items_inv[alias] = iid

    return {"users": users, "users_inv": users_inv, "items": items, "items_inv": items_inv}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="user/item alias 테이블 생성")
    parser.add_argument("--train", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", default="data/id_aliases.json")
    args = parser.parse_args(argv)

    aliases = build_aliases(args.train, args.submission)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False, indent=2)

    print(f"users: {len(aliases['users']):,}  items: {len(aliases['items']):,}  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
