"""build_rag_index.py — submission + train + alias → Evidence Pack JSONL 빌드.

사용법:
    python -m pipeline.build_rag_index \\
        --aliases data/id_aliases.json \\
        --submission data/raw/submission_reranker_lgbm.csv \\
        --train ../baseline/data/train.parquet \\
        --out data/evidence_pack.jsonl

출력:
    data/evidence_pack.jsonl  —  ui/app.py + LangGraph pack_loader가 읽는 파일
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_evidence_pack_with_alias(
    aliases: dict,
    submission_csv: str | Path,
    train_parquet: str | Path,
    out_path: str | Path,
    top_k: int = 10,
    max_users: int | None = None,
) -> int:
    from evidence_pack.adapter import CSVAdapter, ParquetCatalogSource
    from evidence_pack.builder import build_evidence_pack, iter_evidence_pack

    users_map = aliases.get("users", {})
    items_map = aliases.get("items", {})

    rec = CSVAdapter(submission_csv, top_k=top_k, max_users=max_users)
    selected_user_ids = rec.list_user_ids()
    if max_users:
        selected_user_ids = selected_user_ids[:max_users]

    partial = max_users is not None
    cat = ParquetCatalogSource(
        train_parquet,
        user_ids=selected_user_ids if partial else None,
        item_ids=None,
    )

    tmp_path = Path(out_path).with_suffix(".tmp.jsonl")
    build_evidence_pack(rec, cat, tmp_path, max_users=max_users)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for pack in iter_evidence_pack(tmp_path):
            pack.user_alias = users_map.get(pack.user_id)
            for rec_item in pack.recommendations:
                rec_item.item_alias = items_map.get(rec_item.item_id)
            f_out.write(pack.model_dump_json() + "\n")
            n += 1

    tmp_path.unlink(missing_ok=True)
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidence Pack JSONL 빌드 (alias 주입 포함)")
    parser.add_argument("--aliases", default="data/id_aliases.json")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", default="data/evidence_pack.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args(argv)

    with open(args.aliases, encoding="utf-8") as f:
        aliases = json.load(f)

    n = build_evidence_pack_with_alias(
        aliases, args.submission, args.train, args.out,
        top_k=args.top_k, max_users=args.max_users,
    )
    print(f"Wrote {n:,} evidence packs → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())