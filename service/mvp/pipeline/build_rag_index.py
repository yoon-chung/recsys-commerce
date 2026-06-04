"""build_rag_index.py — alias + submission → Evidence Pack 재빌드 + (선택) FAISS.

사용법 (side_project/ 루트에서 실행):
    python -m pipeline.build_rag_index \\
        --aliases data/id_aliases.json \\
        --submission ../outputs/submission_reranker_lgbm.csv \\
        --train ../data/train.parquet \\
        --out data/evidence_pack.jsonl

    # FAISS 유사 유저 인덱스까지
    python -m pipeline.build_rag_index ... --faiss data/user_neighbors.faiss

출력:
    data/evidence_pack.jsonl     — side_project/ui/app.py가 읽는 파일 (덮어씀)
    data/user_neighbors.faiss    — (선택) CF 유사 유저 인덱스
    data/user_neighbors_meta.pkl — (선택) index 번호 → user_alias 매핑
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
    from evidence_pack.schema import EvidencePack

    users_map = aliases.get("users", {})
    items_map = aliases.get("items", {})

    # 1. 원본 UUID 기준으로 Evidence Pack 빌드
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

    # 2. alias 주입 후 최종 저장
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


def build_faiss_index(
    train_parquet: str | Path,
    aliases: dict,
    faiss_out: str | Path,
    meta_out: str | Path,
) -> None:
    """유저-유저 purchase/cart 코사인 유사도 FAISS 인덱스 생성."""
    try:
        import faiss
        import numpy as np
        import pandas as pd
        import pickle
        from scipy.sparse import csr_matrix
    except ImportError:
        raise ImportError("pip install faiss-cpu scipy 필요")

    users_map = aliases.get("users", {})

    df = pd.read_parquet(train_parquet, columns=["user_id", "item_id", "event_type"])
    df = df[df["event_type"].isin(["purchase", "cart"])].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["weight"] = df["event_type"].map({"purchase": 2.0, "cart": 1.0})

    user_list = sorted(df["user_id"].unique())
    item_list = sorted(df["item_id"].unique())
    user_idx = {u: i for i, u in enumerate(user_list)}
    item_idx = {it: i for i, it in enumerate(item_list)}

    rows = df["user_id"].map(user_idx).values
    cols = df["item_id"].map(item_idx).values
    data = df["weight"].values
    mat = csr_matrix((data, (rows, cols)), shape=(len(user_list), len(item_list)))

    # L2 정규화
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    mat_norm = mat.multiply(1.0 / norms[:, None]).toarray().astype("float32")

    index = faiss.IndexFlatIP(mat_norm.shape[1])
    index.add(mat_norm)
    faiss.write_index(index, str(faiss_out))

    meta = {i: users_map.get(uid, uid) for i, uid in enumerate(user_list)}
    with open(meta_out, "wb") as f:
        pickle.dump(meta, f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidence Pack + (선택) FAISS 빌드")
    parser.add_argument("--aliases", default="data/id_aliases.json")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", default="data/evidence_pack.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--faiss", default=None)
    args = parser.parse_args(argv)

    with open(args.aliases, encoding="utf-8") as f:
        aliases = json.load(f)

    print("Evidence Pack 빌드 중...")
    n = build_evidence_pack_with_alias(
        aliases, args.submission, args.train, args.out,
        top_k=args.top_k, max_users=args.max_users,
    )
    print(f"Wrote {n:,} evidence packs → {args.out}")

    if args.faiss:
        print("FAISS 인덱스 빌드 중...")
        meta_out = str(args.faiss).replace(".faiss", "_meta.pkl")
        build_faiss_index(args.train, aliases, args.faiss, meta_out)
        print(f"FAISS index → {args.faiss}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
