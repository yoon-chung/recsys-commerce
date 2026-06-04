"""build_user_vectors.py — 유저 벡터 + FAISS 이웃 인덱스 빌드 (Collaborative 섹션용).

사용법:
    python -m pipeline.build_user_vectors \\
        --profiles data/user_profiles.db \\
        --catalog data/item_catalog.json \\
        --out-dir data \\
        --top-brands 20

산출:
    user_vectors.npy       (N, D) float32 — L2 정규화된 유저 벡터
    user_neighbors.npy     (N, K) int32   — Top-K 유사 유저 row index
    user_alias_to_row.pkl  dict[user_alias → row idx]
    row_to_user_alias.pkl  dict[row idx → user_alias]

벡터 구성:
    [category_l2 가중치, brand 가중치]
    이벤트 가중치: view 1.0 / cart 25.0 / purchase 50.0
    L2 정규화 → 코사인 유사도 = 내적
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path


EVENT_WEIGHT = {"view": 1.0, "cart": 25.0, "purchase": 50.0}
TOP_K = 20


def _load_profiles_with_log(db_path: str | Path) -> list[dict]:
    """profile DB에서 user_alias + event_log만 로드."""
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT user_alias, event_log FROM user_profiles").fetchall()
    con.close()
    return [
        {"user_alias": ua, "event_log": json.loads(elog or "[]")}
        for ua, elog in rows
    ]


def _build_vocabularies(catalog: dict, top_brands: int) -> tuple[list[str], list[str]]:
    """카테고리 L2 + 인기 brand vocabulary 구성."""
    cat_counts: dict[str, int] = defaultdict(int)
    brand_counts: dict[str, int] = defaultdict(int)
    for meta in catalog.values():
        c = meta.get("category_l2")
        b = meta.get("brand")
        ev = meta.get("total_events", 0) or 0
        if c:
            cat_counts[c] += ev
        if b:
            brand_counts[b] += ev
    cats = sorted(cat_counts, key=cat_counts.get, reverse=True)
    brands = sorted(brand_counts, key=brand_counts.get, reverse=True)[:top_brands]
    return cats, brands


def _build_vectors(
    profiles: list[dict],
    catalog: dict,
    cat_vocab: list[str],
    brand_vocab: list[str],
):
    """유저 별 가중 multi-hot 벡터 → L2 정규화 → (N, D) float32."""
    import numpy as np

    cat_idx = {c: i for i, c in enumerate(cat_vocab)}
    brand_idx = {b: i for i, b in enumerate(brand_vocab)}
    D = len(cat_vocab) + len(brand_vocab)
    N = len(profiles)
    vectors = np.zeros((N, D), dtype=np.float32)

    for r, prof in enumerate(profiles):
        for ev in prof["event_log"]:
            item_alias = ev.get("item_id")
            meta = catalog.get(item_alias)
            if not meta:
                continue
            w = EVENT_WEIGHT.get(ev.get("event_type"), 0.0)
            if w == 0:
                continue
            c = meta.get("category_l2")
            if c in cat_idx:
                vectors[r, cat_idx[c]] += w
            b = meta.get("brand")
            if b in brand_idx:
                vectors[r, len(cat_vocab) + brand_idx[b]] += w

    # L2 정규화 (norm 0 유저는 zero vector 유지 → FAISS 내적 0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
    return vectors


def _build_faiss(vectors, k: int = TOP_K, use_gpu: bool = False):
    """FAISS IndexFlatIP + Top-(k+1) 검색 → 자기 자신 제외 Top-k.

    use_gpu=True: faiss-gpu가 설치된 환경에서 GPU 가속 (638K 풀 빌드 권장).
    """
    import faiss
    import numpy as np

    n, d = vectors.shape
    index = faiss.IndexFlatIP(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(vectors)

    # 자기 자신이 항상 첫 번째 → k+1 검색 후 첫 컬럼 제거
    _, I = index.search(vectors, k + 1)
    neighbors = I[:, 1:k + 1].astype(np.int32)  # (N, K)
    return neighbors


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description="유저 벡터 + FAISS 이웃 인덱스 빌드")
    parser.add_argument("--profiles", default="data/user_profiles.db")
    parser.add_argument("--catalog", default="data/item_catalog.json")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--top-brands", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--use-gpu", action="store_true", help="faiss-gpu (638K full build 권장)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1/4 profile + catalog 로드 중...")
    profiles = _load_profiles_with_log(args.profiles)
    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  profiles: {len(profiles):,}  catalog: {len(catalog):,}")

    print("2/4 vocabulary 구성 중...")
    cat_vocab, brand_vocab = _build_vocabularies(catalog, args.top_brands)
    print(f"  cat L2: {len(cat_vocab)}  brand: {len(brand_vocab)}")

    print("3/4 유저 벡터 빌드 중...")
    vectors = _build_vectors(profiles, catalog, cat_vocab, brand_vocab)
    print(f"  vectors: {vectors.shape}")

    print(f"4/4 FAISS 인덱스 + Top-K 이웃 검색 중... (gpu={args.use_gpu})")
    neighbors = _build_faiss(vectors, k=args.top_k, use_gpu=args.use_gpu)
    print(f"  neighbors: {neighbors.shape}")

    # 저장
    np.save(out_dir / "user_vectors.npy", vectors)
    np.save(out_dir / "user_neighbors.npy", neighbors)

    alias_to_row = {p["user_alias"]: i for i, p in enumerate(profiles)}
    row_to_alias = {i: p["user_alias"] for i, p in enumerate(profiles)}
    with (out_dir / "user_alias_to_row.pkl").open("wb") as f:
        pickle.dump(alias_to_row, f)
    with (out_dir / "row_to_user_alias.pkl").open("wb") as f:
        pickle.dump(row_to_alias, f)

    print(f"\nDone → {out_dir}/")
    print(f"  user_vectors.npy      ({vectors.shape})")
    print(f"  user_neighbors.npy    ({neighbors.shape})")
    print(f"  user_alias_to_row.pkl ({len(alias_to_row):,} rows)")
    print(f"  row_to_user_alias.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())