"""exp_007_tifu_knn / train.py -- build TIFU-KNN state matrices.

TIFU-KNN 의 "training" 은 user × item 가중 frequency matrix 빌드 만. iterative
learning 없음. saved/ 에 sparse matrix 2개 (.npz) + 메타 저장:
    user_vec_raw.npz    : own_score 용 (정규화 전)
    user_vec_norm.npz   : KNN cosine 용 (row L2-normalized)
    meta.json           : n_users, n_items, config 요약

inference.py 가 이걸 로드해 KNN + score 계산.

Usage:
    python train.py
    python train.py --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.sparse import save_npz  # noqa: E402

from core.data_loader import load_train_data, load_mappings  # noqa: E402
from core.validation import time_based_split  # noqa: E402

from tifu_knn import TIFUKNN  # noqa: E402

logger = logging.getLogger(__name__)


def build_user_seqs(
    train_df: pd.DataFrame,
    user2idx: dict,
    item2idx: dict,
    event_weights: dict,
) -> dict:
    """Per-user chronological list of (item_idx, event_weight)."""
    df = train_df.copy()
    df["w"] = df["event_type"].map(event_weights).fillna(0.0).astype("float32")
    df = df[df["w"] > 0]
    df["u_idx"] = df["user_id"].map(user2idx)
    df["i_idx"] = df["item_id"].map(item2idx)
    df = df.dropna(subset=["u_idx", "i_idx"])
    df["u_idx"] = df["u_idx"].astype("int64")
    df["i_idx"] = df["i_idx"].astype("int64")

    # event_time 은 string -> datetime, then groupby/sort
    df["ts"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
    df = df.sort_values(["u_idx", "ts"], kind="mergesort")

    user_seqs: dict[int, list] = {}
    for u_idx, sub in df.groupby("u_idx", sort=False):
        user_seqs[int(u_idx)] = list(zip(sub["i_idx"].tolist(), sub["w"].tolist()))
    return user_seqs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--saved-dir", default=str(HERE / "saved"))
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    saved_dir = Path(args.saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Data --------------------------------------------------------
    df_full = load_train_data(cfg["train_data"])
    train_df, _ = time_based_split(
        df_full,
        val_days=cfg["val_days"],
        gt_event_types=cfg["gt_event_types"],
    )
    logger.info("train_df after time_based_split: %s rows", f"{len(train_df):,}")

    # Optional recency window (default null = 4m full; TIFU decay 가 이미 처리)
    if cfg.get("last_days") is not None:
        ts = pd.to_datetime(train_df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
        cutoff = ts.max() - pd.Timedelta(days=int(cfg["last_days"]))
        before = len(train_df)
        train_df = train_df[ts >= cutoff]
        logger.info("last_days=%d filter: %s -> %s rows",
                    cfg["last_days"], f"{before:,}", f"{len(train_df):,}")

    # ---- 2. Mappings -- reuse exp_001 EASE for cross-model 일관성 -------
    ease_saved = (HERE / cfg["ease_saved"]).resolve()
    mappings = load_mappings(str(ease_saved / "mappings"))
    user2idx = mappings["user2idx"]
    item2idx = mappings["item2idx"]
    n_users = len(user2idx)
    n_items = len(item2idx)

    # ---- 3. user_seqs --------------------------------------------------
    t0 = time.time()
    user_seqs = build_user_seqs(train_df, user2idx, item2idx, cfg["event_weights"])
    logger.info("user_seqs built for %s users in %.1fs",
                f"{len(user_seqs):,}", time.time() - t0)

    # ---- 4. Fit TIFUKNN ------------------------------------------------
    tifu = TIFUKNN(
        group_count=cfg["group_count"],
        decay_within=cfg["decay_within"],
        decay_across=cfg["decay_across"],
        knn_k=cfg["knn_k"],
        alpha=cfg["alpha"],
    )
    t0 = time.time()
    tifu.fit(user_seqs, n_users=n_users, n_items=n_items)
    logger.info("TIFUKNN.fit done in %.1fs", time.time() - t0)

    # ---- 5. Save state -------------------------------------------------
    save_npz(saved_dir / "user_vec_raw.npz", tifu.user_vec_raw)
    save_npz(saved_dir / "user_vec_norm.npz", tifu.user_vec_norm)
    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "nnz": int(tifu.user_vec_raw.nnz),
        "group_count": tifu.group_count,
        "decay_within": tifu.decay_within,
        "decay_across": tifu.decay_across,
        "knn_k": tifu.knn_k,
        "alpha": tifu.alpha,
        "event_weights": cfg["event_weights"],
        "last_days": cfg.get("last_days"),
    }
    with open(saved_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("saved state to %s", saved_dir)

    # ---- 6. wandb ------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb

            run = wandb.init(
                entity=cfg.get("wandb_entity"),
                project=cfg["wandb_project"],
                name=cfg["run_name"],
                config=meta,
            )
            wandb.log({
                "n_users": n_users,
                "n_items": n_items,
                "user_vec_nnz": int(tifu.user_vec_raw.nnz),
                "avg_items_per_active_user": float(
                    tifu.user_vec_raw.nnz / max(1, (np.asarray(tifu.user_vec_raw.sum(axis=1)).flatten() > 0).sum())
                ),
            })
            (saved_dir / "wandb_run_id.txt").write_text(run.id)
            run.finish()
        except ImportError:
            logger.warning("wandb not installed -- skipping log")

    logger.info("train.py done")


if __name__ == "__main__":
    main()
