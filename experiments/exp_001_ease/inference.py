"""exp_001_ease / inference.py — score all users in batches, generate submission.

Same artifact structure as exp_000 for ensemble compatibility:
    predictions.parquet  -- top-N candidates per known user (user_id, item_id, score, rank)
    output.csv           -- competition submission for all 638,257 users (popularity fallback)

EASE inference is a single sparse-dense matmul per user batch:
    scores = X_batch @ B  -> shape (batch_size, n_items)
    top-N per user via argpartition + sort.

Usage:
    python inference.py
    python inference.py --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, load_npz

from core.data_loader import load_train_data, load_mappings  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)


def recommend_batch(
    X_batch: csr_matrix,
    B: np.ndarray,
    top_n: int,
    filter_already_liked: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a user batch and return per-user top-N item indices + scores.

    Args:
        X_batch: sparse (batch_size, n_items) user interactions.
        B: dense (n_items, n_items) EASE matrix.
        top_n: number of items per user.
        filter_already_liked: if True, set scores of items in user history to -inf.

    Returns:
        ids: (batch_size, top_n) item indices, sorted by score desc.
        scores: (batch_size, top_n) corresponding scores.
    """
    # sparse-dense matmul: efficient even though B is huge dense
    scores = X_batch @ B  # (batch_size, n_items), dense float32

    if filter_already_liked:
        rows, cols = X_batch.nonzero()
        scores[rows, cols] = -np.inf

    # Top-N: argpartition (O(n)) then sort the top-N
    top_idx = np.argpartition(-scores, top_n, axis=1)[:, :top_n]
    rows = np.arange(scores.shape[0])[:, None]
    top_scores = scores[rows, top_idx]
    sort_pos = np.argsort(-top_scores, axis=1)
    top_idx = top_idx[rows, sort_pos]
    top_scores = top_scores[rows, sort_pos]
    return top_idx.astype(np.int64), top_scores.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--saved-dir", default=str(Path(__file__).parent / "saved"))
    parser.add_argument("--out-dir", default=str(Path(__file__).parent))
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    saved_dir = Path(args.saved_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load model + artifacts ----------------------------------------
    logger.info("loading EASE B matrix from %s", saved_dir / "B.npy")
    B = np.load(saved_dir / "B.npy")
    interactions = load_npz(str(saved_dir / "interactions.npz"))
    mappings = load_mappings(str(saved_dir / "mappings"))
    val_gt_df = pd.read_parquet(saved_dir / "val_gt.parquet")
    with open(saved_dir / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    logger.info(
        "loaded: B shape=%s dtype=%s (%.1fGB), interactions shape=%s, %s users, "
        "val_gt %s rows, %s eval_users",
        B.shape, B.dtype, B.nbytes / 1024**3,
        interactions.shape,
        f"{len(mappings['user2idx']):,}",
        f"{len(val_gt_df):,}",
        f"{len(eval_users):,}",
    )

    # ---- 2. Popularity fallback (re-load + re-split, no train_df.parquet)
    df_full = load_train_data(cfg["train_data"])
    train_df, _ = time_based_split(
        df_full, val_days=cfg["val_days"], gt_event_types=cfg["gt_event_types"]
    )
    popularity = compute_popularity(train_df, top_n=cfg["top_n"])

    # ---- 3. All-users list -----------------------------------------------
    sample = pd.read_csv(cfg["sample_submission"])
    all_users = sample["user_id"].drop_duplicates().tolist()
    logger.info("all_users from sample_submission: %s", f"{len(all_users):,}")

    # ---- 4. Known users + their idx --------------------------------------
    known_users = [u for u in all_users if u in mappings["user2idx"]]
    cold_start = len(all_users) - len(known_users)
    known_user_idxs = np.fromiter(
        (mappings["user2idx"][u] for u in known_users),
        dtype=np.int64,
        count=len(known_users),
    )
    logger.info(
        "predicting: known=%s, cold-start (popularity-only)=%s",
        f"{len(known_users):,}",
        f"{cold_start:,}",
    )

    # ---- 5. Batch inference ----------------------------------------------
    top_n = cfg["top_n"]
    batch_size = cfg["inference_batch_size"]
    n_known = len(known_users)

    all_ids = np.zeros((n_known, top_n), dtype=np.int64)
    all_scores = np.zeros((n_known, top_n), dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_known, batch_size):
        end = min(start + batch_size, n_known)
        batch_idxs = known_user_idxs[start:end]
        X_batch = interactions[batch_idxs]

        ids, scores = recommend_batch(
            X_batch, B, top_n=top_n, filter_already_liked=cfg["filter_already_liked"]
        )
        all_ids[start:end] = ids
        all_scores[start:end] = scores

        # progress every ~10 batches
        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            logger.info("batch %s/%s, %.1fs elapsed", f"{end:,}", f"{n_known:,}", elapsed)

    logger.info("inference done in %.1fs", time.time() - t0)

    # ---- 6. Build predictions.parquet ------------------------------------
    user_repeat = np.repeat(np.asarray(known_users, dtype=object), top_n)
    item_idx_flat = all_ids.reshape(-1)
    score_flat = all_scores.reshape(-1)
    rank_flat = np.tile(np.arange(1, top_n + 1, dtype=np.int32), n_known)

    idx2item_arr = np.empty(len(mappings["idx2item"]), dtype=object)
    for j, it in mappings["idx2item"].items():
        idx2item_arr[j] = it

    pred_df = pd.DataFrame(
        {
            "user_id": user_repeat,
            "item_id": idx2item_arr[item_idx_flat],
            "score": score_flat.astype(np.float64, copy=False),
            "rank": rank_flat,
        }
    )
    pred_path = out_dir / "predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info("wrote %s (%s rows)", pred_path, f"{len(pred_df):,}")

    # ---- 7. Self-val NDCG@10 + recall@10 ---------------------------------
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    ndcg10 = ndcg_at_k_from_df(pred_df, val_gt_eval, k=10)
    recall10 = recall_at_k_from_df(pred_df, val_gt_eval, k=10)
    logger.info(
        "self-val (last %d days, gt=%s, eval_users=%s):",
        cfg["val_days"], cfg["gt_event_types"], f"{len(eval_users):,}",
    )
    logger.info("  NDCG@10   = %.6f", ndcg10)
    logger.info("  recall@10 = %.6f", recall10)

    # ---- 8. Submission CSV -----------------------------------------------
    output_csv = out_dir / "output.csv"
    predictions_to_submission(
        pred_path=str(pred_path),
        output_csv=str(output_csv),
        all_users=all_users,
        mappings=mappings,
        popularity_fallback=popularity,
        items_per_user=cfg["items_per_user"],
    )
    ok = validate_submission(
        str(output_csv),
        expected_users=len(all_users),
        items_per_user=cfg["items_per_user"],
    )
    if not ok:
        raise RuntimeError("validate_submission FAILED -- do not upload")
    logger.info("validate_submission OK")

    # ---- 9. wandb log ----------------------------------------------------
    if args.use_wandb:
        try:
            import wandb

            init_kwargs = dict(
                entity=cfg.get("wandb_entity"),
                project=cfg["wandb_project"],
                name=cfg["run_name"],
                config=cfg,
            )
            run_id_file = saved_dir / "wandb_run_id.txt"
            if run_id_file.exists():
                init_kwargs["id"] = run_id_file.read_text().strip()
                init_kwargs["resume"] = "must"
            else:
                logger.warning(
                    "no wandb_run_id.txt under %s -- inference will spawn a new run",
                    saved_dir,
                )
                init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**init_kwargs)
            wandb.log(
                {
                    "val_ndcg@10": ndcg10,
                    "val_recall@10": recall10,
                    "n_eval_users": len(eval_users),
                    "n_known_users": len(known_users),
                    "n_cold_start": cold_start,
                    "popularity_fallback_size": len(popularity),
                    "B_size_GB": B.nbytes / 1024**3,
                }
            )
            pred_artifact = wandb.Artifact(f'{cfg["run_name"]}_predictions', type="prediction")
            pred_artifact.add_file(str(pred_path))
            pred_artifact.add_file(str(output_csv))
            wandb.log_artifact(pred_artifact)
            wandb_run.finish()
        except ImportError:
            logger.warning("wandb not installed; skipping log")

    logger.info("inference.py done")


if __name__ == "__main__":
    main()
