"""exp_007_tifu_knn / inference.py -- KNN + score for all known users, build
predictions.parquet + output.csv.

Loads saved/user_vec_raw.npz + user_vec_norm.npz + meta.json from train.py.

For each known user (in user_vec): runs TIFU-KNN cosine top-K + α-blended score.
Cold-start (not in train) -> popularity fallback handled by core.submission.

Outputs:
    predictions.parquet   user_id, item_id (UUID), score, rank   for known users
    output.csv            638,257 x 10, popularity fallback for cold-start

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.sparse import load_npz  # noqa: E402

from core.data_loader import load_train_data, load_mappings  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

from tifu_knn import TIFUKNN  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--saved-dir", default=str(HERE / "saved"))
    parser.add_argument("--out-dir", default=str(HERE))
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

    # ---- 1. Load state -------------------------------------------------
    with open(saved_dir / "meta.json") as f:
        meta = json.load(f)
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    logger.info("loaded meta: %s users x %s items, %s nnz",
                f"{n_users:,}", f"{n_items:,}", f"{meta['nnz']:,}")

    tifu = TIFUKNN(
        group_count=meta["group_count"],
        decay_within=meta["decay_within"],
        decay_across=meta["decay_across"],
        knn_k=meta["knn_k"],
        alpha=meta["alpha"],
    )
    tifu.user_vec_raw = load_npz(saved_dir / "user_vec_raw.npz").tocsr()
    tifu.user_vec_norm = load_npz(saved_dir / "user_vec_norm.npz").tocsr()
    tifu.n_users = n_users
    tifu.n_items = n_items

    # ---- 2. Mappings + sample submission --------------------------------
    ease_saved = (HERE / cfg["ease_saved"]).resolve()
    mappings = load_mappings(str(ease_saved / "mappings"))
    user2idx = mappings["user2idx"]
    idx2item = mappings["idx2item"]

    sample = pd.read_csv(cfg["sample_submission"])
    all_users = sample["user_id"].drop_duplicates().tolist()
    logger.info("all_users from sample_submission: %s", f"{len(all_users):,}")

    # active users = nonzero rows in user_vec_raw
    row_nnz = np.asarray(tifu.user_vec_raw.getnnz(axis=1)).flatten()
    active_mask = row_nnz > 0
    n_active = int(active_mask.sum())
    logger.info("active users (nonzero history): %s", f"{n_active:,}")

    # known = users in all_users mapped to active idx
    known_users: list[str] = []
    known_idx: list[int] = []
    for u in all_users:
        if u in user2idx:
            ui = user2idx[u]
            if active_mask[ui]:
                known_users.append(u)
                known_idx.append(ui)
    cold_start = len(all_users) - len(known_users)
    logger.info("predicting: known=%s, cold-start (popularity-only)=%s",
                f"{len(known_users):,}", f"{cold_start:,}")

    # ---- 3. KNN + score ------------------------------------------------
    query_idx = np.asarray(known_idx, dtype=np.int64)
    top_n = int(cfg["top_n"])
    t0 = time.time()
    top_idx, top_scores = tifu.predict_topn(
        query_idx,
        top_n=top_n,
        batch_size=int(cfg["batch_size"]),
    )
    logger.info("predict_topn done in %.1fs", time.time() - t0)

    # ---- 4. Build predictions.parquet ----------------------------------
    n_q = len(known_users)
    user_repeat = np.repeat(np.asarray(known_users, dtype=object), top_n)
    item_uuid_flat = np.array(
        [idx2item[i] for i in top_idx.reshape(-1)], dtype=object
    )
    rank_flat = np.tile(np.arange(1, top_n + 1, dtype=np.int32), n_q)
    pred_df = pd.DataFrame({
        "user_id": user_repeat,
        "item_id": item_uuid_flat,
        "score": top_scores.reshape(-1).astype(np.float64, copy=False),
        "rank": rank_flat,
    })
    pred_path = out_dir / "predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info("wrote %s (%s rows)", pred_path, f"{len(pred_df):,}")

    # ---- 5. Self-val NDCG@10 / recall@10 (reuse exp_001 val_gt) --------
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    ndcg10 = ndcg_at_k_from_df(pred_df, val_gt_eval, k=10)
    recall10 = recall_at_k_from_df(pred_df, val_gt_eval, k=10)
    logger.info("self-val (last %d days, gt=%s, eval_users=%s):",
                cfg["val_days"], cfg["gt_event_types"], f"{len(eval_users):,}")
    logger.info("  NDCG@10   = %.6f", ndcg10)
    logger.info("  recall@10 = %.6f", recall10)

    # ---- 6. Popularity + submission CSV --------------------------------
    df_full = load_train_data(cfg["train_data"])
    train_df, _ = time_based_split(
        df_full,
        val_days=cfg["val_days"],
        gt_event_types=cfg["gt_event_types"],
    )
    popularity = compute_popularity(train_df, top_n=top_n)

    output_csv = out_dir / "output.csv"
    predictions_to_submission(
        pred_path=str(pred_path),
        output_csv=str(output_csv),
        all_users=all_users,
        mappings=mappings,
        popularity_fallback=popularity,
        items_per_user=int(cfg["items_per_user"]),
    )
    ok = validate_submission(
        str(output_csv),
        expected_users=len(all_users),
        items_per_user=int(cfg["items_per_user"]),
    )
    if not ok:
        raise RuntimeError("validate_submission FAILED -- do not upload")
    logger.info("validate_submission OK")

    # ---- 7. wandb log (resume train run) -------------------------------
    if args.use_wandb:
        try:
            import wandb

            init_kwargs = dict(
                entity=cfg.get("wandb_entity"),
                project=cfg["wandb_project"],
                name=cfg["run_name"],
            )
            run_id_file = saved_dir / "wandb_run_id.txt"
            if run_id_file.exists():
                init_kwargs["id"] = run_id_file.read_text().strip()
                init_kwargs["resume"] = "must"
            else:
                init_kwargs["resume"] = "allow"
            run = wandb.init(**init_kwargs)
            wandb.log({
                "val_ndcg@10": ndcg10,
                "val_recall@10": recall10,
                "n_eval_users": len(eval_users),
                "n_known_users": len(known_users),
                "n_cold_start": cold_start,
            })
            pred_artifact = wandb.Artifact(
                f'{cfg["run_name"]}_predictions', type="prediction"
            )
            pred_artifact.add_file(str(pred_path))
            pred_artifact.add_file(str(output_csv))
            wandb.log_artifact(pred_artifact)
            run.finish()
        except ImportError:
            logger.warning("wandb not installed -- skipping log")

    logger.info("inference.py done")


if __name__ == "__main__":
    main()
