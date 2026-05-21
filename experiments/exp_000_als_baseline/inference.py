"""exp_000_als_baseline / inference.py

Load the ALS model saved by train.py and produce:
    predictions.parquet  -- top-N candidates per known user (ensemble input)
    output.csv           -- competition submission for all 638,257 users
plus measured self-val NDCG@10 / recall@10 logged to stdout and wandb.

Usage:
    python inference.py                       # uses ./config.yaml
    python inference.py --config path/to.yaml
    python inference.py --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import load_npz

# NOTE: `implicit.als.AlternatingLeastSquares` is a factory FUNCTION that
# dispatches to the GPU or CPU concrete class based on `use_gpu`. It has no
# `.load` classmethod. Import the concrete class matching the saved backend.
def _als_class(use_gpu: bool):
    if use_gpu:
        from implicit.gpu.als import AlternatingLeastSquares as _Cls
    else:
        from implicit.cpu.als import AlternatingLeastSquares as _Cls
    return _Cls

from shared.data_loader import load_train_data, load_mappings  # noqa: E402
from shared.validation import time_based_split  # noqa: E402
from shared.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from shared.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)


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

    # ---- 1. Load model + train artifacts ----------------------------------
    logger.info("loading ALS model from %s", saved_dir / "als.npz")
    AlsClass = _als_class(cfg["use_gpu"])
    model = AlsClass.load(str(saved_dir / "als.npz"))
    interactions = load_npz(str(saved_dir / "interactions.npz"))
    mappings = load_mappings(str(saved_dir / "mappings"))
    val_gt_df = pd.read_parquet(saved_dir / "val_gt.parquet")
    with open(saved_dir / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    logger.info(
        "loaded: interactions shape=%s, %s users in mappings, val_gt %s rows, %s eval_users",
        interactions.shape,
        f"{len(mappings['user2idx']):,}",
        f"{len(val_gt_df):,}",
        f"{len(eval_users):,}",
    )

    # ---- 2. Recompute popularity fallback from train portion -------------
    # Re-load + re-split rather than save train_df.parquet (~200MB).
    df_full = load_train_data(cfg["train_data"])
    train_df, _ = time_based_split(
        df_full, val_days=cfg["val_days"], gt_event_types=cfg["gt_event_types"]
    )
    popularity = compute_popularity(train_df, top_n=cfg["top_n"])

    # ---- 3. All-users list from sample_submission ------------------------
    sample = pd.read_csv(cfg["sample_submission"])
    all_users = sample["user_id"].drop_duplicates().tolist()
    logger.info("all_users from sample_submission: %s", f"{len(all_users):,}")

    # ---- 4. Recommend top-N for users known to the model -----------------
    known_users = [u for u in all_users if u in mappings["user2idx"]]
    cold_start = len(all_users) - len(known_users)
    known_user_idxs = np.fromiter(
        (mappings["user2idx"][u] for u in known_users), dtype=np.int64, count=len(known_users)
    )
    logger.info(
        "predicting: known=%s, cold-start (popularity-only)=%s",
        f"{len(known_users):,}",
        f"{cold_start:,}",
    )

    top_n = cfg["top_n"]
    ids, scores = model.recommend(
        known_user_idxs,
        interactions[known_user_idxs],
        N=top_n,
        filter_already_liked_items=cfg["filter_already_liked"],
    )
    # ids shape: (len(known_users), top_n); scores same. -1 marks unfilled slots.

    # ---- 5. Decode + build predictions.parquet ---------------------------
    n_known = len(known_users)
    user_repeat = np.repeat(np.asarray(known_users, dtype=object), top_n)
    item_idx_flat = ids.reshape(-1)
    score_flat = scores.reshape(-1)
    rank_flat = np.tile(np.arange(1, top_n + 1, dtype=np.int32), n_known)

    valid = item_idx_flat >= 0
    n_invalid = int((~valid).sum())
    if n_invalid:
        logger.info("dropping %s implicit recommend -1 slots", f"{n_invalid:,}")

    idx2item_arr = np.empty(len(mappings["idx2item"]), dtype=object)
    for j, it in mappings["idx2item"].items():
        idx2item_arr[j] = it

    pred_df = pd.DataFrame(
        {
            "user_id": user_repeat[valid],
            "item_id": idx2item_arr[item_idx_flat[valid]],
            "score": score_flat[valid].astype(np.float64, copy=False),
            "rank": rank_flat[valid],
        }
    )
    pred_path = out_dir / "predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info("wrote %s (%s rows)", pred_path, f"{len(pred_df):,}")

    # ---- 6. Self-val NDCG@10 + recall@10 ---------------------------------
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    ndcg10 = ndcg_at_k_from_df(pred_df, val_gt_eval, k=10)
    recall10 = recall_at_k_from_df(pred_df, val_gt_eval, k=10)
    logger.info(
        "self-val (last %d days, gt=%s, eval_users=%s):",
        cfg["val_days"], cfg["gt_event_types"], f"{len(eval_users):,}",
    )
    logger.info("  NDCG@10   = %.6f", ndcg10)
    logger.info("  recall@10 = %.6f", recall10)

    # ---- 7. Submission CSV + validation ----------------------------------
    output_csv = out_dir / "output.csv"
    predictions_to_submission(
        pred_path=str(pred_path),
        output_csv=str(output_csv),
        all_users=all_users,
        mappings=mappings,
        popularity_fallback=popularity,
        items_per_user=cfg["items_per_user"],
    )
    ok = validate_submission(str(output_csv), expected_users=len(all_users), items_per_user=cfg["items_per_user"])
    if not ok:
        raise RuntimeError("validate_submission FAILED -- do not upload")
    logger.info("validate_submission OK")

    # ---- 8. wandb log ----------------------------------------------------
    if args.use_wandb:
        try:
            import wandb

            # Resume train.py's run (id was persisted to saved/wandb_run_id.txt)
            # so train + inference live as a single wandb run, not two same-named
            # runs. Fall back to a fresh run only if no id was recorded (e.g. train
            # was invoked with --no-wandb).
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
