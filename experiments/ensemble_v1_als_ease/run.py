"""ensemble_v1_als_ease / run.py — RRF combine of exp_000 (ALS) + exp_001 (EASE).

Goal: validate the RRF pipeline + check the hypothesis that two different-family
models (MF factor vs item-item) lift each other vs either alone.

Inputs (relative to this script):
    ../exp_000_als_baseline/predictions.parquet   (top-50 + score + rank)
    ../exp_001_ease/predictions.parquet           (top-50 + score + rank)
    ../exp_001_ease/saved/val_gt.parquet          (shared split — same val_days+gt)
    ../exp_001_ease/saved/eval_users.json
    ../exp_001_ease/saved/mappings/               (for popularity fallback)

Outputs (this folder):
    fused_predictions.parquet   top-50 RRF (gitignored)
    output.csv                  6,382,570 rows submission-ready (gitignored)

Usage:
    python run.py
    python run.py --k-const 30 --no-submission     # ablation
    python run.py --no-wandb
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
import yaml  # noqa: F401  # imported for symmetry with other experiments

from core.data_loader import load_train_data, load_mappings  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)
from core.ensemble import rrf_combine  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--als-pred",
        default=str(here.parent / "exp_000_als_baseline" / "predictions.parquet"),
    )
    parser.add_argument(
        "--ease-pred",
        default=str(here.parent / "exp_001_ease" / "predictions.parquet"),
    )
    parser.add_argument(
        "--ease-saved",
        default=str(here.parent / "exp_001_ease" / "saved"),
        help="reuse val_gt.parquet, eval_users.json, mappings/ from exp_001",
    )
    parser.add_argument("--train-data", default="/root/data/train.parquet")
    parser.add_argument("--sample-submission", default="/root/data/sample_submission.csv")
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--gt-event-types", nargs="+", default=["purchase"])
    parser.add_argument("--k-const", type=int, default=60, help="RRF constant")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--items-per-user", type=int, default=10)
    parser.add_argument("--no-submission", dest="make_submission", action="store_false")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-entity", default="yooni0125-")
    parser.add_argument("--wandb-project", default="cy-commerce-recsys")
    parser.add_argument("--run-name", default="ensemble_v1_als_ease")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ease_saved = Path(args.ease_saved)

    # ---- 1. Load both predictions ----------------------------------------
    logger.info("loading ALS predictions: %s", args.als_pred)
    pred_als = pd.read_parquet(args.als_pred)
    logger.info("  ALS  %s rows, %s users", f"{len(pred_als):,}",
                f"{pred_als['user_id'].nunique():,}")

    logger.info("loading EASE predictions: %s", args.ease_pred)
    pred_ease = pd.read_parquet(args.ease_pred)
    logger.info("  EASE %s rows, %s users", f"{len(pred_ease):,}",
                f"{pred_ease['user_id'].nunique():,}")

    # ---- 2. Shared val_gt + eval_users (reuse exp_001 split) -------------
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    logger.info("val_gt %s rows, %s eval_users",
                f"{len(val_gt_eval):,}", f"{len(eval_users):,}")

    # ---- 3. Baseline single-model NDCG (sanity vs exp logs) --------------
    t0 = time.time()
    als_ndcg10 = ndcg_at_k_from_df(pred_als, val_gt_eval, k=10)
    als_recall10 = recall_at_k_from_df(pred_als, val_gt_eval, k=10)
    ease_ndcg10 = ndcg_at_k_from_df(pred_ease, val_gt_eval, k=10)
    ease_recall10 = recall_at_k_from_df(pred_ease, val_gt_eval, k=10)
    logger.info("baseline single-model self-val (took %.1fs):", time.time() - t0)
    logger.info("  ALS  NDCG@10=%.6f  recall@10=%.6f", als_ndcg10, als_recall10)
    logger.info("  EASE NDCG@10=%.6f  recall@10=%.6f", ease_ndcg10, ease_recall10)

    # ---- 4. RRF fuse ------------------------------------------------------
    logger.info("RRF combine: k_const=%d top_n=%d", args.k_const, args.top_n)
    t0 = time.time()
    fused = rrf_combine([pred_als, pred_ease], k_const=args.k_const, top_n=args.top_n)
    logger.info("  fused %s rows, %s users, took %.1fs",
                f"{len(fused):,}", f"{fused['user_id'].nunique():,}", time.time() - t0)

    # ---- 5. Fused NDCG ---------------------------------------------------
    fused_ndcg10 = ndcg_at_k_from_df(fused, val_gt_eval, k=10)
    fused_recall10 = recall_at_k_from_df(fused, val_gt_eval, k=10)
    logger.info("RRF self-val:")
    logger.info("  fused NDCG@10  = %.6f  (ALS %+.6f / EASE %+.6f)",
                fused_ndcg10, fused_ndcg10 - als_ndcg10, fused_ndcg10 - ease_ndcg10)
    logger.info("  fused recall@10= %.6f  (ALS %+.6f / EASE %+.6f)",
                fused_recall10, fused_recall10 - als_recall10, fused_recall10 - ease_recall10)

    # ---- 6. Write fused parquet ------------------------------------------
    fused_path = here / "fused_predictions.parquet"
    fused.to_parquet(fused_path)
    logger.info("wrote %s (%s rows)", fused_path, f"{len(fused):,}")

    # ---- 7. Optional submission CSV --------------------------------------
    if args.make_submission:
        sample = pd.read_csv(args.sample_submission)
        all_users = sample["user_id"].drop_duplicates().tolist()
        mappings = load_mappings(str(ease_saved / "mappings"))

        df_full = load_train_data(args.train_data)
        train_df, _ = time_based_split(
            df_full, val_days=args.val_days, gt_event_types=args.gt_event_types
        )
        popularity = compute_popularity(train_df, top_n=args.top_n)

        output_csv = here / "output.csv"
        predictions_to_submission(
            pred_path=str(fused_path),
            output_csv=str(output_csv),
            all_users=all_users,
            mappings=mappings,
            popularity_fallback=popularity,
            items_per_user=args.items_per_user,
        )
        ok = validate_submission(
            str(output_csv),
            expected_users=len(all_users),
            items_per_user=args.items_per_user,
        )
        if not ok:
            raise RuntimeError("validate_submission FAILED -- do not upload")
        logger.info("validate_submission OK -- %s ready", output_csv)

    # ---- 8. wandb --------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb

            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.run_name,
                config={
                    "k_const": args.k_const,
                    "top_n": args.top_n,
                    "items_per_user": args.items_per_user,
                    "components": ["exp_000_als_baseline", "exp_001_ease"],
                },
            )
            wandb.log(
                {
                    "als_ndcg@10": als_ndcg10,
                    "als_recall@10": als_recall10,
                    "ease_ndcg@10": ease_ndcg10,
                    "ease_recall@10": ease_recall10,
                    "fused_ndcg@10": fused_ndcg10,
                    "fused_recall@10": fused_recall10,
                    "lift_vs_als_ndcg": fused_ndcg10 - als_ndcg10,
                    "lift_vs_ease_ndcg": fused_ndcg10 - ease_ndcg10,
                    "lift_vs_als_recall": fused_recall10 - als_recall10,
                    "lift_vs_ease_recall": fused_recall10 - ease_recall10,
                }
            )
            run.finish()
        except ImportError:
            logger.warning("wandb not installed; skipping log")

    logger.info("run.py done")


if __name__ == "__main__":
    main()
