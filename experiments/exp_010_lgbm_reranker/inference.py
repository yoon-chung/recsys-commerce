"""exp_010_lgbm_reranker / inference.py -- score all 638k users with LGBM ensemble.

Loads 5 fold models, averages predictions on full feature table (all users),
writes top-10 per user to predictions.parquet + output.csv submission.

Cold-start users (no candidates) -> popularity fallback (via core.submission).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from core.data_loader import load_train_data, load_mappings  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"user_id", "item_id", "label", "is_eval_user",
               "user_top_brand", "item_brand", "user_top_category", "item_category"}
    return [c for c in df.columns if c not in exclude
            and not df[c].dtype.kind in {"O"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--features", default=str(HERE / "cache" / "features_all"))
    parser.add_argument("--saved-dir", default=str(HERE / "saved"))
    parser.add_argument("--out-dir", default=str(HERE))
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

    import lightgbm as lgb  # noqa: PLC0415

    # ---- 1. Load features ----------------------------------------------
    logger.info("loading features ...")
    df_all = pd.read_parquet(args.features)
    feat_cols = select_feature_cols(df_all)
    logger.info("all features: %s rows, %d feature cols",
                f"{len(df_all):,}", len(feat_cols))

    # ---- 2. Load fold models -------------------------------------------
    fold_models = sorted(saved_dir.glob("lgbm_fold*.txt"))
    logger.info("found %d fold models", len(fold_models))

    # ---- 3. Predict (ensemble across folds) ----------------------------
    preds_acc = np.zeros(len(df_all), dtype=np.float64)
    X = df_all[feat_cols]
    t0 = time.time()
    for fp in fold_models:
        model = lgb.Booster(model_file=str(fp))
        preds_acc += model.predict(X)
    preds_acc /= len(fold_models)
    logger.info("scoring done in %.1fs", time.time() - t0)
    df_all["lgbm_score"] = preds_acc.astype(np.float32)

    # ---- 4. Top-10 per user --------------------------------------------
    pred_df = df_all[["user_id", "item_id", "lgbm_score"]].rename(columns={"lgbm_score": "score"})
    pred_df["rank"] = (
        pred_df.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    top_n = 50
    pred_df = pred_df[pred_df["rank"] <= top_n].copy()
    pred_df = pred_df.sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)
    pred_df["score"] = pred_df["score"].astype(np.float64)
    pred_path = out_dir / "predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info("wrote %s (%s rows, %s users)",
                pred_path, f"{len(pred_df):,}", f"{pred_df['user_id'].nunique():,}")

    # ---- 5. Self-val NDCG ----------------------------------------------
    val_gt_df = pd.read_parquet((HERE / cfg["ease_saved"] / "val_gt.parquet").resolve())
    with open((HERE / cfg["ease_saved"] / "eval_users.json").resolve(), encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    ndcg = ndcg_at_k_from_df(pred_df, val_gt_eval, k=10)
    recall = recall_at_k_from_df(pred_df, val_gt_eval, k=10)
    logger.info("self-val (test) NDCG@10=%.6f recall@10=%.6f", ndcg, recall)

    # baseline TIFU
    tifu_pred = pd.read_parquet((HERE / cfg["model_predictions"]["tifu"]).resolve())
    tifu_eval = tifu_pred[tifu_pred["user_id"].isin(eval_users)]
    tifu_ndcg = ndcg_at_k_from_df(tifu_eval, val_gt_eval, k=10)
    tifu_recall = recall_at_k_from_df(tifu_eval, val_gt_eval, k=10)
    logger.info("baseline TIFU NDCG=%.6f recall=%.6f  Δ vs TIFU NDCG: %+.6f",
                tifu_ndcg, tifu_recall, ndcg - tifu_ndcg)

    # ---- 6. Submission CSV ---------------------------------------------
    sample = pd.read_csv(cfg["sample_submission"])
    all_users = sample["user_id"].drop_duplicates().tolist()
    mappings = load_mappings(str((HERE / cfg["ease_saved"] / "mappings").resolve()))

    df_full = load_train_data(cfg["train_data"])
    train_df, _ = time_based_split(
        df_full, val_days=cfg["val_days"], gt_event_types=cfg["gt_event_types"]
    )
    popularity = compute_popularity(train_df, top_n=top_n)

    output_csv = out_dir / "output.csv"
    predictions_to_submission(
        pred_path=str(pred_path),
        output_csv=str(output_csv),
        all_users=all_users,
        mappings=mappings,
        popularity_fallback=popularity,
        items_per_user=cfg["items_per_user_output"],
    )
    ok = validate_submission(
        str(output_csv),
        expected_users=len(all_users),
        items_per_user=cfg["items_per_user_output"],
    )
    if not ok:
        raise RuntimeError("validate_submission FAILED")
    logger.info("validate_submission OK -- %s ready", output_csv)


if __name__ == "__main__":
    main()
