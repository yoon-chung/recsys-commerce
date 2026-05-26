"""exp_010_lgbm_reranker / train_cv.py -- 5-fold CV on eval_users.

Loads features_all.parquet -> filter to eval_users (label-present) ->
5-fold CV by user_id -> train LGBM per fold -> aggregate self-val NDCG@10.
Saves all fold models.

Output:
    saved/lgbm_fold{i}.txt          model dumps
    cv_results.json                 per-fold metrics + overall NDCG
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

from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402

logger = logging.getLogger(__name__)


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything except user_id, item_id, label, is_eval_user)."""
    exclude = {"user_id", "item_id", "label", "is_eval_user",
               "user_top_brand", "item_brand", "user_top_category", "item_category"}
    return [c for c in df.columns if c not in exclude
            and not df[c].dtype.kind in {"O"}]   # drop object columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--features", default=str(HERE / "cache" / "features_all"))
    parser.add_argument("--saved-dir", default=str(HERE / "saved"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    saved_dir = Path(args.saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb  # noqa: PLC0415

    # ---- 1. Load features ----------------------------------------------
    logger.info("loading features ...")
    df_all = pd.read_parquet(args.features)
    logger.info("all features: %s rows", f"{len(df_all):,}")

    # ---- 2. Filter to eval_users (those with label coverage) -----------
    df_eval = df_all[df_all["is_eval_user"] == 1].copy()
    logger.info("eval users subset: %s rows, %s users, %s positives",
                f"{len(df_eval):,}",
                f"{df_eval['user_id'].nunique():,}",
                f"{int(df_eval['label'].sum()):,}")

    feat_cols = select_feature_cols(df_eval)
    logger.info("feature cols (n=%d): %s", len(feat_cols), feat_cols[:8] + ["..."])

    # ---- 3. Load val_gt for NDCG eval ----------------------------------
    val_gt_df = pd.read_parquet((HERE / cfg["ease_saved"] / "val_gt.parquet").resolve())
    with open((HERE / cfg["ease_saved"] / "eval_users.json").resolve(), encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]

    # ---- 4. 5-fold CV by user_id ---------------------------------------
    n_folds = int(cfg["n_folds"])
    seed = int(cfg["random_seed"])
    users = df_eval["user_id"].drop_duplicates().sort_values().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    fold_assign = {u: i % n_folds for i, u in enumerate(users)}
    df_eval["fold"] = df_eval["user_id"].map(fold_assign).astype(np.int8)

    # ---- 5. Train per fold ---------------------------------------------
    lgbm_p = cfg["lgbm"]
    params = {
        "objective": lgbm_p["objective"],
        "metric": lgbm_p["metric"],
        "learning_rate": lgbm_p["learning_rate"],
        "num_leaves": lgbm_p["num_leaves"],
        "max_depth": lgbm_p["max_depth"],
        "min_data_in_leaf": lgbm_p["min_data_in_leaf"],
        "feature_fraction": lgbm_p["feature_fraction"],
        "bagging_fraction": lgbm_p["bagging_fraction"],
        "bagging_freq": lgbm_p["bagging_freq"],
        "lambda_l1": lgbm_p["lambda_l1"],
        "lambda_l2": lgbm_p["lambda_l2"],
        "verbose": -1,
        "seed": seed,
    }

    fold_metrics = []
    oof_pred = np.zeros(len(df_eval), dtype=np.float32)
    oof_idx = df_eval.index.to_numpy()

    feature_importance_acc: dict[str, float] = {c: 0.0 for c in feat_cols}

    t_start = time.time()
    for fold in range(n_folds):
        tr = df_eval[df_eval["fold"] != fold]
        vl = df_eval[df_eval["fold"] == fold]
        logger.info("fold %d: train=%s rows (%s users, %s pos), val=%s rows (%s users, %s pos)",
                    fold,
                    f"{len(tr):,}", f"{tr['user_id'].nunique():,}", f"{int(tr['label'].sum()):,}",
                    f"{len(vl):,}", f"{vl['user_id'].nunique():,}", f"{int(vl['label'].sum()):,}")

        dtrain = lgb.Dataset(tr[feat_cols], label=tr["label"])
        dval = lgb.Dataset(vl[feat_cols], label=vl["label"], reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=lgbm_p["num_boost_round"],
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(lgbm_p["early_stopping_rounds"]),
                lgb.log_evaluation(period=100),
            ],
        )

        vl_pred = model.predict(vl[feat_cols])
        oof_pred[df_eval["fold"].to_numpy() == fold] = vl_pred

        # save
        model.save_model(str(saved_dir / f"lgbm_fold{fold}.txt"))

        # importance
        imp = model.feature_importance(importance_type="gain")
        for c, v in zip(feat_cols, imp):
            feature_importance_acc[c] += float(v) / n_folds

        # quick val NDCG on this fold's users
        vl_pred_df = vl[["user_id", "item_id"]].copy()
        vl_pred_df["score"] = vl_pred
        vl_pred_df["rank"] = (
            vl_pred_df.groupby("user_id")["score"]
            .rank(method="first", ascending=False)
            .astype(np.int32)
        )
        vl_pred_df = vl_pred_df[vl_pred_df["rank"] <= 10]
        vl_gt = val_gt_eval[val_gt_eval["user_id"].isin(vl["user_id"].unique())]
        if len(vl_gt) > 0:
            fold_ndcg = ndcg_at_k_from_df(vl_pred_df, vl_gt, k=10)
            fold_recall = recall_at_k_from_df(vl_pred_df, vl_gt, k=10)
        else:
            fold_ndcg = fold_recall = 0.0
        fold_metrics.append({
            "fold": fold,
            "best_iteration": model.best_iteration,
            "ndcg10": fold_ndcg,
            "recall10": fold_recall,
        })
        logger.info("  fold %d -> NDCG=%.6f recall=%.6f", fold, fold_ndcg, fold_recall)

    logger.info("all folds done in %.1fs", time.time() - t_start)

    # ---- 6. Overall OOF NDCG -------------------------------------------
    df_eval["oof_pred"] = oof_pred
    oof_df = df_eval[["user_id", "item_id", "oof_pred"]].rename(columns={"oof_pred": "score"})
    oof_df["rank"] = (
        oof_df.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    oof_top10 = oof_df[oof_df["rank"] <= 10]
    overall_ndcg = ndcg_at_k_from_df(oof_top10, val_gt_eval, k=10)
    overall_recall = recall_at_k_from_df(oof_top10, val_gt_eval, k=10)
    logger.info("OOF overall NDCG@10=%.6f  recall@10=%.6f", overall_ndcg, overall_recall)

    # Compare to single best (TIFU exp_007b)
    tifu_pred = pd.read_parquet((HERE / cfg["model_predictions"]["tifu"]).resolve())
    tifu_eval = tifu_pred[tifu_pred["user_id"].isin(eval_users)]
    tifu_ndcg = ndcg_at_k_from_df(tifu_eval, val_gt_eval, k=10)
    tifu_recall = recall_at_k_from_df(tifu_eval, val_gt_eval, k=10)
    logger.info("baseline TIFU NDCG=%.6f recall=%.6f", tifu_ndcg, tifu_recall)
    logger.info("Δ vs TIFU NDCG: %+.6f", overall_ndcg - tifu_ndcg)

    # ---- 7. Save metrics + importance ----------------------------------
    top_imp = sorted(feature_importance_acc.items(), key=lambda kv: -kv[1])[:20]
    results = {
        "fold_metrics": fold_metrics,
        "overall_ndcg10": float(overall_ndcg),
        "overall_recall10": float(overall_recall),
        "tifu_baseline_ndcg10": float(tifu_ndcg),
        "tifu_baseline_recall10": float(tifu_recall),
        "delta_vs_tifu": float(overall_ndcg - tifu_ndcg),
        "top_20_features_by_gain": [
            {"feature": c, "gain": v} for c, v in top_imp
        ],
        "n_features": len(feat_cols),
    }
    with open(saved_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote cv_results.json")

    logger.info("top 20 features by gain:")
    for c, v in top_imp:
        logger.info("  %-30s  %.1f", c, v)


if __name__ == "__main__":
    main()
