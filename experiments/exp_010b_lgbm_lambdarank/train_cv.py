"""exp_010b_lgbm_lambdarank / train_cv.py -- 5-fold CV with LGBMRanker (lambdarank).

exp_010 의 binary classifier 를 LambdaRank 로 swap. NDCG@10 직접 최적화.
Features 는 exp_010 의 features_all/ 재사용 (build_features.py 안 돌림).

Key difference vs exp_010:
    - objective: binary -> lambdarank
    - metric: binary_logloss -> ndcg
    - lgb.train -> LGBMRanker.fit (group 파라미터 필요)
    - per-user row count 를 group 으로 전달

Output:
    saved/lgbm_fold{i}.txt          model dumps
    cv_results.json                 per-fold metrics + overall OOF NDCG
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
    exclude = {"user_id", "item_id", "label", "is_eval_user",
               "user_top_brand", "item_brand", "user_top_category", "item_category"}
    return [c for c in df.columns if c not in exclude
            and not df[c].dtype.kind in {"O"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--features", default=None)
    parser.add_argument("--saved-dir", default=str(HERE / "saved"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    features_path = args.features or str((HERE / cfg["features_dir"]).resolve())
    saved_dir = Path(args.saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb  # noqa: PLC0415

    # ---- 1. Load features (exp_010 의 features_all/ 재사용) -----------
    logger.info("loading features from %s ...", features_path)
    df_all = pd.read_parquet(features_path)
    logger.info("all features: %s rows", f"{len(df_all):,}")

    # ---- 2. Filter to eval_users ---------------------------------------
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

    # ---- 5. Train per fold (LambdaRank) --------------------------------
    lgbm_p = cfg["lgbm"]
    ranker_params = {
        "objective": lgbm_p["objective"],          # "lambdarank"
        "metric": lgbm_p["metric"],                # "ndcg"
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
        "n_estimators": lgbm_p["num_boost_round"],
    }
    ndcg_eval_at = lgbm_p.get("ndcg_eval_at", [10])

    fold_metrics = []
    oof_pred = np.zeros(len(df_eval), dtype=np.float32)

    feature_importance_acc: dict[str, float] = {c: 0.0 for c in feat_cols}

    t_start = time.time()
    for fold in range(n_folds):
        tr = df_eval[df_eval["fold"] != fold].sort_values("user_id", kind="mergesort")
        vl = df_eval[df_eval["fold"] == fold].sort_values("user_id", kind="mergesort")
        logger.info("fold %d: train=%s rows (%s users, %s pos), val=%s rows (%s users, %s pos)",
                    fold,
                    f"{len(tr):,}", f"{tr['user_id'].nunique():,}", f"{int(tr['label'].sum()):,}",
                    f"{len(vl):,}", f"{vl['user_id'].nunique():,}", f"{int(vl['label'].sum()):,}")

        tr_groups = tr.groupby("user_id", sort=False).size().tolist()
        vl_groups = vl.groupby("user_id", sort=False).size().tolist()

        ranker = lgb.LGBMRanker(**ranker_params, n_jobs=-1)
        ranker.fit(
            tr[feat_cols], tr["label"],
            group=tr_groups,
            eval_set=[(vl[feat_cols], vl["label"])],
            eval_group=[vl_groups],
            eval_at=ndcg_eval_at,
            callbacks=[
                lgb.early_stopping(lgbm_p["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        vl_pred = ranker.predict(vl[feat_cols], num_iteration=ranker.best_iteration_)
        # OOF index 매칭: vl 의 원본 index 기준
        oof_pred[vl.index] = vl_pred.astype(np.float32)

        # save (Booster format -- exp_010 inference.py 호환)
        ranker.booster_.save_model(str(saved_dir / f"lgbm_fold{fold}.txt"))

        # importance
        imp = ranker.booster_.feature_importance(importance_type="gain")
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
            "best_iteration": ranker.best_iteration_,
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

    # Compare to TIFU and exp_010 binary
    tifu_pred = pd.read_parquet((HERE / cfg["model_predictions"]["tifu"]).resolve())
    tifu_eval = tifu_pred[tifu_pred["user_id"].isin(eval_users)]
    tifu_ndcg = ndcg_at_k_from_df(tifu_eval, val_gt_eval, k=10)
    tifu_recall = recall_at_k_from_df(tifu_eval, val_gt_eval, k=10)
    logger.info("baseline TIFU NDCG=%.6f recall=%.6f", tifu_ndcg, tifu_recall)
    logger.info("Δ vs TIFU NDCG: %+.6f", overall_ndcg - tifu_ndcg)
    logger.info("(exp_010 binary OOF NDCG was 0.3418, current LambdaRank: %.6f, Δ: %+.6f)",
                overall_ndcg, overall_ndcg - 0.3418)

    # ---- 7. Save metrics + importance ----------------------------------
    top_imp = sorted(feature_importance_acc.items(), key=lambda kv: -kv[1])[:20]
    results = {
        "objective": "lambdarank",
        "fold_metrics": fold_metrics,
        "overall_ndcg10": float(overall_ndcg),
        "overall_recall10": float(overall_recall),
        "tifu_baseline_ndcg10": float(tifu_ndcg),
        "tifu_baseline_recall10": float(tifu_recall),
        "delta_vs_tifu": float(overall_ndcg - tifu_ndcg),
        "delta_vs_exp_010_binary": float(overall_ndcg - 0.3418),
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