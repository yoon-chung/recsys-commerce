"""ensemble_v5_zscore / run.py -- per-MODEL global z-score normalization.

v4 (per-user min-max) all-negative -> hypothesis: per-user normalization
이 TIFU 의 "이 user 한테 확신 있음" vs "잘 모름" 정보 손실.

v5 = per-model **global** z-score (모든 user 통합):
    score' = (score - model_mean) / model_std
이러면 TIFU 의 overall confidence 보존, weight 가 진짜 model 강도 반영.

같은 5 모델 + 같은 weight grid 로 비교 -- normalization 만 다름.

Usage:
    python run.py
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

from core.data_loader import load_train_data, load_mappings  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)


MODELS = {
    "tifu":      PROJECT_ROOT / "experiments" / "exp_007_tifu_knn" / "predictions.parquet",
    "bsarec":    PROJECT_ROOT / "experiments" / "exp_002e_bsarec_4w_full" / "predictions.parquet",
    "mbstr":     PROJECT_ROOT / "experiments" / "exp_009_mbstr" / "predictions.parquet",
    "bsarec_cl": PROJECT_ROOT / "experiments" / "exp_006_bsarec_cl" / "predictions.parquet",
    "bert4rec":  PROJECT_ROOT / "experiments" / "exp_005_bert4rec" / "predictions.parquet",
}


CONFIGS: list[tuple[str, dict]] = [
    ("tifu_only",        {"tifu": 1.0}),
    ("team",             {"tifu": 0.66, "bsarec": 0.08, "mbstr": 0.26}),
    ("t50_b25_m25",      {"tifu": 0.50, "bsarec": 0.25, "mbstr": 0.25}),
    ("t75_b10_m15",      {"tifu": 0.75, "bsarec": 0.10, "mbstr": 0.15}),
    ("t80_b10_m10",      {"tifu": 0.80, "bsarec": 0.10, "mbstr": 0.10}),
    ("t75_m25",          {"tifu": 0.75, "mbstr": 0.25}),
    ("t75_b25",          {"tifu": 0.75, "bsarec": 0.25}),
    ("t60_b10_m20_c10",  {"tifu": 0.60, "bsarec": 0.10, "mbstr": 0.20, "bsarec_cl": 0.10}),
    ("t60_b10_m15_c10_e5", {"tifu": 0.60, "bsarec": 0.10, "mbstr": 0.15, "bsarec_cl": 0.10, "bert4rec": 0.05}),
]


def normalize_per_model_zscore(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Global per-MODEL z-score normalization across all (user, item) pairs."""
    df = pred_df.copy()
    mean = df["score"].mean()
    std = df["score"].std()
    if std == 0:
        df["norm_score"] = 0.0
    else:
        df["norm_score"] = (df["score"] - mean) / std
    return df[["user_id", "item_id", "norm_score"]]


def weighted_score_ensemble(
    preds: dict[str, pd.DataFrame],
    weights: dict[str, float],
    top_n: int = 50,
) -> pd.DataFrame:
    parts = []
    for name, w in weights.items():
        if w == 0 or name not in preds:
            continue
        normed = normalize_per_model_zscore(preds[name])
        normed["weighted"] = normed["norm_score"] * w
        parts.append(normed[["user_id", "item_id", "weighted"]])

    long = pd.concat(parts, ignore_index=True)
    fused = (
        long.groupby(["user_id", "item_id"], sort=False)["weighted"]
        .sum()
        .reset_index(name="score")
    )
    fused["rank"] = (
        fused.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    fused = fused[fused["rank"] <= top_n].copy()
    fused = fused.sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--items-per-user", type=int, default=10)
    parser.add_argument("--train-data", default="/root/data/train.parquet")
    parser.add_argument("--sample-submission", default="/root/data/sample_submission.csv")
    parser.add_argument("--ease-saved",
                        default=str(PROJECT_ROOT / "experiments" / "exp_001_ease" / "saved"))
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--gt-event-types", nargs="+", default=["purchase"])
    parser.add_argument("--no-submission", dest="make_submission", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("loading predictions ...")
    preds: dict[str, pd.DataFrame] = {}
    for name, path in MODELS.items():
        if not path.exists():
            logger.warning("  %s MISSING (%s), SKIP", name, path)
            continue
        df = pd.read_parquet(path)
        preds[name] = df
        logger.info("  %s: %s rows, score mean=%.4f std=%.4f",
                    name, f"{len(df):,}", df["score"].mean(), df["score"].std())

    ease_saved = Path(args.ease_saved)
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    logger.info("val_gt %s rows, %s eval_users",
                f"{len(val_gt_eval):,}", f"{len(eval_users):,}")

    # baseline singles
    logger.info("baseline single-model self-val:")
    single_scores = {}
    for name, df in preds.items():
        n = ndcg_at_k_from_df(df, val_gt_eval, k=10)
        r = recall_at_k_from_df(df, val_gt_eval, k=10)
        single_scores[name] = {"ndcg10": n, "recall10": r}
        logger.info("  %-12s NDCG@10=%.6f  recall@10=%.6f", name, n, r)
    best_single_name = max(single_scores, key=lambda k: single_scores[k]["ndcg10"])
    best_single_ndcg = single_scores[best_single_name]["ndcg10"]

    logger.info("=" * 70)
    logger.info("weighted score ensemble grid (per-MODEL z-score):")
    logger.info("%-25s %-12s %-12s %-12s", "config", "NDCG@10", "recall@10", "Δ vs best")

    results = []
    fused_parquets = {}
    t0 = time.time()
    for cfg_name, weights_dict in CONFIGS:
        active = {k: v for k, v in weights_dict.items() if k in preds and v > 0}
        if not active:
            continue
        if len(active) == 1:
            (only_name,) = active.keys()
            fused = preds[only_name]
        else:
            fused = weighted_score_ensemble(preds, active, top_n=args.top_n)
        n = ndcg_at_k_from_df(fused, val_gt_eval, k=10)
        r = recall_at_k_from_df(fused, val_gt_eval, k=10)
        delta = n - best_single_ndcg
        results.append({
            "config": cfg_name,
            "ndcg10": n,
            "recall10": r,
            "delta_vs_best_single": delta,
            "weights": active,
        })
        logger.info("  %-25s %.6f     %.6f     %+.6f", cfg_name, n, r, delta)
        fused_parquets[cfg_name] = fused
    logger.info("grid sweep done in %.1fs", time.time() - t0)

    results.sort(key=lambda x: x["ndcg10"], reverse=True)
    best = results[0]
    logger.info("=" * 70)
    logger.info("BEST: %s   NDCG=%.6f   recall=%.6f   Δ=%+.6f",
                best["config"], best["ndcg10"], best["recall10"],
                best["delta_vs_best_single"])

    best_parquet = HERE / f"fused_best_{best['config']}.parquet"
    fused_parquets[best["config"]].to_parquet(best_parquet)
    logger.info("wrote %s", best_parquet)

    results_dump = {
        "single_scores": single_scores,
        "normalization": "per-model global z-score",
        "configs": [
            {
                "name": r["config"],
                "weights": r["weights"],
                "ndcg10": r["ndcg10"],
                "recall10": r["recall10"],
                "delta_vs_best_single": r["delta_vs_best_single"],
            }
            for r in results
        ],
        "best": {
            "config": best["config"],
            "ndcg10": best["ndcg10"],
            "recall10": best["recall10"],
            "weights": best["weights"],
        },
    }
    with open(HERE / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_dump, f, indent=2, default=str)
    logger.info("wrote results.json")

    if args.make_submission and best["delta_vs_best_single"] > 0:
        sample = pd.read_csv(args.sample_submission)
        all_users = sample["user_id"].drop_duplicates().tolist()
        mappings = load_mappings(str(ease_saved / "mappings"))

        df_full = load_train_data(args.train_data)
        train_df, _ = time_based_split(
            df_full, val_days=args.val_days, gt_event_types=args.gt_event_types
        )
        popularity = compute_popularity(train_df, top_n=args.top_n)

        output_csv = HERE / f"output_{best['config']}.csv"
        predictions_to_submission(
            pred_path=str(best_parquet),
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
            raise RuntimeError("validate_submission FAILED")
        logger.info("validate_submission OK -- %s ready", output_csv)
    elif args.make_submission:
        logger.info("best ensemble <= best single (Δ %+.6f) -- submission SKIP",
                    best["delta_vs_best_single"])

    logger.info("run.py done")


if __name__ == "__main__":
    main()
