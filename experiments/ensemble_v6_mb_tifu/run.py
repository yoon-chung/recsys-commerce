"""ensemble_v6_mb_tifu / run.py -- v4 와 동일하지만 TIFU 를 exp_007b 로 교체.

Context:
    v4/v5 (exp_007 TIFU = 1/1/1 event weights) 모두 dead-end (best ensemble <= TIFU).
    exp_007b (multi-behavior 1/3/5) self-val 0.2933 (+0.0011 vs exp_007 0.2922).
    Standalone 미세하지만 candidate reordering 가능 -> ensemble 다른 dynamics?

v4 와 동일 알고리즘 (per-user min-max + weighted score sum).

Algorithm (per user):
    1. union of top-50 candidates from each model
    2. per-model min-max normalize scores (within user's top-50)
    3. final_score(u, i) = Σ_m  w_m · norm_score(u, i, m)
       (item 이 모델 m 의 top-50 밖이면 norm_score = 0)
    4. sort by final_score desc, take top-50

Models (5 available, all DONE):
    tifu     -- exp_007 (public 0.1175, best)
    bsarec   -- exp_002e BSARec 4w_full (public 0.0975, best transformer)
    mbstr    -- exp_009 MB-STR (multi-behavior signal, NEW)
    bsarec_cl-- exp_006 BSARec+CL hybrid
    bert4rec -- exp_005 BERT4Rec

Weight configs (grid sweep + external reference patterns):
    tifu_only         : TIFU 단독 baseline (= public 0.1175)
    team              : 0.66 TIFU + 0.08 BSARec + 0.26 MB-STR  (external reference 패턴)
    team_w_cl         : 위 + 0.05 BSARec+CL
    t_b_m_equal       : 0.5 TIFU + 0.25 BSARec + 0.25 MB-STR
    t_dominant_5      : 0.7 TIFU + 0.1 BSARec + 0.1 MB-STR + 0.05 CL + 0.05 BERT
    + 추가 grid

Usage:
    python run.py                    # grid sweep + best submission
    python run.py --no-submission
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
    "tifu":      PROJECT_ROOT / "experiments" / "exp_007b_tifu_mb" / "predictions.parquet",   # multi-behavior
    "bsarec":    PROJECT_ROOT / "experiments" / "exp_002e_bsarec_4w_full" / "predictions.parquet",
    "mbstr":     PROJECT_ROOT / "experiments" / "exp_009_mbstr" / "predictions.parquet",
    "bsarec_cl": PROJECT_ROOT / "experiments" / "exp_006_bsarec_cl" / "predictions.parquet",
    "bert4rec":  PROJECT_ROOT / "experiments" / "exp_005_bert4rec" / "predictions.parquet",
}


# (cfg_name, dict of model_name -> weight). weight 0 또는 absent 면 skip.
CONFIGS: list[tuple[str, dict]] = [
    ("tifu_only",        {"tifu": 1.0}),
    # external reference 패턴 + 우리 BSARec 대체
    ("team",             {"tifu": 0.66, "bsarec": 0.08, "mbstr": 0.26}),
    # TIFU 약화 변형
    ("t50_b25_m25",      {"tifu": 0.50, "bsarec": 0.25, "mbstr": 0.25}),
    # TIFU 더 dominant
    ("t75_b10_m15",      {"tifu": 0.75, "bsarec": 0.10, "mbstr": 0.15}),
    ("t80_b10_m10",      {"tifu": 0.80, "bsarec": 0.10, "mbstr": 0.10}),
    # 2-model only
    ("t75_m25",          {"tifu": 0.75, "mbstr": 0.25}),
    ("t75_b25",          {"tifu": 0.75, "bsarec": 0.25}),
    # 4-model
    ("t60_b10_m20_c10",  {"tifu": 0.60, "bsarec": 0.10, "mbstr": 0.20, "bsarec_cl": 0.10}),
    # 5-model all
    ("t60_b10_m15_c10_e5", {"tifu": 0.60, "bsarec": 0.10, "mbstr": 0.15, "bsarec_cl": 0.10, "bert4rec": 0.05}),
]


def normalize_per_user(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-user min-max normalize 'score' column within top-N items."""
    df = pred_df.copy()
    # group: per user, normalize score to [0, 1]
    grp = df.groupby("user_id")["score"]
    mn = grp.transform("min")
    mx = grp.transform("max")
    rng = mx - mn
    # avoid div by zero -- if all same, set normed to 1.0
    df["norm_score"] = np.where(rng > 0, (df["score"] - mn) / rng, 1.0)
    return df[["user_id", "item_id", "norm_score"]]


def weighted_score_ensemble(
    preds: dict[str, pd.DataFrame],   # model_name -> pred_df with score
    weights: dict[str, float],         # model_name -> weight
    top_n: int = 50,
) -> pd.DataFrame:
    """Combine models via per-user normalized weighted score.

    Returns DataFrame with columns (user_id, item_id, score, rank), top_n per user.
    """
    parts = []
    for name, w in weights.items():
        if w == 0 or name not in preds:
            continue
        normed = normalize_per_user(preds[name])
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

    # ---- 1. Load models -------------------------------------------------
    logger.info("loading predictions ...")
    preds: dict[str, pd.DataFrame] = {}
    for name, path in MODELS.items():
        if not path.exists():
            logger.warning("  %s MISSING (%s), SKIP", name, path)
            continue
        df = pd.read_parquet(path)
        preds[name] = df
        logger.info("  %s: %s rows, %s users",
                    name, f"{len(df):,}", f"{df['user_id'].nunique():,}")

    # ---- 2. Shared val_gt + eval_users ---------------------------------
    ease_saved = Path(args.ease_saved)
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    logger.info("val_gt %s rows, %s eval_users",
                f"{len(val_gt_eval):,}", f"{len(eval_users):,}")

    # ---- 3. Baseline singles -------------------------------------------
    logger.info("baseline single-model self-val:")
    single_scores = {}
    for name, df in preds.items():
        n = ndcg_at_k_from_df(df, val_gt_eval, k=10)
        r = recall_at_k_from_df(df, val_gt_eval, k=10)
        single_scores[name] = {"ndcg10": n, "recall10": r}
        logger.info("  %-12s NDCG@10=%.6f  recall@10=%.6f", name, n, r)
    best_single_name = max(single_scores, key=lambda k: single_scores[k]["ndcg10"])
    best_single_ndcg = single_scores[best_single_name]["ndcg10"]
    logger.info("best single: %s (NDCG=%.6f)", best_single_name, best_single_ndcg)

    # ---- 4. Grid sweep --------------------------------------------------
    logger.info("=" * 70)
    logger.info("weighted score ensemble grid sweep (per-user min-max norm):")
    logger.info("%-25s %-12s %-12s %-12s", "config", "NDCG@10", "recall@10", "Δ vs best")

    results = []
    fused_parquets = {}
    t0 = time.time()
    for cfg_name, weights_dict in CONFIGS:
        active = {k: v for k, v in weights_dict.items() if k in preds and v > 0}
        if not active:
            logger.warning("  %s: no active models, SKIP", cfg_name)
            continue
        if len(active) == 1:
            # 단일 모델 = baseline (정규화 안 해도 ranking 동일)
            fused = list(preds.values())[list(preds.keys()).index(list(active.keys())[0])]
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
        logger.info("  %-25s %.6f     %.6f     %+.6f",
                    cfg_name, n, r, delta)
        fused_parquets[cfg_name] = fused
    logger.info("grid sweep done in %.1fs", time.time() - t0)

    # ---- 5. Best + write ------------------------------------------------
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

    # ---- 6. Submission CSV (best only, only if lift) -------------------
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
