"""ensemble_v7_clean_3model / run.py -- RRF on 3 well-calibrated models only.

Hypothesis (from teammate-comparison):
    외부 reference 0.1441 ensemble = TiSASRec + BSARec + TIFU-KNN + MBSTR (4 well-calibrated)
    우리 v3-v6 dead-end = TIFU + BSARec + MB-STR + BSARec_CL + BERT4Rec
                          (2 LOO-overfit 모델이 ensemble 오염)

v7 = TIFU(MB) + BSARec + MB-STR (BSARec_CL, BERT4Rec 제거)
    → ensemble 이 polluted-by-noise 였던 게 맞다면 v7 self-val > single TIFU

Models (3, all well-calibrated):
    tifu_mb:  exp_007b_tifu_mb            -- self-val 0.2933 (multi-behavior)
    bsarec:   exp_002e_bsarec_4w_full     -- self-val 0.2470 (4w full)
    mbstr:    exp_009_mbstr               -- self-val 0.2389 (simplified MB-STR)

Configs swept (tight):
    tifu_only             (baseline)
    1:1:1                 (equal — external reference 패턴 추정)
    3:1:1, 5:1:1          (TIFU-dominant)
    3:1:2, 3:2:1          (assess MB-STR vs BSARec contribution)
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

import pandas as pd  # noqa: E402

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


MODELS = {
    "tifu_mb": PROJECT_ROOT / "experiments" / "exp_007b_tifu_mb" / "predictions.parquet",
    "bsarec":  PROJECT_ROOT / "experiments" / "exp_002e_bsarec_4w_full" / "predictions.parquet",
    "mbstr":   PROJECT_ROOT / "experiments" / "exp_009_mbstr" / "predictions.parquet",
}

CONFIGS: list[tuple[str, dict]] = [
    ("tifu_only",   {"tifu_mb": 1.0}),
    ("equal_1_1_1", {"tifu_mb": 1.0, "bsarec": 1.0, "mbstr": 1.0}),
    ("t3_b1_m1",    {"tifu_mb": 3.0, "bsarec": 1.0, "mbstr": 1.0}),
    ("t5_b1_m1",    {"tifu_mb": 5.0, "bsarec": 1.0, "mbstr": 1.0}),
    ("t3_b1_m2",    {"tifu_mb": 3.0, "bsarec": 1.0, "mbstr": 2.0}),
    ("t3_b2_m1",    {"tifu_mb": 3.0, "bsarec": 2.0, "mbstr": 1.0}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-const", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--items-per-user", type=int, default=10)
    parser.add_argument("--train-data", default="/root/data/train.parquet")
    parser.add_argument("--sample-submission", default="/root/data/sample_submission.csv")
    parser.add_argument("--ease-saved",
                        default=str(PROJECT_ROOT / "experiments" / "exp_001_ease" / "saved"))
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--gt-event-types", nargs="+", default=["purchase"])
    parser.add_argument("--no-submission", dest="make_submission", action="store_false")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-entity", default="yooni0125-")
    parser.add_argument("--wandb-project", default="cy-commerce-recsys")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("loading predictions ...")
    preds: dict[str, pd.DataFrame] = {}
    for name, path in MODELS.items():
        if not path.exists():
            logger.warning("  %s MISSING (%s) -- SKIP", name, path)
            continue
        df = pd.read_parquet(path)
        preds[name] = df
        logger.info("  %-10s %s rows, %s users",
                    name, f"{len(df):,}", f"{df['user_id'].nunique():,}")

    ease_saved = Path(args.ease_saved)
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    logger.info("val_gt %s rows, %s eval_users",
                f"{len(val_gt_eval):,}", f"{len(eval_users):,}")

    logger.info("baseline single-model self-val:")
    single_scores = {}
    for name, df in preds.items():
        n = ndcg_at_k_from_df(df, val_gt_eval, k=10)
        r = recall_at_k_from_df(df, val_gt_eval, k=10)
        single_scores[name] = {"ndcg10": n, "recall10": r}
        logger.info("  %-10s NDCG@10=%.6f  recall@10=%.6f", name, n, r)

    best_single_name = max(single_scores, key=lambda k: single_scores[k]["ndcg10"])
    best_single_ndcg = single_scores[best_single_name]["ndcg10"]
    logger.info("best single: %s (NDCG=%.6f)", best_single_name, best_single_ndcg)

    logger.info("=" * 60)
    logger.info("RRF grid sweep (k_const=%d, top_n=%d):", args.k_const, args.top_n)
    logger.info("%-18s %-12s %-12s %-12s", "config", "NDCG@10", "recall@10", "Δ vs best single")

    results = []
    fused_parquets = {}
    t0 = time.time()
    for cfg_name, weights_dict in CONFIGS:
        active = {k: v for k, v in weights_dict.items() if k in preds}
        if not active:
            logger.warning("  %s: no available models, SKIP", cfg_name)
            continue
        dfs = [preds[k] for k in active]
        ws = [active[k] for k in active]
        if len(dfs) == 1:
            fused = dfs[0]
        else:
            fused = rrf_combine(dfs, k_const=args.k_const, top_n=args.top_n, weights=ws)

        n = ndcg_at_k_from_df(fused, val_gt_eval, k=10)
        r = recall_at_k_from_df(fused, val_gt_eval, k=10)
        delta = n - best_single_ndcg
        results.append({
            "config": cfg_name,
            "ndcg10": n,
            "recall10": r,
            "delta_vs_best_single": delta,
            "weights": ws,
            "models": list(active.keys()),
        })
        logger.info("  %-18s %.6f     %.6f     %+.6f", cfg_name, n, r, delta)
        fused_parquets[cfg_name] = fused

    logger.info("grid sweep done in %.1fs", time.time() - t0)

    results.sort(key=lambda x: x["ndcg10"], reverse=True)
    best = results[0]
    logger.info("=" * 60)
    logger.info("BEST: %s   NDCG@10=%.6f   recall@10=%.6f   Δ %+.6f",
                best["config"], best["ndcg10"], best["recall10"], best["delta_vs_best_single"])

    best_parquet = HERE / f"fused_best_{best['config']}.parquet"
    fused_parquets[best["config"]].to_parquet(best_parquet)
    logger.info("wrote %s", best_parquet)

    results_dump = {
        "single_scores": single_scores,
        "configs": [
            {
                "name": r["config"],
                "models": r["models"],
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
            "models": best["models"],
            "weights": best["weights"],
        },
        "k_const": args.k_const,
        "top_n": args.top_n,
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

    if args.use_wandb:
        try:
            import wandb

            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=f"ensemble_v7_{best['config']}",
                config={
                    "k_const": args.k_const,
                    "top_n": args.top_n,
                    "models_available": list(preds.keys()),
                    "configs_swept": [r["config"] for r in results],
                },
            )
            wandb_log = {f"single_{k}_ndcg": v["ndcg10"] for k, v in single_scores.items()}
            for r in results:
                wandb_log[f"fused_{r['config']}_ndcg"] = r["ndcg10"]
                wandb_log[f"fused_{r['config']}_recall"] = r["recall10"]
                wandb_log[f"fused_{r['config']}_delta"] = r["delta_vs_best_single"]
            wandb_log["best_config"] = best["config"]
            wandb_log["best_ndcg"] = best["ndcg10"]
            wandb_log["best_delta"] = best["delta_vs_best_single"]
            wandb.log(wandb_log)
            run.finish()
        except ImportError:
            logger.warning("wandb not installed; skipping")

    logger.info("run.py done")


if __name__ == "__main__":
    main()
