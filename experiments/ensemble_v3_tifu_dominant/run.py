"""ensemble_v3_tifu_dominant / run.py -- weighted RRF combine, TIFU 중심.

Context:
    - TIFU-KNN (exp_007) public 0.1175, self-val 0.292 -- dominant winner
    - BSARec 4w_full (exp_002e) public 0.0975, self-val 0.247 -- best transformer
    - BSARec+CL (exp_006) self-val 0.235 -- FFT + contrastive, different mechanism
    - BERT4Rec (exp_005) self-val 0.216 -- bidirectional MLM

ensemble_v1/v2 lesson: dominant + weak 합치면 negative. TIFU 가 BSARec 보다
self-val +0.045 우위 -- v3 도 negative 가능성 높음. 그러나:
    - prediction overlap 36% (A1 diagnosis) -> 64% diverge, signal 다양성 있음
    - TIFU 약점: long-tail items (A4 10k+ bin, BSARec +0.009)
    -> TIFU 가 못 잡는 부분 BSARec 가 보완 가능성

Multi-config grid sweep -> 각 weight 조합 self-val 측정 -> best 만 submission.

Configs swept:
    2-model TIFU + BSARec: 1:1, 2:1, 3:1, 5:1, 10:1
    3-model TIFU + BSARec + BSARec+CL: 3:1:1, 5:1:1
    4-model TIFU + BSARec + BSARec+CL + BERT4Rec: 5:1:1:1

Usage:
    python run.py                    # grid sweep, table 출력, best submission 생성
    python run.py --no-submission    # 평가만, csv 미생성
    python run.py --no-wandb
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


# ---- Model paths (relative to PROJECT_ROOT) ------------------------------
MODELS = {
    "tifu":      PROJECT_ROOT / "experiments" / "exp_007_tifu_knn" / "predictions.parquet",
    "bsarec":    PROJECT_ROOT / "experiments" / "exp_002e_bsarec_4w_full" / "predictions.parquet",
    "bsarec_cl": PROJECT_ROOT / "experiments" / "exp_006_bsarec_cl" / "predictions.parquet",
    "bert4rec":  PROJECT_ROOT / "experiments" / "exp_005_bert4rec" / "predictions.parquet",
}

# ---- Grid of weight configurations ----------------------------------------
# Tuple of (config_name, dict of model_name -> weight). Models not in dict skipped.
CONFIGS: list[tuple[str, dict]] = [
    # 2-model: TIFU + BSARec
    ("tifu_only",     {"tifu": 1.0}),                              # baseline (TIFU 단독)
    ("t1_b1",         {"tifu": 1.0, "bsarec": 1.0}),               # equal -- v1/v2 패턴 확인
    ("t2_b1",         {"tifu": 2.0, "bsarec": 1.0}),
    ("t3_b1",         {"tifu": 3.0, "bsarec": 1.0}),
    ("t5_b1",         {"tifu": 5.0, "bsarec": 1.0}),
    ("t10_b1",        {"tifu": 10.0, "bsarec": 1.0}),
    # 3-model: + BSARec+CL hybrid (different mechanism)
    ("t3_b1_c1",      {"tifu": 3.0, "bsarec": 1.0, "bsarec_cl": 1.0}),
    ("t5_b1_c1",      {"tifu": 5.0, "bsarec": 1.0, "bsarec_cl": 1.0}),
    # 4-model: + BERT4Rec (most diverse, weakest single)
    ("t5_b1_c1_e1",   {"tifu": 5.0, "bsarec": 1.0, "bsarec_cl": 1.0, "bert4rec": 1.0}),
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

    # ---- 1. Load all model predictions ----------------------------------
    logger.info("loading predictions ...")
    preds: dict[str, pd.DataFrame] = {}
    for name, path in MODELS.items():
        if not path.exists():
            logger.warning("  %s: MISSING (%s) -- 이 모델 포함 config 는 SKIP", name, path)
            continue
        df = pd.read_parquet(path)
        preds[name] = df
        logger.info("  %s: %s rows, %s users",
                    name, f"{len(df):,}", f"{df['user_id'].nunique():,}")

    # ---- 2. Shared val_gt + eval_users ----------------------------------
    ease_saved = Path(args.ease_saved)
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    logger.info("val_gt %s rows, %s eval_users",
                f"{len(val_gt_eval):,}", f"{len(eval_users):,}")

    # ---- 3. Baseline single-model self-val (sanity) ---------------------
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

    # ---- 4. Grid sweep ---------------------------------------------------
    logger.info("=" * 60)
    logger.info("RRF grid sweep (k_const=%d, top_n=%d):", args.k_const, args.top_n)
    logger.info("%-20s %-12s %-12s %-12s", "config", "NDCG@10", "recall@10", "Δ vs best single")

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
            fused = dfs[0]   # passthrough for tifu_only baseline
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
        logger.info("  %-20s %.6f     %.6f     %+.6f", cfg_name, n, r, delta)
        fused_parquets[cfg_name] = fused

    logger.info("grid sweep done in %.1fs", time.time() - t0)

    # ---- 5. Pick best + write artifacts ----------------------------------
    results.sort(key=lambda x: x["ndcg10"], reverse=True)
    best = results[0]
    logger.info("=" * 60)
    logger.info("BEST: %s   NDCG@10=%.6f   recall@10=%.6f   Δ %+.6f",
                best["config"], best["ndcg10"], best["recall10"], best["delta_vs_best_single"])

    # Save best fused parquet
    best_parquet = HERE / f"fused_best_{best['config']}.parquet"
    fused_parquets[best["config"]].to_parquet(best_parquet)
    logger.info("wrote %s", best_parquet)

    # Dump all results as JSON for log.md update
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

    # ---- 6. Submission CSV (best only) -----------------------------------
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
        logger.info("validate_submission OK -- %s ready for submission", output_csv)
    elif args.make_submission:
        logger.info("best ensemble <= best single (Δ %+.6f) -- submission SKIP",
                    best["delta_vs_best_single"])

    # ---- 7. wandb --------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb

            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=f"ensemble_v3_{best['config']}",
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
