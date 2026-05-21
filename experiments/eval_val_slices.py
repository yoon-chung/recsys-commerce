"""experiments/eval_val_slices.py -- retroactive sub-val NDCG analysis.

For any model that already produced top-N predictions.parquet, compute
NDCG@k and recall@k on three val sub-windows:

    full       : Feb 23-29 (current self-val baseline)
    no_spike   : Feb 23-26 (excludes the Feb 27-29 purchase spike)
    spike_only : Feb 27-29 (the spike days)

`no_spike` is the metric we expect to align best with public NDCG (Mar 1-7)
since it removes the anomalous spike volume from val. Comparing
`full` vs `no_spike` quantifies spike-driven inflation of self-val.

Reads the shared val_gt + eval_users from exp_001_ease/saved/ and applies
the same eval_users filter every other inference.py uses, so numbers are
directly comparable.

Usage (server):
    cd /root/workspace/recsys-commerce/experiments
    python eval_val_slices.py \
        --pred ./exp_000_als_baseline/predictions.parquet \
        --pred ./exp_001_ease/predictions.parquet \
        --pred ./exp_002_bsarec/predictions.parquet
    python eval_val_slices.py --pred ./exp_002b_bsarec_4w/predictions.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from core.validation import build_val_slices  # noqa: E402
from core.metrics import evaluate_val_slices  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="path to a predictions.parquet (repeatable for batch eval)",
    )
    parser.add_argument(
        "--val-gt",
        default=str(HERE / "exp_001_ease" / "saved" / "val_gt.parquet"),
        help="shared val_gt parquet (default: reuse exp_001_ease)",
    )
    parser.add_argument(
        "--eval-users",
        default=str(HERE / "exp_001_ease" / "saved" / "eval_users.json"),
        help="shared eval_users json (default: reuse exp_001_ease)",
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- Load shared val_gt + eval_users ---------------------------------
    val_gt_df = pd.read_parquet(args.val_gt)
    with open(args.eval_users, encoding="utf-8") as f:
        eval_users = set(json.load(f))

    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    logger.info(
        "loaded val_gt %s rows (%s eval_users -> %s rows after filter)",
        f"{len(val_gt_df):,}",
        f"{len(eval_users):,}",
        f"{len(val_gt_eval):,}",
    )

    # ---- Build sub-val slices --------------------------------------------
    logger.info("building default val sub-windows (full / no_spike / spike_only):")
    slices = build_val_slices(val_gt_eval)

    # ---- Evaluate each predictions file ----------------------------------
    rows = []
    for pred_path in args.pred:
        logger.info("evaluating %s", pred_path)
        pred_df = pd.read_parquet(pred_path)
        results = evaluate_val_slices(pred_df, slices, k=args.k)
        for name, r in results.items():
            rows.append(
                {
                    "pred": pred_path,
                    "slice": name,
                    "n_rows": r["n_rows"],
                    "n_users": r["n_users"],
                    f"ndcg@{args.k}": r["ndcg"],
                    f"recall@{args.k}": r["recall"],
                }
            )

    if not rows:
        logger.warning("no results")
        return

    df = pd.DataFrame(rows)
    # Order rows by (pred, slice) for readability.
    slice_order = {n: i for i, n in enumerate(["full", "no_spike", "spike_only"])}
    df["_o"] = df["slice"].map(slice_order).fillna(99)
    df = df.sort_values(["pred", "_o"]).drop(columns="_o").reset_index(drop=True)

    print("\n=== sub-val NDCG / recall ===")
    # Strip the long ./expX path prefix from each pred for compact display.
    df["pred_short"] = df["pred"].apply(lambda p: Path(p).parent.name or p)
    cols = ["pred_short", "slice", "n_users", "n_rows",
            f"ndcg@{args.k}", f"recall@{args.k}"]
    with pd.option_context(
        "display.float_format", "{:.6f}".format,
        "display.max_colwidth", 60,
    ):
        print(df[cols].to_string(index=False))

    # ---- Spike inflation summary (per pred) ------------------------------
    print("\n=== full vs no_spike (spike inflation) ===")
    summary_rows = []
    for pred_path, sub in df.groupby("pred"):
        sub = sub.set_index("slice")
        if "full" in sub.index and "no_spike" in sub.index:
            full = sub.at["full", f"ndcg@{args.k}"]
            nospk = sub.at["no_spike", f"ndcg@{args.k}"]
            ratio = full / nospk if nospk else float("nan")
            summary_rows.append(
                {
                    "pred": Path(pred_path).parent.name or pred_path,
                    f"full_ndcg@{args.k}": full,
                    f"no_spike_ndcg@{args.k}": nospk,
                    "ratio_full_over_no_spike": ratio,
                    "delta": full - nospk,
                }
            )
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        with pd.option_context("display.float_format", "{:.6f}".format):
            print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
