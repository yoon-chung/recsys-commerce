"""exp_002e_bsarec_4w_full / data_prep.py -- 4-week training INCLUDING val window.

Unlike exp_002b (which holds out Feb 23-29 as val), this variant trains on
the last 28 days INCLUDING the Feb 27-29 purchase spike that lives in the
val window. Final-submission style: spend everything on public NDCG.

We bypass time_based_split entirely and slice the last N days directly
from the raw df. Self-val computed by inference.py becomes meaningless
(model trained on val period); only public NDCG is trusted.

This file is gitignored via the global `**/data/` pattern.

Usage:
    python data_prep.py                       # default: last 28 days (Feb 1-29)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from core.data_loader import load_train_data  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="/root/data/train.parquet")
    parser.add_argument(
        "--last-days",
        type=int,
        default=28,
        help="keep only the last N days of the full train.parquet (Feb 1-29 by default).",
    )
    parser.add_argument("--out-dir", default=str(HERE / "data" / "cy_commerce_4w_full"))
    parser.add_argument("--dataset-name", default="cy_commerce_4w_full")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_data(args.train_data)
    df = df.assign(
        timestamp=pd.to_datetime(
            df["event_time"], format="%Y-%m-%d %H:%M:%S %Z"
        ).astype("int64") // 10**9
    )

    cutoff_ts = int(df["timestamp"].max()) - args.last_days * 24 * 3600
    before = len(df)
    train_df = df[df["timestamp"] >= cutoff_ts]
    logger.info(
        "no-holdout last_days=%d filter: %s -> %s rows (kept %.1f%%)",
        args.last_days,
        f"{before:,}",
        f"{len(train_df):,}",
        100.0 * len(train_df) / before,
    )

    out = train_df[["user_id", "item_id", "timestamp"]].rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
        }
    )

    out_path = out_dir / f"{args.dataset_name}.inter"
    out.to_csv(out_path, sep="\t", index=False)

    logger.info(
        "wrote %s -- %s rows, %s unique users, %s unique items",
        out_path,
        f"{len(out):,}",
        f"{out['user_id:token'].nunique():,}",
        f"{out['item_id:token'].nunique():,}",
    )


if __name__ == "__main__":
    main()
