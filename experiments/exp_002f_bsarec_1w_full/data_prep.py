"""exp_002f_bsarec_1w_full / data_prep.py -- 1-week training INCLUDING val (spike-only).

Trains BSARec on the last 7 days = Feb 23-29, which is the spike week
itself (99.7% of all val purchases happen here). High-variance experiment:
the model sees ONLY spike-time patterns, then predicts Mar 1-7.

Self-val computed by inference.py is meaningless (train = val period).
Only public NDCG is trusted.

Usage:
    python data_prep.py                       # default: last 7 days (Feb 23-29)
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
        default=7,
        help="keep only the last N days of the full train.parquet (Feb 23-29 by default).",
    )
    parser.add_argument("--out-dir", default=str(HERE / "data" / "cy_commerce_1w_full"))
    parser.add_argument("--dataset-name", default="cy_commerce_1w_full")
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
