"""exp_002b_bsarec_4w / data_prep.py -- build 4-week RecBole atomic dataset.

Same as exp_002_bsarec/data_prep.py except `--last-days` defaults to 28
(last 4 weeks, Feb 1-29), and output goes to ./data/cy_commerce_4w/.

This file is gitignored via the global `**/data/` pattern.

Usage:
    python data_prep.py                       # default: last 28 days
    python data_prep.py --last-days 14        # narrower ablation
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
from core.validation import time_based_split  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="/root/data/train.parquet")
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument(
        "--gt-event-types",
        nargs="+",
        default=["purchase"],
        help="passed to time_based_split (only affects which rows count as val GT, "
             "not what goes into the training file -- everything before cutoff is train)",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        default=28,
        help="keep only the last N days of train_df (4-week recency window). "
             "default 28 = Feb 1-29.",
    )
    parser.add_argument("--out-dir", default=str(HERE / "data" / "cy_commerce_4w"))
    parser.add_argument("--dataset-name", default="cy_commerce_4w")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_data(args.train_data)
    train_df, _ = time_based_split(
        df,
        val_days=args.val_days,
        gt_event_types=args.gt_event_types,
    )

    # event_time is string -- convert to UNIX seconds (float).
    train_df = train_df.assign(
        timestamp=pd.to_datetime(
            train_df["event_time"], format="%Y-%m-%d %H:%M:%S %Z"
        ).astype("int64") // 10**9
    )

    # Optional recency-window filter (4-week ablation etc.)
    if args.last_days is not None:
        cutoff_ts = int(train_df["timestamp"].max()) - args.last_days * 24 * 3600
        before = len(train_df)
        train_df = train_df[train_df["timestamp"] >= cutoff_ts]
        logger.info(
            "last_days=%d filter: %s -> %s rows (kept %.1f%%)",
            args.last_days,
            f"{before:,}",
            f"{len(train_df):,}",
            100.0 * len(train_df) / before,
        )

    # RecBole atomic-file column convention: <name>:<dtype>
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
    logger.info(
        "file size: %.1f MB",
        out_path.stat().st_size / 1024**2,
    )
    logger.info("done. update config.yaml: data_path=%s, dataset=%s",
                out_dir.parent, args.dataset_name)


if __name__ == "__main__":
    main()
