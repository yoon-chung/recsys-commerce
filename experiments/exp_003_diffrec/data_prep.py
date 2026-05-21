"""exp_003_diffrec / data_prep.py -- build RecBole atomic dataset (full 4 months).

Same as exp_002_bsarec/data_prep.py. DiffRec is a general_recommender so it
operates on the user-item interaction matrix (not sequences); the same .inter
file format works -- RecBole's dataset class handles the difference.

This file is gitignored via the global `**/data/` pattern.

Usage:
    python data_prep.py
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
    )
    parser.add_argument("--out-dir", default=str(HERE / "data" / "cy_commerce"))
    parser.add_argument("--dataset-name", default="cy_commerce")
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

    train_df = train_df.assign(
        timestamp=pd.to_datetime(
            train_df["event_time"], format="%Y-%m-%d %H:%M:%S %Z"
        ).astype("int64") // 10**9
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
    logger.info("file size: %.1f MB", out_path.stat().st_size / 1024**2)


if __name__ == "__main__":
    main()
