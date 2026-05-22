"""exp_009_mbstr / data_prep.py -- atomic file with behavior_id column.

기본 exp_002 와 동일하지만 behavior_id (view=1, cart=2, purchase=3) 추가.

Produces ./data/cy_commerce_mb/cy_commerce_mb.inter, TSV with columns:
    user_id:token    item_id:token    timestamp:float    behavior_id:token

behavior_id 가 sequential 학습 시 RecBole 가 behavior_id_list 로 변환
(load_col 에 포함 + token type).

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

BEHAVIOR_MAP = {"view": 1, "cart": 2, "purchase": 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="/root/data/train.parquet")
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--gt-event-types", nargs="+", default=["purchase"])
    parser.add_argument(
        "--last-days",
        type=int,
        default=None,
        help="(optional) last N days. default None = 4m full. MB-STR 은 4m default.",
    )
    parser.add_argument("--out-dir", default=str(HERE / "data" / "cy_commerce_mb"))
    parser.add_argument("--dataset-name", default="cy_commerce_mb")
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
        ).astype("int64") // 10**9,
        behavior_id=train_df["event_type"].map(BEHAVIOR_MAP),
    )
    train_df = train_df.dropna(subset=["behavior_id"])
    train_df["behavior_id"] = train_df["behavior_id"].astype("int64")

    if args.last_days is not None:
        cutoff_ts = int(train_df["timestamp"].max()) - args.last_days * 24 * 3600
        before = len(train_df)
        train_df = train_df[train_df["timestamp"] >= cutoff_ts]
        logger.info(
            "last_days=%d filter: %s -> %s rows",
            args.last_days, f"{before:,}", f"{len(train_df):,}",
        )

    out = train_df[["user_id", "item_id", "timestamp", "behavior_id"]].rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
            "behavior_id": "behavior_id:token",
        }
    )
    out_path = out_dir / f"{args.dataset_name}.inter"
    out.to_csv(out_path, sep="\t", index=False)

    logger.info(
        "wrote %s -- %s rows, %s users, %s items",
        out_path, f"{len(out):,}",
        f"{out['user_id:token'].nunique():,}",
        f"{out['item_id:token'].nunique():,}",
    )
    logger.info("behavior_id 분포: %s", train_df["behavior_id"].value_counts().to_dict())
    logger.info("file size: %.1f MB", out_path.stat().st_size / 1024**2)


if __name__ == "__main__":
    main()
