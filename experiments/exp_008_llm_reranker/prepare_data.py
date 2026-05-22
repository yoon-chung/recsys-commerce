"""exp_008_llm_reranker / prepare_data.py -- build item metadata + user history caches.

LLM 에 던질 prompt 빌드에 필요한 데이터를 한 번 만들고 parquet 으로 저장.
rerank_poc.py 가 이 캐시들을 로드해 prompt 빌드 시 빠르게 조회.

Outputs (./cache/):
    item_metadata.parquet
        columns: item_id, category, brand, avg_price, total_events
        per item 의 대표 category_code, brand, average price, train 등장 빈도
    user_history.parquet
        columns: user_id, history_text
        per user 최근 N events 의 text 표현
        (e.g. "[purchase] electronics.smartphone / brand=Samsung / $700 (2d ago)")

Usage:
    python prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from core.data_loader import load_train_data  # noqa: E402
from core.validation import time_based_split  # noqa: E402

logger = logging.getLogger(__name__)


def build_item_metadata(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-item: most common category, brand, average price, total events."""
    logger.info("building item metadata ...")
    # category mode
    cat_mode = (
        train_df.dropna(subset=["category_code"])
        .groupby("item_id")["category_code"]
        .agg(lambda s: s.mode().iat[0] if not s.empty else None)
    )
    # brand mode
    brand_mode = (
        train_df.dropna(subset=["brand"])
        .groupby("item_id")["brand"]
        .agg(lambda s: s.mode().iat[0] if not s.empty else None)
    )
    price_mean = train_df.groupby("item_id")["price"].mean()
    total = train_df.groupby("item_id").size()

    meta = pd.DataFrame({
        "category": cat_mode,
        "brand": brand_mode,
        "avg_price": price_mean,
        "total_events": total,
    }).reset_index()
    meta = meta.rename(columns={"index": "item_id"})
    logger.info("  %s items with metadata", f"{len(meta):,}")
    return meta


def build_user_history(
    train_df: pd.DataFrame,
    meta: pd.DataFrame,
    n_recent: int,
) -> pd.DataFrame:
    """Per user 최근 n_recent events 의 text 표현.

    'history_text' 형식:
        [view] electronics.smartphone / Samsung / $700 (3h ago)
        [cart] computers.notebook / Lenovo / $1200 (1d ago)
        [purchase] electronics.smartphone / Samsung / $720 (2d ago)
    """
    logger.info("building user history (last %d events per user)...", n_recent)
    meta_lookup = meta.set_index("item_id")[["category", "brand", "avg_price"]]

    # event_time -> datetime + timestamp seconds
    df = train_df.copy()
    df["ts"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
    cutoff = df["ts"].max()
    df["days_ago"] = (cutoff - df["ts"]).dt.total_seconds() / 86400.0

    # 정렬 desc by ts, take last n_recent per user
    df = df.sort_values(["user_id", "ts"], ascending=[True, False], kind="mergesort")
    df["rn"] = df.groupby("user_id").cumcount()
    df = df[df["rn"] < n_recent]

    # merge metadata
    df = df.merge(meta_lookup, left_on="item_id", right_index=True, how="left")

    def _fmt_event(row) -> str:
        cat = row.get("category") or "unknown"
        brand = row.get("brand") or "unknown"
        price = row.get("price")
        price_str = f"${price:.0f}" if pd.notna(price) else "$?"
        days = row.get("days_ago", 0)
        if days < 1:
            time_str = f"{int(days * 24)}h ago"
        else:
            time_str = f"{int(days)}d ago"
        return f"[{row['event_type']}] {cat} / {brand} / {price_str} ({time_str})"

    logger.info("  formatting events as text...")
    df["event_text"] = df.apply(_fmt_event, axis=1)

    # 최근 -> 과거 순서 (LLM 이 자연스럽게 읽도록 reverse)
    df = df.sort_values(["user_id", "ts"], kind="mergesort")  # oldest first

    history = df.groupby("user_id")["event_text"].apply(
        lambda s: "\n".join(s.tolist())
    ).reset_index().rename(columns={"event_text": "history_text"})

    logger.info("  %s users with history", f"{len(history):,}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--cache-dir", default=str(HERE / "cache"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + split ----------------------------------------------
    df_full = load_train_data(cfg["train_data"])
    train_df, _ = time_based_split(
        df_full,
        val_days=cfg["val_days"],
        gt_event_types=cfg["gt_event_types"],
    )
    logger.info("train_df after split: %s rows", f"{len(train_df):,}")

    # ---- Build caches -----------------------------------------------
    meta = build_item_metadata(train_df)
    history = build_user_history(train_df, meta, n_recent=cfg["n_recent_events"])

    # ---- Write -----------------------------------------------------
    meta_path = cache_dir / "item_metadata.parquet"
    hist_path = cache_dir / "user_history.parquet"
    meta.to_parquet(meta_path)
    history.to_parquet(hist_path)
    logger.info("wrote %s + %s", meta_path, hist_path)

    # quick stats
    avg_hist_len = history["history_text"].str.len().mean()
    avg_hist_lines = history["history_text"].str.count("\n").mean() + 1
    logger.info("avg history: %.0f chars, %.1f lines per user",
                avg_hist_len, avg_hist_lines)


if __name__ == "__main__":
    main()
