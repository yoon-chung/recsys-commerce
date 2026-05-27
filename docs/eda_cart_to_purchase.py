"""docs/eda_cart_to_purchase.py -- cart→purchase 변환 비율 EDA.

질문: 마지막 3일 (Feb 27-29) spike 기간 구매 중 cart 거치지 않은 비율이 평균보다 높은가?
가설: 그렇다면 cart_boost: false 가 도움된 이유가 설명됨 (spike 가 cart-bypass dominant).

실행: python docs/eda_cart_to_purchase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TRAIN_PATH = "/root/data/train.parquet"
SPIKE_START = "2020-02-27"  # log.md 의 val_gt 99.7% 가 Feb 27-29
VAL_GT_END = "2020-02-29 23:59:59"


def main() -> None:
    print(f"loading {TRAIN_PATH} ...")
    df = pd.read_parquet(TRAIN_PATH)
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
    print(f"  total events: {len(df):,}")
    print(f"  date range: {df['event_time'].min()} ~ {df['event_time'].max()}")

    # ---- 1. Cart-then-purchase 판정 -------------------------------------
    purchases = df[df["event_type"] == "purchase"][["user_id", "item_id", "event_time"]].copy()
    carts = df[df["event_type"] == "cart"][["user_id", "item_id", "event_time"]].copy()
    carts = carts.rename(columns={"event_time": "cart_time"})

    print(f"\npurchase events: {len(purchases):,}")
    print(f"cart events:     {len(carts):,}")

    # 각 purchase 에 대해, 같은 (user, item) 의 cart 가 그 이전에 있었는지
    m = purchases.merge(carts, on=["user_id", "item_id"], how="left")
    m["cart_before"] = m["cart_time"].notna() & (m["cart_time"] < m["event_time"])

    # 같은 purchase 가 여러 cart 와 join 됐을 수 있음 -> any
    result = (
        m.groupby(["user_id", "item_id", "event_time"])["cart_before"]
        .any()
        .reset_index()
    )
    print(f"unique purchase records: {len(result):,}")

    # ---- 2. 전체 vs spike 기간 비교 -------------------------------------
    result["is_spike"] = result["event_time"] >= pd.Timestamp(SPIKE_START, tz="UTC")
    result["period"] = result["is_spike"].map({True: "spike (Feb 27-29)", False: "pre-spike (Nov-Feb 26)"})

    summary = (
        result.groupby("period")["cart_before"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n_purchases", "sum": "via_cart", "mean": "via_cart_rate"})
    )
    summary["direct_purchase"] = summary["n_purchases"] - summary["via_cart"]
    summary["direct_rate"] = 1 - summary["via_cart_rate"]
    summary = summary[["n_purchases", "via_cart", "via_cart_rate", "direct_purchase", "direct_rate"]]

    print("\n=== Cart-bypass 비율 비교 ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}" if x < 1 else f"{x:,.0f}"))

    overall = result["cart_before"].mean()
    spike_rate = result[result["is_spike"]]["cart_before"].mean()
    pre_rate = result[~result["is_spike"]]["cart_before"].mean()

    print(f"\n전체 cart-경유 비율:  {overall:.4f}")
    print(f"Pre-spike 비율:       {pre_rate:.4f}")
    print(f"Spike 기간 비율:      {spike_rate:.4f}")
    print(f"차이 (spike - pre):    {spike_rate - pre_rate:+.4f}")

    direction = "↓ 낮음" if spike_rate < pre_rate else "↑ 높음"
    print(f"\n결론: spike 기간 cart-경유 비율이 {direction} (가설 {'O' if spike_rate < pre_rate else 'X'})")

    # ---- 3. 일별 trend ---------------------------------------------------
    result["date"] = result["event_time"].dt.date
    daily = (
        result.groupby("date")["cart_before"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n_purch", "sum": "via_cart", "mean": "via_cart_rate"})
        .tail(20)  # 마지막 20일
    )
    print(f"\n=== 일별 trend (last 20 days) ===")
    print(daily.to_string(float_format=lambda x: f"{x:,.4f}" if x < 1 else f"{x:,.0f}"))


if __name__ == "__main__":
    main()
