"""docs/eda_spike_mechanism.py -- spike (Feb 27-29) mechanism 검증.

가설 후보:
  A. 판촉/단발성 이벤트 (cart 폭증 + 가격 하락)
  B. 데이터셋 cutoff artifact (마지막 3일이라 cart→purchase lag 부족)
  C. 신규 user 유입 (cold-start)
  D. 시즌성 (특정 category/brand)

검증:
  1. 일별 cart count (cart 자체가 폭증했나)
  2. 일별 평균 purchase price (할인 패턴)
  3. First-view → purchase 시간 (impulse buy)
  4. Spike 시 신규 user 비율
  5. Spike 시 top brand / category 분포

실행: python docs/eda_spike_mechanism.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRAIN_PATH = Path(__file__).resolve().parents[1] / "baseline" / "data" / "train.parquet"
if not TRAIN_PATH.exists():
    TRAIN_PATH = Path("/root/data/train.parquet")

SPIKE_START = pd.Timestamp("2020-02-27", tz="UTC")


def main() -> None:
    print(f"loading {TRAIN_PATH} ...")
    df = pd.read_parquet(TRAIN_PATH)
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
    df["date"] = df["event_time"].dt.date
    print(f"  total events: {len(df):,}")
    print(f"  date range: {df['event_time'].min()} ~ {df['event_time'].max()}\n")

    # ---- 1. 일별 event count by type (cart 도 spike?) -------------------
    print("=== [1] 일별 event count (last 20 days) ===")
    daily = df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
    daily["total"] = daily.sum(axis=1)
    print(daily.tail(20).to_string())
    print()

    # ---- 2. 일별 평균 purchase price (할인?) ----------------------------
    print("=== [2] 일별 평균 purchase price (last 20 days) ===")
    p = df[df["event_type"] == "purchase"]
    daily_price = p.groupby("date").agg(
        n_purch=("price", "size"),
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        min_price=("price", "min"),
        max_price=("price", "max"),
    )
    print(daily_price.tail(20).to_string(float_format=lambda x: f"{x:,.2f}"))
    print()

    # ---- 3. First-view → purchase 시간 (impulse buy?) -------------------
    print("=== [3] First-view → purchase 시간 (분 단위) ===")
    first_view = (
        df[df["event_type"] == "view"]
        .groupby(["user_id", "item_id"])["event_time"].min()
        .reset_index()
        .rename(columns={"event_time": "first_view"})
    )
    purchases = df[df["event_type"] == "purchase"][["user_id", "item_id", "event_time"]].rename(
        columns={"event_time": "purchase_time"}
    )
    m = purchases.merge(first_view, on=["user_id", "item_id"], how="left")
    m["minutes_to_purchase"] = (m["purchase_time"] - m["first_view"]).dt.total_seconds() / 60
    m["is_spike"] = m["purchase_time"] >= SPIKE_START
    m["period"] = m["is_spike"].map({True: "spike", False: "pre-spike"})

    by_period = m.groupby("period")["minutes_to_purchase"].agg(
        n=("size"),
        mean=("mean"),
        median=("median"),
        p25=(lambda s: s.quantile(0.25)),
        p75=(lambda s: s.quantile(0.75)),
    )
    print(by_period.to_string(float_format=lambda x: f"{x:,.1f}"))
    print()

    # ---- 4. 신규 user 비율 (cold-start) --------------------------------
    print("=== [4] 신규 user 비율 (해당 일에 처음 등장한 user) ===")
    user_first = df.groupby("user_id")["event_time"].min().dt.date.rename("first_date")
    df_with_first = df.merge(user_first, on="user_id")
    df_with_first["is_new_user"] = df_with_first["date"] == df_with_first["first_date"]

    daily_new = (
        df_with_first[df_with_first["event_type"] == "purchase"]
        .groupby("date")
        .agg(n_purch=("user_id", "size"), n_new_user=("is_new_user", "sum"))
    )
    daily_new["new_user_rate"] = daily_new["n_new_user"] / daily_new["n_purch"]
    print(daily_new.tail(20).to_string(float_format=lambda x: f"{x:,.4f}" if x < 1 else f"{x:,.0f}"))
    print()

    # ---- 5. Spike vs pre-spike top brand --------------------------------
    print("=== [5] Top brand (purchase 기준) ===")
    p_sp = p[p["event_time"] >= SPIKE_START]
    p_pre = p[p["event_time"] < SPIKE_START]
    top_sp = p_sp["brand"].value_counts(dropna=False).head(10)
    top_pre = p_pre["brand"].value_counts(dropna=False).head(10)
    print("\n-- Pre-spike top-10 --")
    print(top_pre.to_string())
    print("\n-- Spike top-10 --")
    print(top_sp.to_string())
    print()

    # ---- 6. Spike vs pre-spike top category ----------------------------
    print("=== [6] Top category_code (purchase 기준) ===")
    cat_sp = p_sp["category_code"].value_counts(dropna=False).head(10)
    cat_pre = p_pre["category_code"].value_counts(dropna=False).head(10)
    print("\n-- Pre-spike top-10 --")
    print(cat_pre.to_string())
    print("\n-- Spike top-10 --")
    print(cat_sp.to_string())


if __name__ == "__main__":
    main()
