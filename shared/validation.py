"""Train/validation splits for the commerce purchase task.

The competition evaluates next-week purchases (2020-03-01..2020-03-07).
For local validation we hold out the last `val_days` of train.parquet
and treat in-window purchases as ground truth -- mirroring the official
eval distribution.

Two strategies are provided:
    - `time_based_split`: global time cutoff (most faithful to the task).
    - `leave_one_out_split`: per-user last purchase held out (useful for
      sequential models like SASRec).

Typical usage:
    df = pd.read_parquet("/root/data/train.parquet")
    train_df, val_gt_df = time_based_split(df, val_days=7)
    eval_users = get_eval_users(val_gt_df, train_df)
    # train model on train_df, predict for eval_users, then:
    from shared.metrics import ndcg_at_k_from_df
    score = ndcg_at_k_from_df(pred_df, val_gt_df, k=10)

No caching is performed by design -- splits are cheap and caching invites
stale-split bugs.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

EVENT_TIME_FMT = "%Y-%m-%d %H:%M:%S %Z"


def _ensure_datetime(s: pd.Series) -> pd.Series:
    """Return a tz-naive datetime64[ns] view of `s`.

    Accepts:
      - already-datetime columns (tz-aware or naive)
      - strings in the competition format '%Y-%m-%d %H:%M:%S %Z'

    Always returns naive UTC (tz stripped) so direct comparisons are safe.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s
    parsed = pd.to_datetime(s, format=EVENT_TIME_FMT, utc=True, errors="raise")
    return parsed.dt.tz_localize(None)


def time_based_split(
    df: pd.DataFrame,
    val_days: int = 7,
    gt_event_types: list = ["purchase"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` by global time cutoff.

    cutoff = df['event_time'].max() - val_days
        train_df: events with event_time <= cutoff (all event types kept)
        val_gt_df: events with event_time > cutoff AND event_type in
                   `gt_event_types` (default: ['purchase'])

    Args:
        df: Long-format event log with columns 'user_id', 'item_id',
            'event_time', 'event_type'. `event_time` may be string in the
            competition format or already-parsed datetime.
        val_days: Number of trailing days held out for validation (default 7,
            matching the official 1-week eval window).
        gt_event_types: Which event types count as relevant in val_gt_df.
            Default ['purchase'] matches the competition metric.

    Returns:
        (train_df, val_gt_df). val_gt_df is already filtered to the
        relevant event types and is directly consumable by
        `shared.metrics.ndcg_at_k_from_df` as the `gt_df` argument.

    Raises:
        ValueError: missing required columns or val_days <= 0.
    """
    for col in ("user_id", "item_id", "event_time", "event_type"):
        if col not in df.columns:
            raise ValueError(f"df missing required column: {col}")
    if val_days <= 0:
        raise ValueError(f"val_days must be positive, got {val_days}")

    times = _ensure_datetime(df["event_time"])
    cutoff = times.max() - pd.Timedelta(days=val_days)
    train_mask = times <= cutoff
    val_mask = ~train_mask

    train_df = df.loc[train_mask].copy()
    val_raw = df.loc[val_mask]
    val_gt_df = val_raw[val_raw["event_type"].isin(gt_event_types)].copy()

    logger.info(
        "time_based_split: cutoff=%s, val_days=%d",
        cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        val_days,
    )
    logger.info(
        "  train     : %s rows, %s users",
        f"{len(train_df):,}",
        f"{train_df['user_id'].nunique():,}",
    )
    logger.info(
        "  val (raw) : %s rows, %s users",
        f"{len(val_raw):,}",
        f"{val_raw['user_id'].nunique():,}",
    )
    logger.info(
        "  val_gt    : %s rows, %s users (event_types=%s)",
        f"{len(val_gt_df):,}",
        f"{val_gt_df['user_id'].nunique():,}",
        gt_event_types,
    )
    return train_df, val_gt_df


def leave_one_out_split(
    df: pd.DataFrame,
    gt_event_types: list = ["purchase"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's last event in `gt_event_types` for validation.

    For every user that has at least one event matching `gt_event_types`,
    the most recent such event (by `event_time`) is moved into val_gt_df.
    All other events (including earlier purchases by the same user) remain
    in train_df.

    Useful for sequential models (SASRec, BERT4Rec) where the task is to
    predict the next interaction given the prior sequence.

    Args:
        df: Long-format event log.
        gt_event_types: Event types eligible for hold-out (default ['purchase']).

    Returns:
        (train_df, val_gt_df). val_gt_df has exactly one row per user that
        had a qualifying event.
    """
    for col in ("user_id", "item_id", "event_time", "event_type"):
        if col not in df.columns:
            raise ValueError(f"df missing required column: {col}")

    times = _ensure_datetime(df["event_time"])
    target_mask = df["event_type"].isin(gt_event_types)
    if not target_mask.any():
        logger.info("leave_one_out_split: no rows match gt_event_types=%s", gt_event_types)
        return df.copy(), df.iloc[:0].copy()

    target_idx = df.index[target_mask]
    target_times = times.loc[target_idx]
    # idxmax over time within each user -> index of that user's last target event.
    last_idx = (
        pd.DataFrame({"user_id": df.loc[target_idx, "user_id"], "_t": target_times})
        .groupby("user_id")["_t"]
        .idxmax()
        .values
    )

    val_gt_df = df.loc[last_idx].copy()
    train_df = df.drop(last_idx).copy()

    logger.info(
        "leave_one_out_split: train=%s rows, val_gt=%s rows (%s users, event_types=%s)",
        f"{len(train_df):,}",
        f"{len(val_gt_df):,}",
        f"{val_gt_df['user_id'].nunique():,}",
        gt_event_types,
    )
    return train_df, val_gt_df


def get_eval_users(val_gt_df: pd.DataFrame, train_df: pd.DataFrame) -> set:
    """Return the set of users that should be evaluated.

    A user is evaluated only if they (a) have at least one ground-truth
    event in val_gt_df and (b) also appear in train_df. Cold-start users
    that exist only in the val window are excluded because collaborative
    models cannot produce informed predictions for them. This matches the
    official eval, which guarantees no cold-start IDs.

    Args:
        val_gt_df: Validation ground-truth events.
        train_df: Train portion.

    Returns:
        set of user_id values to evaluate.
    """
    for col_df, name in ((val_gt_df, "val_gt_df"), (train_df, "train_df")):
        if "user_id" not in col_df.columns:
            raise ValueError(f"{name} missing column: user_id")

    val_users = set(val_gt_df["user_id"].unique())
    train_users = set(train_df["user_id"].unique())
    eval_users = val_users & train_users
    logger.info(
        "get_eval_users: val=%s, train=%s, eval (intersect)=%s, dropped (cold-start)=%s",
        f"{len(val_users):,}",
        f"{len(train_users):,}",
        f"{len(eval_users):,}",
        f"{len(val_users - train_users):,}",
    )
    return eval_users


if __name__ == "__main__":
    import math

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- Toy event log spanning days 0..9 from 2020-02-20 ----------------
    # Day 0-3 -> train portion with val_days=3.
    # Day 7-9 -> val portion.
    base = pd.Timestamp("2020-02-20 00:00:00")
    rows = [
        ("u1", "A", "view",     base + pd.Timedelta(days=0)),
        ("u2", "B", "purchase", base + pd.Timedelta(days=1)),
        ("u1", "B", "view",     base + pd.Timedelta(days=2)),
        ("u3", "B", "view",     base + pd.Timedelta(days=3)),
        ("u3", "B", "purchase", base + pd.Timedelta(days=7)),   # val gt
        ("u1", "A", "purchase", base + pd.Timedelta(days=8)),   # val gt
        ("u4", "D", "purchase", base + pd.Timedelta(days=8)),   # val gt, cold-start user
        ("u3", "C", "view",     base + pd.Timedelta(days=9)),   # val raw only (not purchase)
    ]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "event_type", "event_time"])

    print("=== time_based_split(val_days=3) ===")
    train_df, val_gt_df = time_based_split(df, val_days=3)
    print(f"train rows: {len(train_df)}, val_gt rows: {len(val_gt_df)}")
    print(f"val_gt:\n{val_gt_df[['user_id', 'item_id', 'event_type']].to_string(index=False)}")
    assert len(train_df) == 4, f"expected 4 train rows, got {len(train_df)}"
    # val_gt = 3 purchase rows in val window (u3/B, u1/A, u4/D)
    assert len(val_gt_df) == 3, f"expected 3 val_gt rows, got {len(val_gt_df)}"
    assert set(val_gt_df["item_id"]) == {"A", "B", "D"}
    # u3/C view on day 9 is in val window but not a purchase -> excluded from val_gt
    assert "C" not in val_gt_df["item_id"].values

    print("\n=== get_eval_users ===")
    eval_users = get_eval_users(val_gt_df, train_df)
    print(f"eval_users: {sorted(eval_users)}")
    # u4 is cold-start (no events in train_df) -> excluded
    assert eval_users == {"u1", "u3"}, f"unexpected eval_users: {eval_users}"

    print("\n=== gt_event_types=['purchase', 'cart'] ===")
    df_cart = df.copy()
    df_cart.loc[len(df_cart)] = ("u1", "B", "cart", base + pd.Timedelta(days=8))
    _, val_gt_pc = time_based_split(df_cart, val_days=3, gt_event_types=["purchase", "cart"])
    print(f"val_gt with cart:\n{val_gt_pc[['user_id', 'item_id', 'event_type']].to_string(index=False)}")
    # u1 now has both A (purchase) and B (cart) in val window
    u1_items = set(val_gt_pc[val_gt_pc["user_id"] == "u1"]["item_id"])
    assert u1_items == {"A", "B"}, f"u1 should have {{A,B}}: {u1_items}"

    print("\n=== string event_time parsing ===")
    df_str = df.copy()
    df_str["event_time"] = df_str["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    print(f"event_time dtype before: {df_str['event_time'].dtype}")
    train_s, val_gt_s = time_based_split(df_str, val_days=3)
    assert len(train_s) == 4 and len(val_gt_s) == 3
    print("string parsing OK")

    print("\n=== leave_one_out_split ===")
    train_loo, val_gt_loo = leave_one_out_split(df)
    print(f"train rows: {len(train_loo)}, val_gt rows: {len(val_gt_loo)}")
    print(f"val_gt:\n{val_gt_loo[['user_id', 'item_id', 'event_type', 'event_time']].sort_values('user_id').to_string(index=False)}")
    # Each user with at least 1 purchase contributes exactly 1 val row.
    # u1's purchases: only A/day8 -> held out
    # u2: only B/day1 -> held out
    # u3: only B/day7 -> held out
    # u4: only D/day8 -> held out
    assert len(val_gt_loo) == 4, f"expected 4 val_gt rows, got {len(val_gt_loo)}"
    assert set(val_gt_loo["user_id"]) == {"u1", "u2", "u3", "u4"}
    # train_df keeps non-purchase events
    assert len(train_loo) == len(df) - len(val_gt_loo)
    # u1 had A view, B view in train_df (both views remain since only purchase was held out)
    assert {("u1", "A", "view"), ("u1", "B", "view")}.issubset(
        set(train_loo[["user_id", "item_id", "event_type"]].apply(tuple, axis=1))
    )

    print("\n=== leave_one_out: user with multiple purchases keeps earlier ones ===")
    df_multi = df.copy()
    # Add an earlier purchase for u1: A on day 4 (should stay in train, day 8 still held out)
    df_multi.loc[len(df_multi)] = ("u1", "A", "purchase", base + pd.Timedelta(days=4))
    train_m, val_gt_m = leave_one_out_split(df_multi)
    u1_train_purchases = train_m[(train_m["user_id"] == "u1") & (train_m["event_type"] == "purchase")]
    u1_val = val_gt_m[val_gt_m["user_id"] == "u1"]
    print(f"u1 train purchases: {len(u1_train_purchases)} (day 4 should remain)")
    print(f"u1 val: {len(u1_val)} (day 8, the latest)")
    assert len(u1_train_purchases) == 1
    assert len(u1_val) == 1
    assert u1_val["event_time"].iloc[0] == base + pd.Timedelta(days=8)

    print("\n=== integration with shared.metrics.ndcg_at_k_from_df ===")
    from metrics import ndcg_at_k_from_df

    train_df, val_gt_df = time_based_split(df, val_days=3)
    eval_users = get_eval_users(val_gt_df, train_df)
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    # Construct fake predictions: u1 gets A at rank 1, u3 gets B at rank 2.
    pred_df = pd.DataFrame(
        [
            ("u1", "A", 1), ("u1", "X", 2), ("u1", "Y", 3),
            ("u3", "X", 1), ("u3", "B", 2), ("u3", "Y", 3),
        ],
        columns=["user_id", "item_id", "rank"],
    )
    score = ndcg_at_k_from_df(pred_df, val_gt_eval, k=10)
    expected = (1.0 + 1.0 / math.log2(3)) / 2
    print(f"NDCG@10: {score:.6f} (expected {expected:.6f})")
    assert abs(score - expected) < 1e-9

    print("\nAll sanity checks passed.")
