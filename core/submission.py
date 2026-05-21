"""Submission helpers for the commerce recommendation task.

Pipeline:
    raw model scores  --save_predictions-->   predictions.parquet (top-50, ranked)
    predictions.parquet --predictions_to_submission-->   output.csv (top-10, 6.4M rows)
    output.csv  --validate_submission-->  bool

The predictions.parquet is the standard ensemble input: top-50 candidates
per user with raw model scores, so downstream stages (rerank, ensemble)
can re-mix without re-running the model.

The output.csv is the file uploaded to the competition portal:
    columns: user_id, item_id (no header? -> there IS a header per CLAUDE.md)
    exactly 6,382,570 rows = 638,257 users x 10 items
    no duplicate item within a user
    cold-start / under-filled users padded from popularity_fallback
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_predictions(
    scores: np.ndarray,
    user_ids: list,
    mappings: dict,
    output_path: str,
    top_n: int = 50,
) -> None:
    """Extract per-user top-N items from a dense scores matrix and persist.

    Output parquet schema:
        user_id : object  (taken verbatim from `user_ids`)
        item_id : object  (decoded via mappings['idx2item'])
        score   : float64 (raw model score, copied through)
        rank    : int32   (1..top_n, within-user)

    Args:
        scores: (n_users, n_items) float ndarray. scores[i, j] is the
            predicted affinity of user_ids[i] for the item with column
            index j. Must not contain NaN/inf (argpartition behavior is
            undefined). Row i corresponds to user_ids[i]; column j decodes
            via mappings['idx2item'][j].
        user_ids: List of length n_users. Whatever values appear here are
            written to the parquet's user_id column verbatim -- typically
            the original UUID strings. The caller pre-decodes from int
            indices via mappings['idx2user'] if needed.
        mappings: Output of `data_loader.build_id_mappings`. Must contain
            'idx2item' OR 'item2idx' (inverse derived on the fly).
        output_path: Destination .parquet path. Parent directory created
            if missing.
        top_n: Number of items to keep per user (default 50). Clamped to
            n_items.

    Notes:
        Memory: builds a (n_users, top_n) index array and a same-sized
        score array. For 638k users x top_n=50 that's ~250MB float64.
        The input `scores` itself can be much larger -- the caller is
        responsible for batching if it doesn't fit memory.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    n_users, n_items = scores.shape
    if len(user_ids) != n_users:
        raise ValueError(
            f"len(user_ids)={len(user_ids)} != scores.shape[0]={n_users}"
        )
    top_n = min(top_n, n_items)
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    if "idx2item" in mappings:
        idx2item = mappings["idx2item"]
    elif "item2idx" in mappings:
        idx2item = {v: k for k, v in mappings["item2idx"].items()}
    else:
        raise ValueError("mappings must contain idx2item or item2idx")

    # Build a positional item_id lookup array so we can numpy-fancy-index
    # without 32M Python dict lookups.
    item_id_array = np.empty(n_items, dtype=object)
    for j in range(n_items):
        item_id_array[j] = idx2item[j]

    # Partial top-N per row: argpartition is O(n_items) per row, much
    # faster than full sort when top_n << n_items.
    # We negate scores so 'kth=top_n-1' picks the top-N largest.
    neg = -scores
    part_idx = np.argpartition(neg, top_n - 1, axis=1)[:, :top_n]
    part_neg = np.take_along_axis(neg, part_idx, axis=1)
    sort_order = np.argsort(part_neg, axis=1, kind="stable")
    top_idx = np.take_along_axis(part_idx, sort_order, axis=1)
    top_scores = np.take_along_axis(scores, top_idx, axis=1)

    # Flatten to long format.
    user_ids_repeat = np.repeat(np.asarray(user_ids, dtype=object), top_n)
    item_idx_flat = top_idx.reshape(-1)
    item_ids_flat = item_id_array[item_idx_flat]
    scores_flat = top_scores.reshape(-1)
    ranks = np.tile(np.arange(1, top_n + 1, dtype=np.int32), n_users)

    out = pd.DataFrame({
        "user_id": user_ids_repeat,
        "item_id": item_ids_flat,
        "score": scores_flat.astype(np.float64, copy=False),
        "rank": ranks,
    })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)

    logger.info(
        "save_predictions: wrote %s (%s rows = %s users x top_%d)",
        out_path,
        f"{len(out):,}",
        f"{n_users:,}",
        top_n,
    )


def compute_popularity(train_df: pd.DataFrame, top_n: int = 50) -> list[str]:
    """Top-N items by purchase frequency in `train_df`.

    Used as a cold-start fallback in `predictions_to_submission`.

    Args:
        train_df: Long-format event log with 'item_id' and 'event_type'.
        top_n: Number of items to return (default 50).

    Returns:
        List of item_id values, most-purchased first.
    """
    for col in ("item_id", "event_type"):
        if col not in train_df.columns:
            raise ValueError(f"train_df missing column: {col}")

    purchases = train_df.loc[train_df["event_type"] == "purchase", "item_id"]
    counts = purchases.value_counts().head(top_n)
    items = counts.index.tolist()
    logger.info(
        "compute_popularity: top_%d items computed from %s purchase events",
        top_n,
        f"{len(purchases):,}",
    )
    return items


def predictions_to_submission(
    pred_path: str,
    output_csv: str,
    all_users: Sequence,
    mappings: dict,
    popularity_fallback: Sequence,
    items_per_user: int = 10,
) -> None:
    """Convert ranked predictions into the competition CSV.

    Steps per user (in `all_users` order):
        1. Pull predictions for the user (sorted by rank ascending if
           present, else by score descending).
        2. Dedup items, keep the first occurrence (highest-ranked).
        3. Keep the first `items_per_user` after dedup.
        4. If short, pad with items from `popularity_fallback`, skipping
           anything already picked. Cold-start users (no predictions) are
           filled entirely from the fallback.

    Args:
        pred_path: Path to a predictions parquet produced by
            `save_predictions` (or any file with user_id, item_id, and
            rank/score columns).
        output_csv: Destination CSV path (will be overwritten).
        all_users: The full list of users the submission must cover (e.g.
            638,257 for this competition). Output rows follow this order.
        mappings: Accepted for API symmetry but currently unused -- pred
            parquet already stores decoded UUIDs. Kept in the signature
            so future callers can pass it without code changes.
        popularity_fallback: Item IDs ordered by descending popularity.
            Must contain at least `items_per_user` unique entries.
        items_per_user: How many items per user in the submission
            (default 10, the competition requirement).
    """
    del mappings  # reserved for forward-compat; pred parquet is pre-decoded

    pred = pd.read_parquet(pred_path)
    for col in ("user_id", "item_id"):
        if col not in pred.columns:
            raise ValueError(f"pred parquet missing column: {col}")
    if "rank" in pred.columns:
        pred = pred.sort_values(["user_id", "rank"], ascending=[True, True])
    elif "score" in pred.columns:
        pred = pred.sort_values(["user_id", "score"], ascending=[True, False])
    # Defensive dedup -- save_predictions doesn't emit duplicates, but
    # ensembled inputs can.
    pred = pred.drop_duplicates(["user_id", "item_id"], keep="first")

    pred_by_user = pred.groupby("user_id", sort=False)["item_id"].apply(list).to_dict()

    pop_list = list(dict.fromkeys(popularity_fallback))  # dedup preserving order
    if len(pop_list) < items_per_user:
        raise ValueError(
            f"popularity_fallback has {len(pop_list)} unique items; "
            f"need at least items_per_user={items_per_user}"
        )

    flat_users: list = []
    flat_items: list = []
    n_cold_start = 0
    n_padded = 0

    for uid in all_users:
        items = pred_by_user.get(uid)
        if items is None:
            items = []
            n_cold_start += 1

        seen: set = set()
        picked: list = []
        for it in items:
            if it not in seen:
                seen.add(it)
                picked.append(it)
                if len(picked) == items_per_user:
                    break

        if len(picked) < items_per_user:
            if items:  # had some predictions but had to pad
                n_padded += 1
            for it in pop_list:
                if len(picked) == items_per_user:
                    break
                if it not in seen:
                    seen.add(it)
                    picked.append(it)

        if len(picked) < items_per_user:
            raise ValueError(
                f"user {uid!r}: only {len(picked)}/{items_per_user} items "
                "even after popularity fallback. Increase popularity_fallback size."
            )

        flat_users.extend([uid] * items_per_user)
        flat_items.extend(picked)

    out = pd.DataFrame({"user_id": flat_users, "item_id": flat_items})
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info(
        "predictions_to_submission: wrote %s (%s rows, %s users, "
        "cold-start filled=%s, padded=%s)",
        out_path,
        f"{len(out):,}",
        f"{len(all_users):,}",
        f"{n_cold_start:,}",
        f"{n_padded:,}",
    )


def validate_submission(
    csv_path: str,
    expected_users: int = 638_257,
    items_per_user: int = 10,
) -> bool:
    """Check that `csv_path` conforms to the submission spec.

    Args:
        csv_path: Path to the submission CSV.
        expected_users: Expected unique user count (default 638,257 =
            the competition's full user set).
        items_per_user: Required item count per user (default 10).

    Returns:
        True if every check passes, else False (with reasons printed).
    """
    expected_rows = expected_users * items_per_user
    sub = pd.read_csv(csv_path)

    ok = True

    if list(sub.columns) != ["user_id", "item_id"]:
        print(f"FAIL columns: got {list(sub.columns)}, expected ['user_id', 'item_id']")
        ok = False

    if sub.shape != (expected_rows, 2):
        print(f"FAIL shape: got {sub.shape}, expected ({expected_rows}, 2)")
        ok = False

    n_users = sub["user_id"].nunique()
    if n_users != expected_users:
        print(f"FAIL user count: got {n_users:,}, expected {expected_users:,}")
        ok = False

    per_user = sub.groupby("user_id").size()
    bad_count = per_user[per_user != items_per_user]
    if len(bad_count):
        sample = bad_count.head(5).to_dict()
        print(
            f"FAIL per-user count: {len(bad_count):,} users not at "
            f"{items_per_user} items. Sample: {sample}"
        )
        ok = False

    dup_per_user = sub.groupby("user_id")["item_id"].agg(lambda s: s.duplicated().any())
    n_dup = int(dup_per_user.sum())
    if n_dup:
        offenders = dup_per_user[dup_per_user].head(5).index.tolist()
        print(f"FAIL duplicates: {n_dup:,} users have duplicate items. Sample: {offenders}")
        ok = False

    if ok:
        print(
            f"OK: shape={sub.shape}, users={n_users:,}, "
            f"{items_per_user}/user, no duplicates"
        )
    return ok


if __name__ == "__main__":
    import shutil
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="submission_test_"))
    print(f"\n=== using tmpdir: {tmpdir} ===")

    try:
        # ---- Setup: 5 users (4 with predictions, 1 cold-start) x 8 items
        all_users = [f"u-{i}" for i in range(5)]   # u-0..u-4; u-4 is cold-start
        items = [f"item-{i}" for i in range(8)]    # item-0..item-7
        # Forward+inverse mappings
        item2idx = {it: i for i, it in enumerate(items)}
        idx2item = {i: it for it, i in item2idx.items()}
        mappings = {"item2idx": item2idx, "idx2item": idx2item}

        # ---- compute_popularity from a fake train_df --------------------
        print("\n=== compute_popularity ===")
        train_rows = []
        # Purchase distribution: item-0 (most) > item-1 > ... > item-7 (least)
        for j, item in enumerate(items):
            n_purchases = 8 - j  # 8, 7, ..., 1
            for _ in range(n_purchases):
                train_rows.append(("buyer", item, "purchase"))
        # Add some views/carts that should NOT count
        for item in items:
            train_rows.append(("viewer", item, "view"))
        train_df = pd.DataFrame(train_rows, columns=["user_id", "item_id", "event_type"])
        popularity = compute_popularity(train_df, top_n=8)
        print(f"popularity (top-8): {popularity}")
        assert popularity == items, f"expected items in order, got {popularity}"

        # ---- save_predictions -------------------------------------------
        print("\n=== save_predictions ===")
        # 4 users with predictions, deliberately rigged so user u-i prefers item-i highest.
        rng = np.random.default_rng(0)
        n_users_pred = 4
        scores = rng.random((n_users_pred, 8))
        for i in range(n_users_pred):
            scores[i, i] = 100.0          # rank 1 for each user
            scores[i, (i + 1) % 8] = 50.0  # rank 2
        pred_path = tmpdir / "predictions.parquet"
        save_predictions(
            scores=scores,
            user_ids=all_users[:n_users_pred],  # u-0..u-3
            mappings=mappings,
            output_path=str(pred_path),
            top_n=5,
        )
        pred_df = pd.read_parquet(pred_path)
        print(pred_df.head(10).to_string(index=False))
        assert list(pred_df.columns) == ["user_id", "item_id", "score", "rank"]
        assert pred_df.shape == (n_users_pred * 5, 4)
        # Rank 1 for u-0 must be item-0 (rigged)
        u0_rank1 = pred_df[(pred_df["user_id"] == "u-0") & (pred_df["rank"] == 1)]
        assert u0_rank1["item_id"].iloc[0] == "item-0", f"got {u0_rank1['item_id'].iloc[0]}"
        # Rank 2 for u-0 must be item-1
        u0_rank2 = pred_df[(pred_df["user_id"] == "u-0") & (pred_df["rank"] == 2)]
        assert u0_rank2["item_id"].iloc[0] == "item-1"
        # rank values are 1..5 per user, monotonically increasing
        for uid in all_users[:n_users_pred]:
            user_pred = pred_df[pred_df["user_id"] == uid].sort_values("rank")
            assert list(user_pred["rank"]) == [1, 2, 3, 4, 5]
            # score is descending within user
            assert (user_pred["score"].diff().dropna() <= 0).all(), f"scores not descending for {uid}"

        # ---- predictions_to_submission ----------------------------------
        print("\n=== predictions_to_submission (with cold-start user u-4) ===")
        sub_path = tmpdir / "submission.csv"
        # popularity has only 8 items; we need >= 10 for the real comp default,
        # so use items_per_user=5 for this toy run.
        predictions_to_submission(
            pred_path=str(pred_path),
            output_csv=str(sub_path),
            all_users=all_users,
            mappings=mappings,
            popularity_fallback=popularity,
            items_per_user=5,
        )
        sub = pd.read_csv(sub_path)
        print(sub.to_string(index=False))
        assert sub.shape == (5 * 5, 2)
        # u-4 (cold-start) should be filled entirely from popularity (first 5)
        u4 = sub[sub["user_id"] == "u-4"]["item_id"].tolist()
        assert u4 == popularity[:5], f"u-4 should be popularity[:5], got {u4}"
        # u-0 rank-1 item should be item-0 (rigged)
        assert sub[sub["user_id"] == "u-0"]["item_id"].iloc[0] == "item-0"
        # Output order follows all_users order
        assert sub["user_id"].tolist()[:5] == ["u-0"] * 5

        # ---- validate_submission OK case --------------------------------
        print("\n=== validate_submission (OK) ===")
        ok = validate_submission(str(sub_path), expected_users=5, items_per_user=5)
        assert ok is True

        # ---- validate_submission failure modes --------------------------
        print("\n=== validate_submission FAIL: wrong shape ===")
        bad1 = sub_path.with_name("bad_shape.csv")
        sub.head(20).to_csv(bad1, index=False)
        assert validate_submission(str(bad1), expected_users=5, items_per_user=5) is False

        print("\n=== validate_submission FAIL: duplicate item per user ===")
        bad2 = sub_path.with_name("bad_dup.csv")
        sub_dup = sub.copy()
        # Force u-0 to have item-0 twice (replace rank 2 with item-0)
        u0_mask = sub_dup["user_id"] == "u-0"
        first_idx = sub_dup[u0_mask].index[0]
        second_idx = sub_dup[u0_mask].index[1]
        sub_dup.loc[second_idx, "item_id"] = sub_dup.loc[first_idx, "item_id"]
        sub_dup.to_csv(bad2, index=False)
        assert validate_submission(str(bad2), expected_users=5, items_per_user=5) is False

        print("\n=== validate_submission FAIL: wrong per-user count ===")
        bad3 = sub_path.with_name("bad_count.csv")
        # Drop one row -> u-0 has 4 items instead of 5
        sub.drop(sub.index[0]).to_csv(bad3, index=False)
        assert validate_submission(str(bad3), expected_users=5, items_per_user=5) is False

        # ---- popularity_fallback too short raises -----------------------
        print("\n=== popularity_fallback too short raises ===")
        try:
            predictions_to_submission(
                pred_path=str(pred_path),
                output_csv=str(tmpdir / "should_not_exist.csv"),
                all_users=all_users,
                mappings=mappings,
                popularity_fallback=items[:3],  # only 3 items, need 5
                items_per_user=5,
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            print(f"correctly raised: {e}")

        # ---- Padding case: user has 1 pred, needs 5; fills from popularity
        print("\n=== padding case ===")
        # Build a tiny prediction parquet where u-0 has only 1 unique item
        tiny_pred = pd.DataFrame({
            "user_id": ["u-0"],
            "item_id": ["item-5"],
            "score": [10.0],
            "rank": [1],
        })
        tiny_path = tmpdir / "tiny.parquet"
        tiny_pred.to_parquet(tiny_path)
        sub_pad_path = tmpdir / "sub_pad.csv"
        predictions_to_submission(
            pred_path=str(tiny_path),
            output_csv=str(sub_pad_path),
            all_users=["u-0"],
            mappings=mappings,
            popularity_fallback=popularity,
            items_per_user=5,
        )
        sub_pad = pd.read_csv(sub_pad_path)
        u0_items = sub_pad["item_id"].tolist()
        print(f"u-0 items: {u0_items}")
        # First is the predicted item-5; rest are popularity (excluding item-5 already taken)
        assert u0_items[0] == "item-5"
        # Remaining filled from popularity[:5] = item-0..item-4 (none overlap with item-5)
        assert u0_items[1:] == popularity[:4]

        # ---- ensemble dedup case ----------------------------------------
        print("\n=== ensemble dedup case ===")
        dup_pred = pd.DataFrame({
            "user_id": ["u-0", "u-0", "u-0", "u-0", "u-0", "u-0"],
            "item_id": ["item-0", "item-0", "item-1", "item-2", "item-3", "item-4"],
            "score": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            "rank": [1, 2, 3, 4, 5, 6],
        })
        dup_path = tmpdir / "dup.parquet"
        dup_pred.to_parquet(dup_path)
        sub_dedup_path = tmpdir / "sub_dedup.csv"
        predictions_to_submission(
            pred_path=str(dup_path),
            output_csv=str(sub_dedup_path),
            all_users=["u-0"],
            mappings=mappings,
            popularity_fallback=popularity,
            items_per_user=5,
        )
        sub_dedup = pd.read_csv(sub_dedup_path)
        items_out = sub_dedup["item_id"].tolist()
        print(f"u-0 dedup items: {items_out}")
        # Duplicate item-0 should appear only once, then item-1..item-4
        assert items_out == ["item-0", "item-1", "item-2", "item-3", "item-4"]

        # ---- Final output format showcase (matches baseline ALS format) -
        print("\n=== final submission.csv format ===")
        with open(sub_path) as f:
            head_lines = [next(f) for _ in range(6)]
        print("".join(head_lines).rstrip())

        print("\nAll sanity checks passed.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"cleaned up {tmpdir}")
