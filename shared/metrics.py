"""Evaluation metrics for the commerce recommendation task.

The primary metric is binary-relevance NDCG@10, the official competition
metric. recall@k is included as a debugging companion to inspect candidate-
generation quality independently of ranking.

Both metrics are vectorized via numpy so they scale to ~640k users in a
few seconds on a single CPU core.

Conventions
-----------
- `predicted` is a list of per-user ranked item lists. `predicted[i]` is
  position-aligned with `list(ground_truth.keys())[i]`. The caller is
  responsible for filtering out cold-start users before calling, so that
  `len(predicted) == len(ground_truth)`.
- Each inner list of `predicted` must be sorted by score descending; ties
  broken arbitrarily. Lists shorter than `k` are zero-padded internally.
- `ground_truth` maps user_id -> set of relevant (purchased) item_ids.
  Users whose set is empty are excluded from the averaged score.
"""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd


def _discounts(k: int) -> np.ndarray:
    """Return 1 / log2(rank + 1) for rank = 1..k, shape (k,)."""
    return 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))


def ndcg_at_k(
    predicted: Sequence[Sequence[Hashable]],
    ground_truth: Mapping[Hashable, set],
    k: int = 10,
) -> float:
    """Mean binary-relevance NDCG@k.

    Formula:
        DCG@k  = sum_{i=1..k} rel_i / log2(i + 1),   rel_i in {0, 1}
        IDCG@k = sum_{i=1..min(k, |gt|)} 1 / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

    Args:
        predicted: One ranked list of item_ids per user, position-aligned
            with `list(ground_truth.keys())`. Inner lists shorter than k
            are zero-padded; longer lists are truncated.
        ground_truth: {user_id: set(item_ids)} of relevant items.
            Users with an empty set are excluded from the mean.
        k: Cutoff rank (default 10).

    Returns:
        Mean NDCG@k over users with non-empty ground truth.
        Returns 0.0 if no user has any relevant item.

    Raises:
        ValueError: if k <= 0 or len(predicted) != len(ground_truth).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"len(predicted)={len(predicted)} != len(ground_truth)={len(ground_truth)}; "
            "predicted must be position-aligned with ground_truth.keys()"
        )

    user_ids = list(ground_truth.keys())
    n_users = len(user_ids)
    discounts = _discounts(k)

    rel = np.zeros((n_users, k), dtype=np.float64)
    gt_sizes = np.empty(n_users, dtype=np.int64)
    for i, uid in enumerate(user_ids):
        gt = ground_truth[uid]
        gt_sizes[i] = len(gt)
        preds_i = predicted[i]
        row = [1.0 if p in gt else 0.0 for p in preds_i[:k]]
        if row:
            rel[i, : len(row)] = row

    dcg = rel @ discounts

    # IDCG via cumulative discounts: cum[i] = sum of first (i+1) discounts.
    cum = np.cumsum(discounts)
    capped = np.minimum(gt_sizes, k)
    idcg = np.zeros(n_users, dtype=np.float64)
    nonzero = capped > 0
    idcg[nonzero] = cum[capped[nonzero] - 1]

    mask = gt_sizes > 0
    if not mask.any():
        return 0.0

    ndcg = np.zeros(n_users, dtype=np.float64)
    ndcg[mask] = dcg[mask] / idcg[mask]
    return float(ndcg[mask].mean())


def recall_at_k(
    predicted: Sequence[Sequence[Hashable]],
    ground_truth: Mapping[Hashable, set],
    k: int = 10,
) -> float:
    """Mean recall@k = |top-k ∩ gt| / |gt|.

    Useful as a debugging companion to NDCG: high recall + low NDCG means
    candidates are present but mis-ranked; low recall means the model
    fails at candidate generation.

    Duplicates in `predicted` are collapsed via set intersection, so a
    user predicting `[A, A, B]` against gt `{A}` gets recall 1.0, not 2.0.

    Args:
        predicted: See `ndcg_at_k`.
        ground_truth: See `ndcg_at_k`.
        k: Cutoff rank (default 10).

    Returns:
        Mean recall@k over users with non-empty ground truth.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"len(predicted)={len(predicted)} != len(ground_truth)={len(ground_truth)}"
        )

    user_ids = list(ground_truth.keys())
    n_users = len(user_ids)
    hits = np.zeros(n_users, dtype=np.float64)
    gt_sizes = np.empty(n_users, dtype=np.int64)
    for i, uid in enumerate(user_ids):
        gt = ground_truth[uid]
        gt_sizes[i] = len(gt)
        topk = predicted[i][:k]
        hits[i] = len(set(topk) & gt)

    mask = gt_sizes > 0
    if not mask.any():
        return 0.0

    return float((hits[mask] / gt_sizes[mask]).mean())


def _df_to_lists(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> tuple[list[list], dict]:
    """Convert long-format DataFrames into (predicted_list, ground_truth_dict).

    Sorting rule for pred_df:
        - if `rank` column present: sort by (user_id asc, rank asc)
        - elif `score` column present: sort by (user_id asc, score desc)
        - else: assume already sorted within each user group.

    Only users appearing in gt_df survive; predicted users absent from
    gt_df are treated as cold-start and dropped (matches the contract of
    `ndcg_at_k`).
    """
    for col in ("user_id", "item_id"):
        if col not in pred_df.columns:
            raise ValueError(f"pred_df missing column: {col}")
        if col not in gt_df.columns:
            raise ValueError(f"gt_df missing column: {col}")

    if "rank" in pred_df.columns:
        pred_sorted = pred_df.sort_values(["user_id", "rank"], ascending=[True, True])
    elif "score" in pred_df.columns:
        pred_sorted = pred_df.sort_values(["user_id", "score"], ascending=[True, False])
    else:
        pred_sorted = pred_df

    ground_truth: dict = (
        gt_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    pred_grouped = pred_sorted.groupby("user_id", sort=False)["item_id"].apply(list)
    predicted = [pred_grouped.get(uid, []) for uid in ground_truth.keys()]
    return predicted, ground_truth


def ndcg_at_k_from_df(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    k: int = 10,
) -> float:
    """DataFrame wrapper around `ndcg_at_k`. See that function for semantics.

    Args:
        pred_df: columns must include user_id, item_id, and one of
            (rank ascending, score descending). If neither sort key is
            present, rows are assumed pre-sorted within each user group.
        gt_df: columns user_id, item_id (one row per relevant (u, i) pair).
        k: Cutoff rank (default 10).

    Returns:
        Mean NDCG@k.
    """
    predicted, ground_truth = _df_to_lists(pred_df, gt_df)
    return ndcg_at_k(predicted, ground_truth, k=k)


def recall_at_k_from_df(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    k: int = 10,
) -> float:
    """DataFrame wrapper around `recall_at_k`. See `ndcg_at_k_from_df`."""
    predicted, ground_truth = _df_to_lists(pred_df, gt_df)
    return recall_at_k(predicted, ground_truth, k=k)


if __name__ == "__main__":
    import math

    # ---- Toy example (matches CLAUDE.md spec) -----------------------------
    # 3 users, GT: [{A}, {B}, {A, C}], pred: [A, B, C] for all three.
    ground_truth = {"u1": {"A"}, "u2": {"B"}, "u3": {"A", "C"}}
    predicted = [
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["A", "B", "C"],
    ]

    # Hand-computed NDCG@10:
    # u1: rel=[1,0,0,...]  DCG = 1/log2(2)              = 1.0
    #                       IDCG = 1/log2(2)             = 1.0
    #                       NDCG = 1.0
    # u2: rel=[0,1,0,...]  DCG = 1/log2(3)
    #                       IDCG = 1/log2(2)             = 1.0
    #                       NDCG = 1 / log2(3)
    # u3: rel=[1,0,1,...]  DCG = 1/log2(2) + 1/log2(4)  = 1.5
    #                       IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    #                       NDCG = 1.5 / (1 + 1/log2(3))
    per_user_expected = [
        1.0,
        1.0 / math.log2(3),
        1.5 / (1.0 + 1.0 / math.log2(3)),
    ]
    expected_ndcg = sum(per_user_expected) / 3
    actual_ndcg = ndcg_at_k(predicted, ground_truth, k=10)

    print("=== NDCG@10 toy example ===")
    print(f"per-user expected : {[round(v, 6) for v in per_user_expected]}")
    print(f"mean    expected  : {expected_ndcg:.6f}")
    print(f"mean    actual    : {actual_ndcg:.6f}")
    assert abs(actual_ndcg - expected_ndcg) < 1e-9, "NDCG mismatch"

    # ---- Recall@10 --------------------------------------------------------
    # All 3 users have all gt items inside the top-3 predictions -> recall=1.0 each.
    actual_recall = recall_at_k(predicted, ground_truth, k=10)
    print("\n=== recall@10 toy example ===")
    print(f"mean expected: 1.000000")
    print(f"mean actual  : {actual_recall:.6f}")
    assert abs(actual_recall - 1.0) < 1e-9, "recall mismatch"

    # ---- Cold-start exclusion --------------------------------------------
    # Adding a user with empty GT should NOT change the mean.
    gt_with_empty = {**ground_truth, "u4": set()}
    pred_with_empty = predicted + [["X", "Y", "Z"]]
    ndcg_with_empty = ndcg_at_k(pred_with_empty, gt_with_empty, k=10)
    print("\n=== cold-start exclusion ===")
    print(f"NDCG@10 with empty-gt user appended: {ndcg_with_empty:.6f}")
    print(f"  (matches base case: {abs(ndcg_with_empty - expected_ndcg) < 1e-9})")
    assert abs(ndcg_with_empty - expected_ndcg) < 1e-9

    # ---- DataFrame wrapper -----------------------------------------------
    pred_df = pd.DataFrame(
        [
            ("u1", "A", 1), ("u1", "B", 2), ("u1", "C", 3),
            ("u2", "A", 1), ("u2", "B", 2), ("u2", "C", 3),
            ("u3", "A", 1), ("u3", "B", 2), ("u3", "C", 3),
        ],
        columns=["user_id", "item_id", "rank"],
    )
    gt_df = pd.DataFrame(
        [("u1", "A"), ("u2", "B"), ("u3", "A"), ("u3", "C")],
        columns=["user_id", "item_id"],
    )
    df_ndcg = ndcg_at_k_from_df(pred_df, gt_df, k=10)
    df_recall = recall_at_k_from_df(pred_df, gt_df, k=10)
    print("\n=== DataFrame wrapper ===")
    print(f"NDCG@10  from df: {df_ndcg:.6f}")
    print(f"recall@10 from df: {df_recall:.6f}")
    assert abs(df_ndcg - expected_ndcg) < 1e-9, "DataFrame NDCG mismatch"
    assert abs(df_recall - 1.0) < 1e-9, "DataFrame recall mismatch"

    # ---- Edge cases ------------------------------------------------------
    # All-perfect ranking with |gt| == k.
    gt_full = {"u": {"A", "B", "C"}}
    pred_full = [["A", "B", "C"]]
    assert abs(ndcg_at_k(pred_full, gt_full, k=3) - 1.0) < 1e-9
    print("\nedge case |gt|==k perfect ranking: NDCG=1.0 OK")

    # Empty predictions -> NDCG = 0 for that user.
    gt_one = {"u": {"A"}}
    pred_empty = [[]]
    assert ndcg_at_k(pred_empty, gt_one, k=10) == 0.0
    print("edge case empty predicted list: NDCG=0 OK")

    print("\nAll sanity checks passed.")
