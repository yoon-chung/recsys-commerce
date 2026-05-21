"""Ensemble utilities for combining predictions from multiple models.

Currently provides:
    - rrf_combine: Reciprocal Rank Fusion (Cormack et al., 2009)

RRF is rank-based, so it does NOT require model scores to be calibrated to
each other. This is the right default fusion for heterogeneous models
(EASE item-item vs ALS factor vs sequential SASRec etc.).

    RRF_score(u, i) = sum_{m in models} 1 / (k_const + rank_m(u, i))

where rank_m(u, i) is the rank of item i for user u under model m, and
k_const dampens the contribution of low-ranked items (paper default: 60).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def rrf_combine(
    pred_dfs: Sequence[pd.DataFrame],
    k_const: int = 60,
    top_n: int = 50,
) -> pd.DataFrame:
    """Reciprocal Rank Fusion across multiple ranked-list models.

    Args:
        pred_dfs: One DataFrame per model. Each must have columns
            (user_id, item_id, rank), where `rank` is 1-indexed and lower
            is better. Models may rank different items per user; missing
            items contribute 0 to the fused score.
        k_const: RRF constant. Larger -> ranks matter less. Default 60.
        top_n: Number of items to keep per user in the output.

    Returns:
        DataFrame with columns (user_id, item_id, score, rank), sorted by
        (user_id asc, rank asc), exactly top_n rows per user that has any
        predictions across the input DataFrames. `score` is the RRF score
        and `rank` is the fused rank (1..top_n).
    """
    if not pred_dfs:
        raise ValueError("rrf_combine: pred_dfs is empty")
    for i, df in enumerate(pred_dfs):
        for col in ("user_id", "item_id", "rank"):
            if col not in df.columns:
                raise ValueError(f"pred_dfs[{i}] missing column: {col}")

    parts = []
    for df in pred_dfs:
        parts.append(
            pd.DataFrame(
                {
                    "user_id": df["user_id"].to_numpy(),
                    "item_id": df["item_id"].to_numpy(),
                    "_rrf": 1.0 / (k_const + df["rank"].to_numpy(dtype=np.float64)),
                }
            )
        )
    long = pd.concat(parts, ignore_index=True)

    fused = (
        long.groupby(["user_id", "item_id"], sort=False)["_rrf"]
        .sum()
        .reset_index(name="score")
    )
    fused["rank"] = (
        fused.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    fused = fused[fused["rank"] <= top_n].copy()
    fused = fused.sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)
    return fused


if __name__ == "__main__":
    # ---- Toy example ------------------------------------------------------
    # Model A ranks: [X, Y, Z]  -> ranks 1, 2, 3
    # Model B ranks: [Y, X, W]  -> ranks 1, 2, 3
    # k_const = 60
    #   X: 1/61 + 1/62 = 0.03253...
    #   Y: 1/62 + 1/61 = 0.03253...  (tied with X)
    #   Z: 1/63        = 0.01587...
    #   W: 1/63        = 0.01587...  (tied with Z)
    df_a = pd.DataFrame(
        [("u", "X", 1), ("u", "Y", 2), ("u", "Z", 3)],
        columns=["user_id", "item_id", "rank"],
    )
    df_b = pd.DataFrame(
        [("u", "Y", 1), ("u", "X", 2), ("u", "W", 3)],
        columns=["user_id", "item_id", "rank"],
    )
    fused = rrf_combine([df_a, df_b], k_const=60, top_n=4)
    print(fused)
    assert len(fused) == 4
    # X and Y tie, Z and W tie. method='first' breaks ties by appearance.
    print("\nrrf_combine sanity check passed.")
