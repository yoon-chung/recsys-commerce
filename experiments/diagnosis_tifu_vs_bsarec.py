"""Diagnosis: TIFU-KNN vs BSARec -- why classical beats deep on this dataset.

5 analyses on eval_users 928명 (exp_001 EASE 의 val_gt + eval_users 재사용):
  A1. Prediction overlap (TIFU top-10 ∩ BSARec top-10 per user)
  A2. Repeat-ratio bins (user 의 view 반복도가 TIFU 강점인지)
  A3. Sequence length bins (BSARec max_seq=50 cap 영향 진단)
  A4. Item popularity bins (long-tail item hit rate)
  A5. Per-user NDCG delta (TIFU win/lose/tie 비율 + 분포)

Outputs:
  docs/diagnosis/summary.json    -- 모든 숫자 요약
  docs/diagnosis/a*_*.csv        -- per-bin 테이블
  docs/diagnosis/a*_*.png        -- 시각화 (histogram, bar)

Server-runnable -- 로컬 predictions.parquet truncated 이슈로 server 에서 실행.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "docs" / "diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
TIFU_PRED = PROJECT_ROOT / "experiments" / "exp_007_tifu_knn" / "predictions.parquet"
BSAREC_PRED = (
    PROJECT_ROOT / "experiments" / "exp_002e_bsarec_4w_full" / "predictions.parquet"
)
EASE_SAVED = PROJECT_ROOT / "experiments" / "exp_001_ease" / "saved"
TRAIN_PARQUET = Path("/root/data/train.parquet")  # server canonical


# ----------------------------------------------------------------------
# Per-user NDCG@10 (binary relevance)
# ----------------------------------------------------------------------
def per_user_ndcg10(pred_df: pd.DataFrame, gt_map: dict) -> pd.Series:
    """Returns Series of NDCG@10 indexed by user_id (only users present in gt_map)."""
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))  # rank 1..10
    cum = np.cumsum(discounts)

    # keep only top-10 candidates, sort by rank
    pred = pred_df[pred_df["rank"] <= 10].sort_values(["user_id", "rank"])
    grouped = pred.groupby("user_id", sort=False)

    out = {}
    for uid, group in grouped:
        if uid not in gt_map:
            continue
        gt = gt_map[uid]
        if not gt:
            continue
        items = group["item_id"].tolist()
        rel = np.array([1.0 if it in gt else 0.0 for it in items[:10]], dtype=np.float64)
        if len(rel) < 10:
            rel = np.pad(rel, (0, 10 - len(rel)))
        dcg = float(rel @ discounts)
        idcg = float(cum[min(len(gt), 10) - 1])
        out[uid] = dcg / idcg if idcg > 0 else 0.0
    return pd.Series(out)


# ----------------------------------------------------------------------
# Load all data
# ----------------------------------------------------------------------
print("Loading predictions / val_gt / eval_users / train ...")
tifu = pd.read_parquet(TIFU_PRED)
bsarec = pd.read_parquet(BSAREC_PRED)
val_gt = pd.read_parquet(EASE_SAVED / "val_gt.parquet")
with open(EASE_SAVED / "eval_users.json", encoding="utf-8") as f:
    eval_users = set(json.load(f))
train_df = pd.read_parquet(TRAIN_PARQUET)

print(f"  tifu pred:   {len(tifu):,} rows ({tifu['user_id'].nunique():,} users)")
print(f"  bsarec pred: {len(bsarec):,} rows ({bsarec['user_id'].nunique():,} users)")
print(f"  val_gt:      {len(val_gt):,} rows ({val_gt['user_id'].nunique():,} users)")
print(f"  eval_users:  {len(eval_users):,}")
print(f"  train_df:    {len(train_df):,} rows")

# Restrict to eval users
val_gt_eval = val_gt[val_gt["user_id"].isin(eval_users)]
gt_map = val_gt_eval.groupby("user_id")["item_id"].apply(set).to_dict()

tifu_eval = tifu[tifu["user_id"].isin(eval_users)]
bsarec_eval = bsarec[bsarec["user_id"].isin(eval_users)]

print(f"  after eval_users filter: gt={sum(len(s) for s in gt_map.values())} purchases, "
      f"{len(gt_map)} users")

results: dict = {}


# ----------------------------------------------------------------------
# A1. Prediction overlap
# ----------------------------------------------------------------------
print("\nA1. Prediction overlap (TIFU top-10 ∩ BSARec top-10)...")
tifu_top10 = tifu[tifu["rank"] <= 10].groupby("user_id")["item_id"].apply(set)
bsarec_top10 = bsarec[bsarec["rank"] <= 10].groupby("user_id")["item_id"].apply(set)
common = tifu_top10.index.intersection(bsarec_top10.index)
overlap = pd.Series(
    [len(tifu_top10[u] & bsarec_top10[u]) for u in common],
    index=common,
)

a1 = {
    "n_users": int(len(overlap)),
    "mean_overlap": float(overlap.mean()),
    "median_overlap": float(overlap.median()),
    "p25": float(overlap.quantile(0.25)),
    "p75": float(overlap.quantile(0.75)),
    "pct_zero_overlap": float((overlap == 0).mean()),
    "pct_full_overlap": float((overlap == 10).mean()),
}
results["A1_overlap"] = a1
print(f"  mean={a1['mean_overlap']:.2f} median={a1['median_overlap']:.0f} "
      f"p25={a1['p25']:.0f} p75={a1['p75']:.0f}")
print(f"  zero overlap users: {a1['pct_zero_overlap']*100:.1f}%, "
      f"full overlap: {a1['pct_full_overlap']*100:.1f}%")

plt.figure(figsize=(8, 4))
overlap.value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Per-user overlap: |TIFU top-10 ∩ BSARec top-10|")
plt.xlabel("Overlap count (0-10)")
plt.ylabel("# Users (all 638k)")
plt.tight_layout()
plt.savefig(OUT_DIR / "a1_overlap_histogram.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# Per-user NDCG (eval_users only, for A2-A5)
# ----------------------------------------------------------------------
print("\nComputing per-user NDCG@10 on eval_users...")
ndcg_tifu = per_user_ndcg10(tifu_eval, gt_map)
ndcg_bsarec = per_user_ndcg10(bsarec_eval, gt_map)
common_eval = ndcg_tifu.index.intersection(ndcg_bsarec.index)
ndcg_tifu = ndcg_tifu.loc[common_eval]
ndcg_bsarec = ndcg_bsarec.loc[common_eval]
delta = ndcg_tifu - ndcg_bsarec
print(f"  eval users with both NDCG: {len(common_eval)}")
print(f"  mean TIFU NDCG = {ndcg_tifu.mean():.4f}")
print(f"  mean BSARec NDCG = {ndcg_bsarec.mean():.4f}")
print(f"  mean delta = {delta.mean():.4f}")


# ----------------------------------------------------------------------
# A2. Repeat-ratio bins
# ----------------------------------------------------------------------
print("\nA2. Repeat-ratio bins (1 - distinct_items / total_events)...")
user_distinct = train_df.groupby("user_id")["item_id"].nunique()
user_total = train_df.groupby("user_id").size()
user_repeat_ratio = 1 - (user_distinct / user_total)
repeat_per_user = user_repeat_ratio.reindex(common_eval).fillna(0)

a2_df = pd.DataFrame({
    "tifu_ndcg": ndcg_tifu,
    "bsarec_ndcg": ndcg_bsarec,
    "repeat_ratio": repeat_per_user,
})
a2_df["bin"] = pd.cut(
    a2_df["repeat_ratio"],
    [-0.01, 0.0, 0.1, 0.3, 1.01],
    labels=["no_repeat (=0)", "low (0-0.1]", "mid (0.1-0.3]", "high (>0.3)"],
)
a2_summary = a2_df.groupby("bin", observed=True).agg(
    n_users=("tifu_ndcg", "count"),
    tifu_ndcg=("tifu_ndcg", "mean"),
    bsarec_ndcg=("bsarec_ndcg", "mean"),
).reset_index()
a2_summary["delta"] = a2_summary["tifu_ndcg"] - a2_summary["bsarec_ndcg"]
a2_summary.to_csv(OUT_DIR / "a2_repeat_bins.csv", index=False)
results["A2_repeat_bins"] = a2_summary.to_dict(orient="records")
print(a2_summary.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(a2_summary))
w = 0.35
ax.bar(x - w/2, a2_summary["tifu_ndcg"], w, label="TIFU", color="steelblue")
ax.bar(x + w/2, a2_summary["bsarec_ndcg"], w, label="BSARec", color="orange")
ax.set_xticks(x)
ax.set_xticklabels(a2_summary["bin"].astype(str), rotation=15)
ax.set_ylabel("Mean NDCG@10")
ax.set_title("A2. NDCG@10 by user repeat-ratio bin")
ax.legend()
for i, row in a2_summary.iterrows():
    ax.text(i, max(row["tifu_ndcg"], row["bsarec_ndcg"]) + 0.005,
            f"n={row['n_users']}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "a2_repeat_bins.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# A3. Sequence length bins (total events per user in train)
# ----------------------------------------------------------------------
print("\nA3. Sequence length bins (total events per user)...")
user_seqlen = user_total.reindex(common_eval).fillna(0)
a3_df = pd.DataFrame({
    "tifu_ndcg": ndcg_tifu,
    "bsarec_ndcg": ndcg_bsarec,
    "seqlen": user_seqlen,
})
a3_df["bin"] = pd.cut(
    a3_df["seqlen"],
    [-1, 3, 6, 15, 50, 1e9],
    labels=["1-3", "4-6", "7-15", "16-50", "50+"],
)
a3_summary = a3_df.groupby("bin", observed=True).agg(
    n_users=("tifu_ndcg", "count"),
    tifu_ndcg=("tifu_ndcg", "mean"),
    bsarec_ndcg=("bsarec_ndcg", "mean"),
).reset_index()
a3_summary["delta"] = a3_summary["tifu_ndcg"] - a3_summary["bsarec_ndcg"]
a3_summary.to_csv(OUT_DIR / "a3_seqlen_bins.csv", index=False)
results["A3_seqlen_bins"] = a3_summary.to_dict(orient="records")
print(a3_summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(a3_summary))
ax.bar(x - w/2, a3_summary["tifu_ndcg"], w, label="TIFU", color="steelblue")
ax.bar(x + w/2, a3_summary["bsarec_ndcg"], w, label="BSARec", color="orange")
ax.set_xticks(x)
ax.set_xticklabels(a3_summary["bin"].astype(str))
ax.set_ylabel("Mean NDCG@10")
ax.set_title("A3. NDCG@10 by user sequence length (BSARec max_seq=50 cap)")
ax.legend()
for i, row in a3_summary.iterrows():
    ax.text(i, max(row["tifu_ndcg"], row["bsarec_ndcg"]) + 0.005,
            f"n={row['n_users']}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "a3_seqlen_bins.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# A4. Item popularity bins for GT items
# ----------------------------------------------------------------------
print("\nA4. GT item popularity bins (hit rate per bin)...")
item_pop = train_df.groupby("item_id").size().sort_values(ascending=False)
item_pop_rank = pd.Series(np.arange(len(item_pop)), index=item_pop.index, name="pop_rank")
gt_items = val_gt_eval.merge(item_pop_rank.reset_index(), on="item_id", how="left")
gt_items["pop_bin"] = pd.cut(
    gt_items["pop_rank"],
    [-1, 100, 1000, 10000, 1e9],
    labels=["top-100", "101-1k", "1k-10k", "10k+"],
)

tifu_top10_map = {u: tifu_top10[u] for u in tifu_top10.index}
bsarec_top10_map = {u: bsarec_top10[u] for u in bsarec_top10.index}


def hit(row, m):
    return int(row["item_id"] in m.get(row["user_id"], set()))


gt_items["tifu_hit"] = gt_items.apply(lambda r: hit(r, tifu_top10_map), axis=1)
gt_items["bsarec_hit"] = gt_items.apply(lambda r: hit(r, bsarec_top10_map), axis=1)
a4_summary = gt_items.groupby("pop_bin", observed=True).agg(
    n_gt=("tifu_hit", "count"),
    tifu_hit_rate=("tifu_hit", "mean"),
    bsarec_hit_rate=("bsarec_hit", "mean"),
).reset_index()
a4_summary["delta"] = a4_summary["tifu_hit_rate"] - a4_summary["bsarec_hit_rate"]
a4_summary.to_csv(OUT_DIR / "a4_item_pop_bins.csv", index=False)
results["A4_item_pop_bins"] = a4_summary.to_dict(orient="records")
print(a4_summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(a4_summary))
ax.bar(x - w/2, a4_summary["tifu_hit_rate"], w, label="TIFU", color="steelblue")
ax.bar(x + w/2, a4_summary["bsarec_hit_rate"], w, label="BSARec", color="orange")
ax.set_xticks(x)
ax.set_xticklabels(a4_summary["pop_bin"].astype(str))
ax.set_ylabel("Hit rate (% GT in top-10)")
ax.set_title("A4. Hit rate by GT item popularity")
ax.legend()
for i, row in a4_summary.iterrows():
    ax.text(i, max(row["tifu_hit_rate"], row["bsarec_hit_rate"]) + 0.005,
            f"n={row['n_gt']}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "a4_item_pop_bins.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# A5. Per-user NDCG delta histogram + win/lose/tie
# ----------------------------------------------------------------------
print("\nA5. Per-user NDCG delta (TIFU - BSARec)...")
a5 = {
    "n_users": int(len(delta)),
    "mean_delta": float(delta.mean()),
    "median_delta": float(delta.median()),
    "pct_tifu_strict_win": float((delta > 0).mean()),
    "pct_bsarec_strict_win": float((delta < 0).mean()),
    "pct_tie": float((delta == 0).mean()),
    "pct_tie_both_zero": float(((delta == 0) & (ndcg_tifu == 0)).mean()),
}
results["A5_per_user_delta"] = a5
print(f"  TIFU win  : {a5['pct_tifu_strict_win']*100:.1f}%")
print(f"  BSARec win: {a5['pct_bsarec_strict_win']*100:.1f}%")
print(f"  tie       : {a5['pct_tie']*100:.1f}% "
      f"(both zero: {a5['pct_tie_both_zero']*100:.1f}%)")
print(f"  mean delta: {a5['mean_delta']:.4f}")

plt.figure(figsize=(8, 4))
nonzero = delta[delta != 0]
plt.hist(nonzero, bins=30, color="steelblue", edgecolor="black")
plt.axvline(0, color="red", linestyle="--", label="tie")
plt.axvline(delta.mean(), color="green", linestyle="-", label=f"mean={delta.mean():.4f}")
plt.title(f"A5. Per-user NDCG@10 delta (TIFU - BSARec), nonzero only "
          f"(n={len(nonzero)})")
plt.xlabel("delta")
plt.ylabel("# Users")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "a5_delta_histogram.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# Dump
# ----------------------------------------------------------------------
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)

print(f"\nDone. Output in {OUT_DIR}/")
for p in sorted(OUT_DIR.glob("*")):
    print(f"  {p.name}")
