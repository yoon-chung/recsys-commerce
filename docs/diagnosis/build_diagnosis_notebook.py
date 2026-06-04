"""Build docs/diagnosis_tifu_vs_bsarec.ipynb from the .py script + .md writeup.

Once-off generator. After running this, the .ipynb is the source of truth;
this script is kept for regeneration if the markdown narrative is updated.

Usage:
    python scripts/build_diagnosis_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "diagnosis_tifu_vs_bsarec.ipynb"
SUMMARY_JSON = ROOT / "docs" / "diagnosis" / "summary.json"

# Load already-computed summary so cells can render expected outputs as
# comments (notebook is portfolio-readable without re-running).
with open(SUMMARY_JSON, encoding="utf-8") as f:
    SUMMARY = json.load(f)

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str, outputs: list | None = None) -> None:
    cell = nbf.v4.new_code_cell(text)
    if outputs:
        cell.outputs = outputs
        cell.execution_count = 1
    cells.append(cell)


def text_output(text: str) -> dict:
    return nbf.v4.new_output(
        output_type="stream",
        name="stdout",
        text=text,
    )


# ----------------------------------------------------------------------
# Title + TL;DR
# ----------------------------------------------------------------------
md(
    """# TIFU-KNN vs BSARec — Why Classical Beats Deep on This Dataset

**Date**: 2026-05-22 · **Author**: cy

같은 데이터에서 100줄짜리 2020 paper KNN method (TIFU-KNN, SIGIR 2020) 가 transformer 4종을 **public NDCG +20.5%** 로 압도. 5개 segment-level 진단 분석으로 mechanism 정량 증명.

## TL;DR

| 강점 영역 | TIFU 우위 (NDCG@10) | n_users |
|---|---:|---:|
| **Repeat-buyer users** (repeat_ratio > 0.3) | **+0.053** | 521 / 923 (56%) |
| **Long-history users** (seq_len 50+) | **+0.091** | 190 / 923 (21%) |
| **Popular GT items** (top-100) | **+0.092** (hit rate) | 422 / 1,223 GT (34%) |

**Mechanism**:
1. Frequency vector 가 user-item repeat 패턴을 직접 모델링 → BSARec attention 의 parametric 압축으로 손실되는 신호
2. TIFU 는 모든 history 활용 (decay 로 weight) → BSARec `max_seq=50` cap 정보 손실 회피
3. Decay weighting 이 Feb 27-29 spike target items 자동 강조

**Public scores**:
- TIFU-KNN (exp_007): **0.1175**
- BSARec 4w_full (exp_002e): 0.0975
- Δ: **+0.0200 (+20.5%)**
"""
)

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
md(
    """## Setup

| 항목 | 값 |
|---|---|
| Eval users | 928명 (`exp_001 EASE 의 eval_users.json` 재사용) |
| GT (val) | 1,223 purchase events, Feb 23-29 |
| TIFU-KNN model | exp_007 (4m holdout, paper default) |
| BSARec model | exp_002e (4w_full, spike+) |
| Per-user NDCG@10 | binary relevance, 직접 구현 |

A1 만 전체 638k user, 나머지 A2-A5 는 eval_users 928명."""
)

code(
    """import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib 없음 -- plot skip, CSV/숫자만 출력")

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "docs" else Path.cwd()
TIFU_PRED = PROJECT_ROOT / "experiments" / "exp_007_tifu_knn" / "predictions.parquet"
BSAREC_PRED = PROJECT_ROOT / "experiments" / "exp_002e_bsarec_4w_full" / "predictions.parquet"
EASE_SAVED = PROJECT_ROOT / "experiments" / "exp_001_ease" / "saved"
TRAIN_PARQUET = Path("/root/data/train.parquet")  # server canonical path
"""
)

# helper
code(
    '''def per_user_ndcg10(pred_df: pd.DataFrame, gt_map: dict) -> pd.Series:
    """Returns Series of NDCG@10 (binary relevance) indexed by user_id."""
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    cum = np.cumsum(discounts)
    pred = pred_df[pred_df["rank"] <= 10].sort_values(["user_id", "rank"])
    out = {}
    for uid, group in pred.groupby("user_id", sort=False):
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
'''
)

# Load data
code(
    """print("Loading...")
tifu = pd.read_parquet(TIFU_PRED)
bsarec = pd.read_parquet(BSAREC_PRED)
val_gt = pd.read_parquet(EASE_SAVED / "val_gt.parquet")
with open(EASE_SAVED / "eval_users.json", encoding="utf-8") as f:
    eval_users = set(json.load(f))
train_df = pd.read_parquet(TRAIN_PARQUET)

val_gt_eval = val_gt[val_gt["user_id"].isin(eval_users)]
gt_map = val_gt_eval.groupby("user_id")["item_id"].apply(set).to_dict()
tifu_eval = tifu[tifu["user_id"].isin(eval_users)]
bsarec_eval = bsarec[bsarec["user_id"].isin(eval_users)]

print(f"tifu pred   : {len(tifu):,} rows, {tifu['user_id'].nunique():,} users")
print(f"bsarec pred : {len(bsarec):,} rows, {bsarec['user_id'].nunique():,} users")
print(f"val_gt eval : {len(val_gt_eval):,} purchases, {len(gt_map):,} users")
""",
    outputs=[
        text_output(
            "Loading...\n"
            "tifu pred   : 31,193,300 rows, 623,866 users\n"
            "bsarec pred : 31,193,300 rows, 623,866 users\n"
            "val_gt eval : 1,223 purchases, 928 users\n"
        )
    ],
)

# ----------------------------------------------------------------------
# A1
# ----------------------------------------------------------------------
md(
    """## A1. Prediction overlap — 두 모델이 진짜 다른 후보를 뽑는가?

`per-user overlap = |TIFU top-10 ∩ BSARec top-10|` (전체 638k user 대상)"""
)

a1 = SUMMARY["A1_overlap"]
code(
    """tifu_top10 = tifu[tifu["rank"] <= 10].groupby("user_id")["item_id"].apply(set)
bsarec_top10 = bsarec[bsarec["rank"] <= 10].groupby("user_id")["item_id"].apply(set)
common = tifu_top10.index.intersection(bsarec_top10.index)
overlap = pd.Series(
    [len(tifu_top10[u] & bsarec_top10[u]) for u in common],
    index=common,
)
print(f"n_users          : {len(overlap):,}")
print(f"mean overlap     : {overlap.mean():.2f} / 10")
print(f"median           : {overlap.median():.0f}")
print(f"p25 / p75        : {overlap.quantile(0.25):.0f} / {overlap.quantile(0.75):.0f}")
print(f"zero overlap     : {(overlap == 0).mean()*100:.2f}%")
print(f"full overlap     : {(overlap == 10).mean()*100:.2f}%")

if HAS_MPL:
    plt.figure(figsize=(8, 4))
    overlap.value_counts().sort_index().plot(kind="bar", color="steelblue")
    plt.title("Per-user overlap: |TIFU top-10 ∩ BSARec top-10|")
    plt.xlabel("Overlap count (0-10)"); plt.ylabel("# Users")
    plt.tight_layout(); plt.show()
""",
    outputs=[
        text_output(
            f"n_users          : {a1['n_users']:,}\n"
            f"mean overlap     : {a1['mean_overlap']:.2f} / 10\n"
            f"median           : {a1['median_overlap']:.0f}\n"
            f"p25 / p75        : {a1['p25']:.0f} / {a1['p75']:.0f}\n"
            f"zero overlap     : {a1['pct_zero_overlap']*100:.2f}%\n"
            f"full overlap     : {a1['pct_full_overlap']*100:.2f}%\n"
        )
    ],
)

md(
    """**해석**:
- 평균 **36% (3.63/10)** 만 겹친다 = **64% 가 다른 후보**
- Ensemble 가치 큼: TIFU recall 0.39 + BSARec recall 0.32 → 후보 set diversity 높음
- Zero-overlap (완전 disjoint) 0.76% 만 — 같은 user/item universe 라 완전 disjoint 은 드묾"""
)

# Compute per-user NDCG (shared by A2-A5)
code(
    """print("Computing per-user NDCG@10 on eval users...")
ndcg_tifu = per_user_ndcg10(tifu_eval, gt_map)
ndcg_bsarec = per_user_ndcg10(bsarec_eval, gt_map)
common_eval = ndcg_tifu.index.intersection(ndcg_bsarec.index)
ndcg_tifu = ndcg_tifu.loc[common_eval]
ndcg_bsarec = ndcg_bsarec.loc[common_eval]
delta = ndcg_tifu - ndcg_bsarec
print(f"eval users (both): {len(common_eval)}")
print(f"mean TIFU NDCG   : {ndcg_tifu.mean():.4f}")
print(f"mean BSARec NDCG : {ndcg_bsarec.mean():.4f}")
print(f"mean delta       : {delta.mean():.4f}")
""",
    outputs=[
        text_output(
            "Computing per-user NDCG@10 on eval users...\n"
            "eval users (both): 923\n"
            "mean TIFU NDCG   : 0.2922\n"
            "mean BSARec NDCG : 0.2471\n"
            "mean delta       : 0.0451\n"
        )
    ],
)

# ----------------------------------------------------------------------
# A2
# ----------------------------------------------------------------------
md(
    """## A2. Repeat-ratio bins — TIFU 강점이 진짜 repeat user 에 집중되는가? ★

`user_repeat_ratio = 1 - distinct_items / total_events`
- 0 = 매번 다른 item, 1 = 모든 event 가 같은 item 반복"""
)

a2_rows = SUMMARY["A2_repeat_bins"]
a2_table = "bin            n_users  tifu_ndcg  bsarec_ndcg     delta\n"
for r in a2_rows:
    a2_table += (
        f"{r['bin']:<14}  {r['n_users']:>5}  {r['tifu_ndcg']:>9.4f}  "
        f"{r['bsarec_ndcg']:>11.4f}  {r['delta']:>+8.4f}\n"
    )
code(
    '''user_distinct = train_df.groupby("user_id")["item_id"].nunique()
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
print(a2_summary.to_string(index=False))
''',
    outputs=[text_output(a2_table)],
)

md(
    """**해석 — 가설 정확히 확인**:
- **No-repeat user (n=101)**: BSARec 가 살짝 우위 (−0.004). TIFU frequency 무용지물
- **Repeat-ratio 가 높을수록 TIFU 우위 점진 확대**
- **High-repeat user (n=521, eval 의 56%)**: TIFU **+0.053 NDCG 우위**
- 결론: **TIFU 의 강점은 정확히 "user X 가 item Y 를 반복 view/cart" 신호의 직접 모델링**

이 데이터의 dominant user behavior 는 "본 것 또 보기" — view 99.78% + spike 99.7% 의 repeat-purchase 패턴. TIFU 의 `user_freq_vec` 가 이 신호를 직접 잡고, BSARec 의 parametric attention 은 압축/추상화하며 손실."""
)

# ----------------------------------------------------------------------
# A3
# ----------------------------------------------------------------------
md(
    """## A3. Sequence length bins — BSARec `max_seq=50` cap 영향 ★★

`user_seqlen = train 의 user 별 total events 수`"""
)

a3_rows = SUMMARY["A3_seqlen_bins"]
a3_table = "bin     n_users  tifu_ndcg  bsarec_ndcg     delta\n"
for r in a3_rows:
    a3_table += (
        f"{r['bin']:<6}  {r['n_users']:>5}  {r['tifu_ndcg']:>9.4f}  "
        f"{r['bsarec_ndcg']:>11.4f}  {r['delta']:>+8.4f}\n"
    )
code(
    '''user_seqlen = user_total.reindex(common_eval).fillna(0)
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
print(a3_summary.to_string(index=False))
''',
    outputs=[text_output(a3_table)],
)

md(
    """**해석 — 가장 강력한 mechanism evidence**:
- **50+ user (n=190, eval 의 21%)**: TIFU **+0.091 NDCG 압도**
- BSARec 는 `max_seq=50` 으로 최근 50개만 attention 입력 → **51번째 이후 history 완전 손실**
- TIFU 는 모든 history 를 group 7개로 decay 적용해 활용 → 정보 손실 X
- 1-3 짧은 seq 에서도 TIFU 우위 (+0.014) — repeat 신호가 짧은 history 에서도 작동
- 16-50 (cap 안) 에서도 +0.043 → cap 영향만이 아니라 mechanism 자체 차이

**Portfolio talking point**:
> "BSARec `max_seq=50` cap 으로 long-history user (eval 의 21%) 의 정보가 완전 손실되는 걸 데이터로 잡음. TIFU 는 group-based decay 로 모든 history 활용 → 50+ user 에서 +0.091 NDCG 우위. Architecture 선택이 실데이터 user segment 에 미치는 영향의 구체적 예시.\""""
)

# ----------------------------------------------------------------------
# A4
# ----------------------------------------------------------------------
md(
    """## A4. Item popularity bins — GT items 의 popularity 별 hit rate

`item_pop_rank = train 등장 빈도 기준 rank (0 = 가장 popular)`
Per (user, GT_item) hit rate (top-10 안에 들어갔는지)."""
)

a4_rows = SUMMARY["A4_item_pop_bins"]
a4_table = "pop_bin    n_gt   tifu_hit  bsarec_hit     delta\n"
for r in a4_rows:
    a4_table += (
        f"{r['pop_bin']:<9}  {r['n_gt']:>5}  {r['tifu_hit_rate']:>7.3%}  "
        f"{r['bsarec_hit_rate']:>9.3%}  {r['delta']:>+8.4f}\n"
    )
code(
    '''item_pop = train_df.groupby("item_id").size().sort_values(ascending=False)
item_pop_rank = pd.Series(np.arange(len(item_pop)), index=item_pop.index, name="pop_rank")
gt_items = val_gt_eval.merge(item_pop_rank.reset_index(), on="item_id", how="left")
gt_items["pop_bin"] = pd.cut(
    gt_items["pop_rank"],
    [-1, 100, 1000, 10000, 1e9],
    labels=["top-100", "101-1k", "1k-10k", "10k+"],
)

tifu_top10_map = tifu_top10.to_dict()
bsarec_top10_map = bsarec_top10.to_dict()

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
print(a4_summary.to_string(index=False))
''',
    outputs=[text_output(a4_table)],
)

md(
    """**해석**:
- **Top-100 popular items (n=422, GT 의 34%)**: TIFU 압도 (+9.2% hit rate)
- TIFU 의 KNN aggregation 이 popular item 의 collaborative signal 잘 활용
- **Long-tail (10k+) 에서는 BSARec 가 살짝 우위** (−0.9%) — KNN 의 long-tail 약점
- 그러나 GT 의 10k+ 비중 작음 (8.7%) → 전체 점수 영향 작음

Feb 27-29 spike 의 target items 가 top-100 에 집중돼 있을 가능성 — TIFU 의 decay weighting 이 spike 직전 view/cart 빈도 ↑ 한 items 를 자동 강조하는 것과 일치."""
)

# ----------------------------------------------------------------------
# A5
# ----------------------------------------------------------------------
md(
    """## A5. Per-user NDCG delta — TIFU 가 균등하게 좋은가, super-user 만 좋은가?"""
)

a5 = SUMMARY["A5_per_user_delta"]
code(
    """print(f"n_users          : {len(delta)}")
print(f"mean delta       : {delta.mean():+.4f}")
print(f"median delta     : {delta.median():+.4f}")
print(f"TIFU strict win  : {(delta > 0).mean()*100:.1f}%")
print(f"BSARec strict win: {(delta < 0).mean()*100:.1f}%")
print(f"tie              : {(delta == 0).mean()*100:.1f}%")
print(f"  └ both NDCG=0  : {((delta == 0) & (ndcg_tifu == 0)).mean()*100:.1f}%")

if HAS_MPL:
    plt.figure(figsize=(8, 4))
    nonzero = delta[delta != 0]
    plt.hist(nonzero, bins=30, color="steelblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", label="tie")
    plt.axvline(delta.mean(), color="green", label=f"mean={delta.mean():.4f}")
    plt.title(f"Per-user NDCG@10 delta (TIFU - BSARec), nonzero only (n={len(nonzero)})")
    plt.xlabel("delta"); plt.ylabel("# Users"); plt.legend()
    plt.tight_layout(); plt.show()
""",
    outputs=[
        text_output(
            f"n_users          : {a5['n_users']}\n"
            f"mean delta       : {a5['mean_delta']:+.4f}\n"
            f"median delta     : {a5['median_delta']:+.4f}\n"
            f"TIFU strict win  : {a5['pct_tifu_strict_win']*100:.1f}%\n"
            f"BSARec strict win: {a5['pct_bsarec_strict_win']*100:.1f}%\n"
            f"tie              : {a5['pct_tie']*100:.1f}%\n"
            f"  └ both NDCG=0  : {a5['pct_tie_both_zero']*100:.1f}%\n"
        )
    ],
)

md(
    """**해석**:
- 절반 (54%) user 는 양쪽 모두 NDCG = 0 → **task 자체의 어려움** (val 의 99.7% spike 가 만든 distribution shift)
- 풀리는 user (46%) 중 TIFU 가 명확한 우위: **TIFU win 19.3% vs BSARec win 11.5% = 1.68배 빈도**
- 평균 +0.045 → A2/A3/A4 의 segment 별 우위가 누적된 결과

**중요한 nuance**: 단일 평균 metric (NDCG 0.292 vs 0.247) 으로는 잡히지 않는 user 분포 구조. 분석 가능한 segment 안에서만 비교해야 → A2/A3/A4 처럼 bin 별 진단이 핵심."""
)

# ----------------------------------------------------------------------
# Conclusions
# ----------------------------------------------------------------------
md(
    """## Conclusions — Why Classical Wins

1. **Dominant signal mismatch**:
   - 데이터의 핵심 signal = user-item **repeat frequency + temporal decay** (view 99.78% + spike 99.7%)
   - TIFU 는 이 signal 을 mechanism 자체로 직접 모델링 (user_vec 행렬 + decay weighting)
   - BSARec 의 parametric attention 은 signal 을 압축하며 일부 손실

2. **Information loss at scale**:
   - BSARec `max_seq=50` cap → long-history user (eval 의 21%) 의 정보 완전 손실
   - TIFU 는 group-based decay 로 모든 history 활용
   - 50+ user 에서 TIFU +0.091 NDCG 우위

3. **Popular item handling**:
   - Top-100 popular items (GT 의 34%) 에서 TIFU +9.2% hit rate
   - KNN aggregation 의 collaborative signal vs transformer 의 individual sequence signal

4. **Ensemble potential**:
   - 두 모델 후보 set 64% diverge
   - Long-tail items (8.7% of GT) 는 BSARec 가 살짝 우위 — 보완 관계
   - Ensemble v3 에서 weighted RRF 로 두 강점 결합 가능"""
)

md(
    """## Production E-commerce 연결

이 진단이 보여주는 mechanism 은 **production retail RecSys 의 industry pattern 과 정확히 일치**:

| Production reality | 우리 진단 |
|---|---|
| Amazon/Coupang 의 multi-stage retrieval 1차 candidate generator 로 classical KNN/co-visit | TIFU-KNN 단독으로 transformer 압도 |
| Repeat-purchase signal 의 explicit modeling (user 별 frequency vector) | TIFU 의 `user_freq_vec` |
| Temporal recency weighting (최근 본 item 우선) | TIFU 의 `decay_within`, `decay_across` |
| Cold-start fallback = popularity | 우리 cold-start (14k user) popularity_fallback 그대로 |

Research SOTA paper-chase 와 production reality 의 gap 을 **같은 데이터에서 정량 검증**."""
)

md(
    """## Interview Talking Point

> "Commerce 데이터 (4개월 행동 로그) 에 sequential transformer 4종 (BSARec, BERT4Rec, BSARec+CL hybrid, MB-STR) 을 ablation 했지만, 100줄짜리 2020 paper KNN method (TIFU-KNN) 가 public NDCG +20.5% 로 압도했습니다.
>
> 5개 segment-level 진단 분석으로 이유를 정량 검증했습니다: (1) repeat-buyer user 의 56% 에서 TIFU +0.053 우위 — 'user-item repeat frequency' 가 dominant signal 인데 transformer 의 parametric attention 으론 직접 잡지 못함. (2) long-history user 21% 에서 TIFU +0.091 우위 — BSARec `max_seq=50` cap 의 정보 손실 정량 증명. (3) top-100 popular items 에서 TIFU +9.2% hit rate — KNN aggregation 의 collaborative signal 우위.
>
> 이게 Amazon/Coupang 의 production RecSys 가 multi-stage retrieval 의 1차 candidate generator 로 classical KNN/co-visit 류를 유지하는 mechanism 입니다. Research paper SOTA 와 production reality 의 gap 을 데이터로 직접 검증했습니다."

## Future work

1. **Ensemble v3** — TIFU (popular item 강점) + BSARec (long-tail 강점) + BSARec+CL hybrid (recall 0.41) weighted RRF
2. **LLM reranker** on TIFU top-50 → contextual rerank 로 마지막 ordering 보정
3. **Mechanism transfer test** — 다른 retail dataset (Yoochoose, RetailRocket) 에서 같은 진단 재현되는지"""
)

# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10",
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(cells)} cells)")
