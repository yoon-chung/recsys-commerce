"""exp_010_lgbm_reranker / build_features.py -- per-(user, candidate) feature 빌드.

LGBM reranker 의 input 을 한 번 계산하고 parquet 으로 저장.
train_cv.py + inference.py 에서 재사용 (모두 같은 feature space).

Output:
    cache/features_all.parquet
        columns: user_id, item_id, [40여개 feature columns], label (val_gt 있는 user 만)
        rows: 638k user × ~150 candidate = ~95M rows

Note: 638k user × 150 candidates = 95M rows = ~5-10 GB memory.
충분 (server 251GB RAM). 하지만 효율적인 vectorization 필수.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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


# ----------------------------------------------------------------------
# Candidate pool: union of top-K per model
# ----------------------------------------------------------------------
def build_candidate_pool(
    model_preds: dict[str, pd.DataFrame],
    top_k_per_model: int,
) -> pd.DataFrame:
    """Union top-K from each model into (user_id, item_id) candidates."""
    parts = []
    for name, df in model_preds.items():
        sub = df[df["rank"] <= top_k_per_model][["user_id", "item_id"]].copy()
        sub["src"] = name
        parts.append(sub)
    candidates = pd.concat(parts, ignore_index=True)
    # dedup (user, item) — keep first src for tracking
    candidates = candidates.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    candidates = candidates[["user_id", "item_id"]]
    return candidates


# ----------------------------------------------------------------------
# Feature group: model ranks/scores
# ----------------------------------------------------------------------
def add_model_rank_features(
    cand: pd.DataFrame,
    model_preds: dict[str, pd.DataFrame],
    top_k: int,
) -> pd.DataFrame:
    """Per model add rank, normalized score, in_top_10 binary."""
    df = cand
    for name, pred_df in model_preds.items():
        # Per-user min-max normalize score within top-K
        sub = pred_df[pred_df["rank"] <= top_k][["user_id", "item_id", "rank", "score"]].copy()
        grp = sub.groupby("user_id")["score"]
        mn = grp.transform("min")
        mx = grp.transform("max")
        rng = mx - mn
        sub[f"{name}_norm_score"] = np.where(rng > 0, (sub["score"] - mn) / rng, 1.0)
        sub[f"{name}_rank"] = sub["rank"]
        sub[f"{name}_in_top10"] = (sub["rank"] <= 10).astype(np.int8)
        sub = sub[["user_id", "item_id", f"{name}_norm_score", f"{name}_rank", f"{name}_in_top10"]]
        df = df.merge(sub, on=["user_id", "item_id"], how="left")
    # NaN means item not in this model's top-K -> sentinel values
    for name in model_preds:
        df[f"{name}_norm_score"] = df[f"{name}_norm_score"].fillna(0.0).astype(np.float32)
        df[f"{name}_rank"] = df[f"{name}_rank"].fillna(top_k + 10).astype(np.int16)   # sentinel beyond top-K
        df[f"{name}_in_top10"] = df[f"{name}_in_top10"].fillna(0).astype(np.int8)
    # Cross-model: # of models with this item in top-10
    in_top10_cols = [f"{n}_in_top10" for n in model_preds]
    df["n_models_in_top10"] = df[in_top10_cols].sum(axis=1).astype(np.int8)
    return df


# ----------------------------------------------------------------------
# Feature group: item popularity
# ----------------------------------------------------------------------
def build_item_popularity(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per item: total/view/cart/purchase counts, unique users, days active, avg price."""
    df = train_df.copy()
    df["ts"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
    cutoff = df["ts"].max()
    df["days_ago"] = (cutoff - df["ts"]).dt.total_seconds() / 86400.0

    total = df.groupby("item_id").size().rename("item_total_events")
    view = df[df["event_type"] == "view"].groupby("item_id").size().rename("item_view")
    cart = df[df["event_type"] == "cart"].groupby("item_id").size().rename("item_cart")
    purchase = df[df["event_type"] == "purchase"].groupby("item_id").size().rename("item_purchase")
    unique_users = df.groupby("item_id")["user_id"].nunique().rename("item_unique_users")
    days_first = df.groupby("item_id")["days_ago"].max().rename("item_days_since_first")
    days_last = df.groupby("item_id")["days_ago"].min().rename("item_days_since_last")
    avg_price = df.groupby("item_id")["price"].mean().rename("item_avg_price")
    pop = pd.concat([total, view, cart, purchase, unique_users,
                     days_first, days_last, avg_price], axis=1).reset_index()
    for col in ["item_view", "item_cart", "item_purchase"]:
        pop[col] = pop[col].fillna(0)
    # log-transform popularity counts (skewed)
    for col in ["item_total_events", "item_view", "item_cart", "item_purchase", "item_unique_users"]:
        pop[f"{col}_log1p"] = np.log1p(pop[col]).astype(np.float32)
    return pop


# ----------------------------------------------------------------------
# Feature group: user activity
# ----------------------------------------------------------------------
def build_user_activity(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df["ts"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
    cutoff = df["ts"].max()
    df["days_ago"] = (cutoff - df["ts"]).dt.total_seconds() / 86400.0

    total = df.groupby("user_id").size().rename("user_total_events")
    distinct_items = df.groupby("user_id")["item_id"].nunique().rename("user_distinct_items")
    view = df[df["event_type"] == "view"].groupby("user_id").size().rename("user_view")
    cart = df[df["event_type"] == "cart"].groupby("user_id").size().rename("user_cart")
    purchase = df[df["event_type"] == "purchase"].groupby("user_id").size().rename("user_purchase")
    days_first = df.groupby("user_id")["days_ago"].max().rename("user_days_since_first")
    days_last = df.groupby("user_id")["days_ago"].min().rename("user_days_recency")
    act = pd.concat([total, distinct_items, view, cart, purchase, days_first, days_last], axis=1).reset_index()
    for col in ["user_view", "user_cart", "user_purchase"]:
        act[col] = act[col].fillna(0)
    act["user_repeat_ratio"] = 1 - act["user_distinct_items"] / act["user_total_events"]
    act["user_cart_view_ratio"] = act["user_cart"] / act["user_view"].clip(lower=1)
    act["user_purchase_view_ratio"] = act["user_purchase"] / act["user_view"].clip(lower=1)
    for col in ["user_total_events", "user_distinct_items", "user_view", "user_cart", "user_purchase"]:
        act[f"{col}_log1p"] = np.log1p(act[col]).astype(np.float32)
    return act


# ----------------------------------------------------------------------
# Feature group: user-item history
# ----------------------------------------------------------------------
def build_user_item_history(train_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Per (user, item) past interaction stats. Memory-tight version."""
    df = train_df[["user_id", "item_id", "event_type", "event_time"]].copy()
    df["ts"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S %Z")
    cutoff = df["ts"].max()
    df["days_ago"] = ((cutoff - df["ts"]).dt.total_seconds() / 86400.0).astype(np.float32)
    df = df.drop(columns=["event_time", "ts"])

    grp = df.groupby(["user_id", "item_id"], sort=False)
    total = grp.size().astype(np.int16).rename("ui_total_events")
    days_last = grp["days_ago"].min().rename("ui_days_since_last")
    hist = pd.concat([total, days_last], axis=1)

    for ev in ["view", "cart", "purchase"]:
        cnt = (df[df["event_type"] == ev]
               .groupby(["user_id", "item_id"], sort=False).size()
               .astype(np.int16).rename(f"ui_{ev}"))
        hist = hist.join(cnt, how="left")
    for col in ["ui_view", "ui_cart", "ui_purchase"]:
        hist[col] = hist[col].fillna(0).astype(np.int16)
    return hist.reset_index()


# ----------------------------------------------------------------------
# Feature group: user-item affinity (brand/category match)
# ----------------------------------------------------------------------
def build_affinity_lookups(train_df: pd.DataFrame) -> dict:
    """Return small lookup Series (indexed by user_id / item_id) for brand+category top1."""
    db = train_df.dropna(subset=["brand"])
    user_brand_top1 = (
        db.groupby("user_id")["brand"]
        .agg(lambda s: s.mode().iat[0] if not s.empty else None)
    )
    item_brand = (
        db.groupby("item_id")["brand"]
        .agg(lambda s: s.mode().iat[0] if not s.empty else None)
    )
    dc = train_df.dropna(subset=["category_code"])
    user_cat_top1 = (
        dc.groupby("user_id")["category_code"]
        .agg(lambda s: s.mode().iat[0] if not s.empty else None)
    )
    item_cat = (
        dc.groupby("item_id")["category_code"]
        .agg(lambda s: s.mode().iat[0] if not s.empty else None)
    )
    return {
        "user_brand_top1": user_brand_top1,
        "item_brand": item_brand,
        "user_cat_top1": user_cat_top1,
        "item_cat": item_cat,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
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

    # ---- 1. Load data ---------------------------------------------------
    logger.info("loading train data + time_based_split ...")
    df_full = load_train_data(cfg["train_data"])
    train_df, _ = time_based_split(
        df_full,
        val_days=cfg["val_days"],
        gt_event_types=cfg["gt_event_types"],
    )
    logger.info("train_df: %s rows", f"{len(train_df):,}")

    # ---- 2. Load model predictions --------------------------------------
    logger.info("loading model predictions ...")
    model_preds = {}
    for name, rel in cfg["model_predictions"].items():
        p = (HERE / rel).resolve()
        if not p.exists():
            logger.warning("  %s MISSING (%s), SKIP", name, p)
            continue
        df = pd.read_parquet(p)
        model_preds[name] = df
        logger.info("  %s: %s rows", name, f"{len(df):,}")

    # ---- 3. Candidate pool ---------------------------------------------
    t0 = time.time()
    candidates = build_candidate_pool(model_preds, cfg["top_k_per_model"])
    logger.info("candidates: %s (user, item) pairs in %.1fs",
                f"{len(candidates):,}", time.time() - t0)

    # ---- 4. Features ---------------------------------------------------
    fg = cfg["feature_groups"]

    if fg.get("model_ranks", True):
        logger.info("[A] model_ranks features ...")
        t0 = time.time()
        candidates = add_model_rank_features(candidates, model_preds, cfg["top_k_per_model"])
        logger.info("  done in %.1fs (cols=%d)", time.time() - t0, len(candidates.columns))

    if fg.get("item_popularity", True):
        logger.info("[B] item_popularity features ...")
        t0 = time.time()
        item_pop = build_item_popularity(train_df)
        candidates = candidates.merge(item_pop, on="item_id", how="left")
        logger.info("  done in %.1fs (cols=%d)", time.time() - t0, len(candidates.columns))

    if fg.get("user_activity", True):
        logger.info("[C] user_activity features ...")
        t0 = time.time()
        user_act = build_user_activity(train_df)
        candidates = candidates.merge(user_act, on="user_id", how="left")
        logger.info("  done in %.1fs (cols=%d)", time.time() - t0, len(candidates.columns))

    # ---- 5. Pre-compute lookups for chunked [D]+[E]+label loop ---------
    do_hist = fg.get("user_item_history", True)
    do_aff = fg.get("user_item_affinity", True)

    ui_hist = build_user_item_history(train_df, candidates) if do_hist else None
    if ui_hist is not None:
        logger.info("[D] hist lookup built: %s rows", f"{len(ui_hist):,}")

    aff_lk = build_affinity_lookups(train_df) if do_aff else None
    if aff_lk is not None:
        logger.info("[E] affinity lookups built (n_user_brand=%d, n_item_brand=%d)",
                    len(aff_lk["user_brand_top1"]), len(aff_lk["item_brand"]))

    val_gt = pd.read_parquet((HERE / cfg["ease_saved"] / "val_gt.parquet").resolve())
    with open((HERE / cfg["ease_saved"] / "eval_users.json").resolve(), encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt[val_gt["user_id"].isin(eval_users)][["user_id", "item_id"]].copy()
    val_gt_eval["label"] = np.int8(1)

    if fg.get("co_occurrence", False):
        logger.warning("[F] co_occurrence -- TODO, skipping")

    # ---- 6. Chunked merge + write -------------------------------------
    # features_all = directory of parquet files (read with pd.read_parquet(dir))
    # peak memory: only one chunk in RAM at a time (~3GB) + lookups
    out_dir = cache_dir / "features_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear stale parts
    for p in out_dir.glob("part_*.parquet"):
        p.unlink()

    n_chunks = 8
    user_ids_sorted = np.sort(candidates["user_id"].unique())
    chunk_size = (len(user_ids_sorted) + n_chunks - 1) // n_chunks

    n_pos_total = 0
    n_eval_users_total = 0
    n_rows_total = 0
    t0 = time.time()
    for i in range(n_chunks):
        users_chunk = set(user_ids_sorted[i * chunk_size:(i + 1) * chunk_size])
        cand_chunk = candidates[candidates["user_id"].isin(users_chunk)].copy()

        # [D] history merge
        if ui_hist is not None:
            hist_chunk = ui_hist[ui_hist["user_id"].isin(users_chunk)]
            cand_chunk = cand_chunk.merge(hist_chunk, on=["user_id", "item_id"], how="left")
            for col in ["ui_total_events", "ui_view", "ui_cart", "ui_purchase"]:
                cand_chunk[col] = cand_chunk[col].fillna(0).astype(np.int16)
            cand_chunk["ui_days_since_last"] = cand_chunk["ui_days_since_last"].fillna(999).astype(np.float32)

        # [E] affinity via map() lookups (no merge)
        if aff_lk is not None:
            cand_chunk["user_top_brand"] = cand_chunk["user_id"].map(aff_lk["user_brand_top1"])
            cand_chunk["item_brand"] = cand_chunk["item_id"].map(aff_lk["item_brand"])
            cand_chunk["brand_match"] = (
                (cand_chunk["user_top_brand"] == cand_chunk["item_brand"])
                .fillna(False).astype(np.int8)
            )
            cand_chunk["user_top_category"] = cand_chunk["user_id"].map(aff_lk["user_cat_top1"])
            cand_chunk["item_category"] = cand_chunk["item_id"].map(aff_lk["item_cat"])
            cand_chunk["category_match"] = (
                (cand_chunk["user_top_category"] == cand_chunk["item_category"])
                .fillna(False).astype(np.int8)
            )

        # labels
        gt_chunk = val_gt_eval[val_gt_eval["user_id"].isin(users_chunk)]
        cand_chunk = cand_chunk.merge(gt_chunk, on=["user_id", "item_id"], how="left")
        cand_chunk["label"] = cand_chunk["label"].fillna(0).astype(np.int8)
        cand_chunk["is_eval_user"] = cand_chunk["user_id"].isin(eval_users).astype(np.int8)

        n_pos_total += int(cand_chunk["label"].sum())
        n_eval_users_total += int(cand_chunk[cand_chunk["is_eval_user"] == 1]["user_id"].nunique())
        n_rows_total += len(cand_chunk)

        part_path = out_dir / f"part_{i:02d}.parquet"
        cand_chunk.to_parquet(part_path)
        logger.info("  chunk %d/%d: %s rows, %d cols, wrote %s (%.1f MB)",
                    i + 1, n_chunks, f"{len(cand_chunk):,}", len(cand_chunk.columns),
                    part_path.name, part_path.stat().st_size / 1024**2)
        del cand_chunk

    logger.info("all chunks written in %.1fs", time.time() - t0)
    logger.info("totals: %s rows, %s positives across %s eval users",
                f"{n_rows_total:,}", f"{n_pos_total:,}", f"{n_eval_users_total:,}")
    logger.info("output dir: %s", out_dir)


if __name__ == "__main__":
    main()
