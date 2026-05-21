"""Loader and ID mapping utilities for the commerce purchase task.

The competition ships user/item identifiers as UUID strings. Models train
on contiguous 0-indexed integers, so we maintain a stable str <-> int
mapping derived from the full train set and cache it to disk for reuse
across experiments.

Cache layout
------------
`cache_mappings` writes two files into `cache_dir`:

    <cache_dir>/user2idx.json   {uuid_str: int_idx}
    <cache_dir>/item2idx.json   {uuid_str: int_idx}

Only the forward (str -> int) mappings are persisted. The inverse
(int -> str) is reconstructed on load. Storing int-keyed dicts in JSON is
lossy (json silently casts keys to strings on dump, and loading them back
gives string keys -- a subtle int/str mismatch bug). Skipping the inverse
on disk avoids the issue entirely with no real cost (one O(N) dict comp
on load).

** Do NOT cache to /root/data/ ** -- that directory holds the organizer's
data and is .gitignored / public-release-forbidden. The caller chooses
cache_dir explicitly.

Typical usage
-------------
    df = load_train_data("/root/data/train.parquet")
    mappings = get_or_build_mappings(df, cache_dir="./mappings_cache")
    df = add_idx_columns(df, mappings)
    # df now has user_idx, item_idx int64 columns
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EVENT_TIME_FMT = "%Y-%m-%d %H:%M:%S %Z"


def load_train_data(
    path: str = "/root/data/train.parquet",
    parse_time: bool = True,
) -> pd.DataFrame:
    """Load the competition train parquet.

    Args:
        path: Filesystem path to train.parquet. Default points to the
            canonical server location; on other machines pass an explicit path.
        parse_time: If True and `event_time` is still a string, parse it
            with the competition format '%Y-%m-%d %H:%M:%S %Z' into a
            tz-naive datetime64[ns] column (UTC, tz stripped).

    Returns:
        DataFrame with the parquet contents. `event_time` is datetime64[ns]
        when `parse_time=True`, untouched otherwise.
    """
    logger.info("loading train data from %s", path)
    df = pd.read_parquet(path)

    if parse_time and "event_time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["event_time"]):
            df["event_time"] = (
                pd.to_datetime(df["event_time"], format=EVENT_TIME_FMT, utc=True)
                .dt.tz_localize(None)
            )

    logger.info("  shape: %s", df.shape)
    if "event_time" in df.columns:
        logger.info(
            "  event_time range: %s ~ %s",
            df["event_time"].min(),
            df["event_time"].max(),
        )
    if "user_id" in df.columns:
        logger.info("  unique users: %s", f"{df['user_id'].nunique():,}")
    if "item_id" in df.columns:
        logger.info("  unique items: %s", f"{df['item_id'].nunique():,}")
    return df


def build_id_mappings(df: pd.DataFrame) -> dict:
    """Build 0-indexed integer mappings for user_id and item_id.

    Unique IDs are sorted before assignment so the same df always produces
    the same mapping (reproducibility across runs / experiments).

    Args:
        df: DataFrame with `user_id` and `item_id` columns.

    Returns:
        dict with four entries:
            'user2idx': dict[str, int]
            'idx2user': dict[int, str]
            'item2idx': dict[str, int]
            'idx2item': dict[int, str]
    """
    for col in ("user_id", "item_id"):
        if col not in df.columns:
            raise ValueError(f"df missing column: {col}")

    users = sorted(df["user_id"].unique().tolist())
    items = sorted(df["item_id"].unique().tolist())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}

    logger.info(
        "build_id_mappings: %s users, %s items",
        f"{len(user2idx):,}",
        f"{len(item2idx):,}",
    )
    return {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }


def cache_mappings(mappings: dict, cache_dir: str) -> None:
    """Persist forward mappings to `cache_dir` as JSON.

    Writes user2idx.json and item2idx.json. The inverse maps (idx2user,
    idx2item) are NOT written -- they are rebuilt on load to side-step the
    int-key serialization issue described in the module docstring.

    Args:
        mappings: Output of `build_id_mappings`. Must contain at least
            'user2idx' and 'item2idx'.
        cache_dir: Directory to write into. Created if missing. MUST NOT
            be `/root/data/` -- caller is responsible for choosing a path
            outside the organizer data directory.
    """
    for key in ("user2idx", "item2idx"):
        if key not in mappings:
            raise ValueError(f"mappings missing required key: {key}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    u_file = cache_path / "user2idx.json"
    i_file = cache_path / "item2idx.json"
    with u_file.open("w", encoding="utf-8") as f:
        json.dump(mappings["user2idx"], f, ensure_ascii=False)
    with i_file.open("w", encoding="utf-8") as f:
        json.dump(mappings["item2idx"], f, ensure_ascii=False)

    logger.info(
        "cache_mappings: wrote %s (%s users) and %s (%s items)",
        u_file,
        f"{len(mappings['user2idx']):,}",
        i_file,
        f"{len(mappings['item2idx']):,}",
    )


def load_mappings(cache_dir: str) -> dict:
    """Load forward mappings from `cache_dir` and rebuild inverse maps.

    Args:
        cache_dir: Directory holding user2idx.json and item2idx.json.

    Returns:
        Same shape as `build_id_mappings` output: dict with user2idx,
        idx2user, item2idx, idx2item.

    Raises:
        FileNotFoundError: if either JSON file is missing.
    """
    cache_path = Path(cache_dir)
    u_file = cache_path / "user2idx.json"
    i_file = cache_path / "item2idx.json"
    if not u_file.exists():
        raise FileNotFoundError(f"mapping file not found: {u_file}")
    if not i_file.exists():
        raise FileNotFoundError(f"mapping file not found: {i_file}")

    with u_file.open("r", encoding="utf-8") as f:
        user2idx: dict[str, int] = json.load(f)
    with i_file.open("r", encoding="utf-8") as f:
        item2idx: dict[str, int] = json.load(f)

    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}

    logger.info(
        "load_mappings: loaded %s users, %s items from %s",
        f"{len(user2idx):,}",
        f"{len(item2idx):,}",
        cache_dir,
    )
    return {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }


def get_or_build_mappings(
    df: pd.DataFrame,
    cache_dir: str,
    force_rebuild: bool = False,
) -> dict:
    """Load mappings from cache if present, otherwise build and cache.

    Args:
        df: Source DataFrame for building (used only on cache miss or
            when force_rebuild=True).
        cache_dir: Where to read/write JSON caches.
        force_rebuild: If True, rebuild from df even if cache exists,
            overwriting the cached files.

    Returns:
        Full mappings dict (see `build_id_mappings`).
    """
    cache_path = Path(cache_dir)
    u_file = cache_path / "user2idx.json"
    i_file = cache_path / "item2idx.json"
    cache_hit = u_file.exists() and i_file.exists()

    if cache_hit and not force_rebuild:
        logger.info("get_or_build_mappings: cache HIT at %s", cache_dir)
        return load_mappings(cache_dir)

    reason = "force_rebuild=True" if cache_hit else "cache MISS"
    logger.info("get_or_build_mappings: rebuilding (%s)", reason)
    mappings = build_id_mappings(df)
    cache_mappings(mappings, cache_dir)
    return mappings


def add_idx_columns(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Return a copy of `df` with int64 `user_idx` and `item_idx` columns.

    Args:
        df: DataFrame with `user_id` and `item_id` columns.
        mappings: Output of `build_id_mappings` / `load_mappings`.

    Returns:
        New DataFrame (input not mutated) with two appended int64 columns.

    Raises:
        ValueError: if any user_id or item_id in df is not in the mapping
            (indicates cold-start IDs the mapping wasn't built from).
    """
    for col in ("user_id", "item_id"):
        if col not in df.columns:
            raise ValueError(f"df missing column: {col}")

    user_idx = df["user_id"].map(mappings["user2idx"])
    item_idx = df["item_id"].map(mappings["item2idx"])

    n_missing_users = int(user_idx.isna().sum())
    n_missing_items = int(item_idx.isna().sum())
    if n_missing_users or n_missing_items:
        raise ValueError(
            f"add_idx_columns: {n_missing_users} user_id and {n_missing_items} item_id "
            "rows not in mappings (cold-start). Rebuild mappings on the full "
            "vocabulary or filter df to known IDs first."
        )

    df_out = df.copy()
    df_out["user_idx"] = user_idx.astype("int64")
    df_out["item_idx"] = item_idx.astype("int64")
    return df_out


if __name__ == "__main__":
    import shutil
    import tempfile
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- Build a synthetic train.parquet stand-in -----------------------
    # We can't reach /root/data/ from a local laptop, so we write a small
    # parquet to a temp dir and exercise the full cycle against it.
    tmpdir = Path(tempfile.mkdtemp(prefix="dataloader_test_"))
    print(f"\n=== using tmpdir: {tmpdir} ===")
    parquet_path = tmpdir / "train.parquet"
    cache_dir = tmpdir / "cache"

    base = pd.Timestamp("2020-02-20 00:00:00")
    rng_rows = []
    # 5 users, 6 items, mix of view/cart/purchase events with string event_time
    # to mirror the real schema (event_time is object/str in the raw file).
    users = [f"user-uuid-{i:02d}" for i in range(5)]
    items = [f"item-uuid-{i:02d}" for i in range(6)]
    for u_i, u in enumerate(users):
        for offset in range(3):
            rng_rows.append({
                "user_id": u,
                "item_id": items[(u_i + offset) % len(items)],
                "user_session": f"sess-{u_i}-{offset}",
                "event_time": (base + pd.Timedelta(days=u_i + offset)).strftime("%Y-%m-%d %H:%M:%S") + " UTC",
                "category_code": "electronics.smartphone" if offset == 0 else None,
                "brand": "samsung" if offset < 2 else None,
                "price": 199.99 + u_i * 10,
                "event_type": ["view", "cart", "purchase"][offset],
            })
    raw_df = pd.DataFrame(rng_rows)
    print(f"synthetic raw df: shape={raw_df.shape}, event_time dtype={raw_df['event_time'].dtype}")
    raw_df.to_parquet(parquet_path)

    try:
        # ---- load_train_data ---------------------------------------------
        print("\n=== load_train_data(parse_time=True) ===")
        df = load_train_data(str(parquet_path))
        assert pd.api.types.is_datetime64_any_dtype(df["event_time"]), "event_time not parsed"
        assert df.shape == raw_df.shape

        # ---- build_id_mappings -------------------------------------------
        print("\n=== build_id_mappings ===")
        mappings = build_id_mappings(df)
        assert set(mappings.keys()) == {"user2idx", "idx2user", "item2idx", "idx2item"}
        # forward maps are str -> int
        first_user = next(iter(mappings["user2idx"]))
        first_item = next(iter(mappings["item2idx"]))
        assert isinstance(first_user, str) and isinstance(mappings["user2idx"][first_user], int)
        assert isinstance(first_item, str) and isinstance(mappings["item2idx"][first_item], int)
        # inverse maps are int -> str
        first_uidx = next(iter(mappings["idx2user"]))
        first_iidx = next(iter(mappings["idx2item"]))
        assert isinstance(first_uidx, int) and isinstance(mappings["idx2user"][first_uidx], str)
        assert isinstance(first_iidx, int) and isinstance(mappings["idx2item"][first_iidx], str)
        # 0..N-1 coverage
        assert sorted(mappings["user2idx"].values()) == list(range(len(users)))
        assert sorted(mappings["item2idx"].values()) == list(range(len(items)))
        # round-trip: idx -> str -> idx
        for u, i in mappings["user2idx"].items():
            assert mappings["idx2user"][i] == u
        for it, i in mappings["item2idx"].items():
            assert mappings["idx2item"][i] == it
        # deterministic: same df -> same mapping (sorted IDs)
        m2 = build_id_mappings(df)
        assert mappings["user2idx"] == m2["user2idx"]
        assert mappings["item2idx"] == m2["item2idx"]
        print(f"user2idx sample: {dict(list(mappings['user2idx'].items())[:3])}")
        print(f"idx2item sample: {dict(list(mappings['idx2item'].items())[:3])}")

        # ---- cache_mappings + load_mappings round-trip -------------------
        print("\n=== cache_mappings + load_mappings ===")
        cache_mappings(mappings, str(cache_dir))
        assert (cache_dir / "user2idx.json").exists()
        assert (cache_dir / "item2idx.json").exists()
        # idx2user/idx2item are NOT serialized to disk
        assert not (cache_dir / "idx2user.json").exists()
        assert not (cache_dir / "idx2item.json").exists()
        loaded = load_mappings(str(cache_dir))
        assert loaded["user2idx"] == mappings["user2idx"]
        assert loaded["item2idx"] == mappings["item2idx"]
        # inverse rebuilt with correct int keys (not str)
        assert all(isinstance(k, int) for k in loaded["idx2user"].keys())
        assert all(isinstance(k, int) for k in loaded["idx2item"].keys())
        assert loaded["idx2user"] == mappings["idx2user"]
        assert loaded["idx2item"] == mappings["idx2item"]
        print("round-trip OK (int keys preserved on inverse)")

        # ---- get_or_build_mappings: cache HIT path -----------------------
        print("\n=== get_or_build_mappings (cache HIT) ===")
        t0 = time.perf_counter()
        m_hit = get_or_build_mappings(df, str(cache_dir))
        dt_hit = time.perf_counter() - t0
        assert m_hit["user2idx"] == mappings["user2idx"]
        print(f"cache hit took {dt_hit * 1000:.2f}ms")

        # ---- get_or_build_mappings: force_rebuild ------------------------
        print("\n=== get_or_build_mappings (force_rebuild=True) ===")
        m_force = get_or_build_mappings(df, str(cache_dir), force_rebuild=True)
        assert m_force["user2idx"] == mappings["user2idx"]

        # ---- get_or_build_mappings: cache MISS ---------------------------
        print("\n=== get_or_build_mappings (cache MISS) ===")
        empty_cache = tmpdir / "cache_empty"
        m_miss = get_or_build_mappings(df, str(empty_cache))
        assert m_miss["user2idx"] == mappings["user2idx"]
        assert (empty_cache / "user2idx.json").exists()

        # ---- add_idx_columns ---------------------------------------------
        print("\n=== add_idx_columns ===")
        df_idx = add_idx_columns(df, mappings)
        assert "user_idx" in df_idx.columns and "item_idx" in df_idx.columns
        assert df_idx["user_idx"].dtype == "int64"
        assert df_idx["item_idx"].dtype == "int64"
        # original df not mutated
        assert "user_idx" not in df.columns
        # idx columns match mappings
        for _, row in df_idx.iterrows():
            assert mappings["user2idx"][row["user_id"]] == row["user_idx"]
            assert mappings["item2idx"][row["item_id"]] == row["item_idx"]
        # round-trip: idx -> id
        first_row = df_idx.iloc[0]
        assert mappings["idx2user"][first_row["user_idx"]] == first_row["user_id"]
        assert mappings["idx2item"][first_row["item_idx"]] == first_row["item_id"]
        print(df_idx[["user_id", "user_idx", "item_id", "item_idx", "event_type"]].head().to_string(index=False))

        # ---- cold-start ID should raise ----------------------------------
        print("\n=== add_idx_columns: cold-start ID raises ===")
        df_cold = df.copy()
        df_cold.loc[df_cold.index[0], "user_id"] = "unseen-user"
        try:
            add_idx_columns(df_cold, mappings)
            raise AssertionError("expected ValueError for cold-start user_id")
        except ValueError as e:
            print(f"correctly raised: {e}")

        # ---- parse_time=False keeps strings ------------------------------
        print("\n=== load_train_data(parse_time=False) ===")
        df_str = load_train_data(str(parquet_path), parse_time=False)
        assert not pd.api.types.is_datetime64_any_dtype(df_str["event_time"]), (
            f"event_time should NOT be datetime when parse_time=False, "
            f"got dtype={df_str['event_time'].dtype}"
        )
        print(f"event_time dtype preserved: {df_str['event_time'].dtype}")

        print("\nAll sanity checks passed.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"cleaned up {tmpdir}")
