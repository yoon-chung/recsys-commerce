"""본체 추천 파이프라인 ↔ LLM 레이어 경계면.

두 가지 Protocol을 정의한다:
- RecsysAdapter: "어떤 사용자에게 무엇이 추천되었나" (필수: top-K, 옵션: rich signals)
- CatalogSource: "아이템 메타·인기도·사용자 히스토리" (보조 신호)

본체 코드를 import 하지 않는다. CSV/parquet artifact만 소비한다.

기본 구현:
- MockAdapter: 본체 없이도 동작하는 결정적 더미 데이터.
- CSVAdapter: 본체 submission CSV (user_id, item_id) 롱 포맷.
- ParquetCatalogSource: train.parquet에서 카탈로그·인기도·히스토리 집계.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import random
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class RecsysAdapter(Protocol):
    """본체 추천 결과를 제공하는 어댑터."""

    def list_user_ids(self) -> list[str]:
        ...

    def get_top_k(self, user_id: str) -> list[str]:
        """1-indexed 순위 순서대로 item_id 리스트 반환."""
        ...

    def get_rich_signals(self, user_id: str) -> dict | None:
        """{item_id: {"ranks": {model: rank}, "ensemble_rank": int, "cart_boosted": bool}} 또는 None."""
        ...


@runtime_checkable
class CatalogSource(Protocol):
    """카탈로그·인기도·사용자 히스토리·사용자 컨텍스트 제공자."""

    def get_catalog(self) -> dict[str, dict]:
        """{item_id: {"category_code", "brand", "price"}}."""
        ...

    def get_popularity(self) -> dict[str, dict]:
        """{item_id: {"item_pop_log1p", "item_recent14_pop_log1p"}}."""
        ...

    def get_user_history(self, user_id: str) -> dict[str, set]:
        """{"viewed": set, "carted": set, "purchased": set}."""
        ...

    def get_event_history(self, user_id: str) -> list[dict]:
        """[{"item_id", "event_type", "days_ago"}] — revisit 후보 생성용."""
        ...

    def get_user_context(self, user_id: str) -> dict:
        """UserContext schema-compatible dict."""
        ...


# ---------- MockAdapter (RecsysAdapter + CatalogSource) -----------------

CATEGORIES = [
    "apparel.shoes",
    "apparel.shoes.keds",
    "apparel.tshirt",
    "apparel.dress",
    "apparel.jumper",
    "apparel.skirt",
]
BRANDS = ["kapika", "respect", "goodloot", "baden", "rooman", "weekend", "rieker", "fassen"]
MODELS = ["tisasrec", "bsarec", "mbstr", "tifu_knn"]


def category_l2_of(category_code: str | None) -> str | None:
    if not category_code:
        return None
    parts = str(category_code).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(category_code)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


class MockAdapter:
    """본체 없이 동작하는 결정적 mock — RecsysAdapter + CatalogSource를 동시에 구현.

    seed가 같으면 결과가 항상 같다. 데모·테스트·CI에 사용.
    """

    def __init__(self, n_users: int = 50, n_items: int = 200, top_k: int = 10, seed: int = 42):
        rng = random.Random(seed)
        self._users = [f"mock-user-{i:04d}" for i in range(n_users)]
        self._items = [f"mock-item-{i:04d}" for i in range(n_items)]

        # 카탈로그
        self._catalog: dict[str, dict] = {}
        for item in self._items:
            category_code = rng.choice(CATEGORIES)
            self._catalog[item] = {
                "category_code": category_code,
                "category_l2": category_l2_of(category_code),
                "brand": rng.choice(BRANDS),
                "price": round(rng.uniform(15, 120), 2),
            }
        # 인기도
        self._pop: dict[str, dict] = {}
        for item in self._items:
            views = rng.randint(10, 500)
            carts = int(views * rng.uniform(0.0, 0.02))
            purchases = int(views * rng.uniform(0.0, 0.004))
            recent_views = int(views * rng.uniform(0.05, 0.5))
            prev_views = int(views * rng.uniform(0.05, 0.5))
            recent_all = recent_views + int(carts * 0.5) + int(purchases * 0.5)
            self._pop[item] = {
                "item_pop_log1p": math.log1p(views + carts + purchases),
                "item_recent14_pop_log1p": math.log1p(recent_all),
                "item_view_count": views,
                "item_cart_count": carts,
                "item_purchase_count": purchases,
                "item_v2c_smoothed": (carts + 50 * 0.002) / (views + 50),
                "item_v2p_smoothed": (purchases + 50 * 0.00025) / (views + 50),
                "item_recent14_view_count": recent_views,
                "item_prev14_view_count": prev_views,
                "item_recent_trend_ratio": (recent_views + 1.0) / (prev_views + 1.0),
            }

        # 사용자별 top-K + rich signals + history
        self._top: dict[str, list[str]] = {}
        self._signals: dict[str, dict[str, dict]] = {}
        self._history: dict[str, dict[str, set]] = {}
        self._event_history: dict[str, list[dict]] = {}
        for u in self._users:
            seed_int = int(hashlib.md5(u.encode()).hexdigest()[:8], 16)
            urng = random.Random(seed_int)
            top = urng.sample(self._items, min(top_k, len(self._items)))
            self._top[u] = top

            sig: dict[str, dict] = {}
            for rank, item in enumerate(top, 1):
                per_model: dict[str, int] = {}
                for m in MODELS:
                    if urng.random() < 0.7:
                        per_model[m] = urng.randint(1, 10)
                sig[item] = {
                    "ranks": per_model,
                    "ensemble_rank": rank,
                    "cart_boosted": urng.random() < 0.1,
                }
            self._signals[u] = sig

            event_pool = list(dict.fromkeys(top + urng.sample(self._items, min(20, len(self._items)))))
            n_events = urng.randint(5, 15)
            events: list[dict] = []

            # revisit 데모가 항상 보이도록 carted-but-not-purchased 패턴 1~2개 보장.
            must_cart = urng.sample(event_pool, min(urng.randint(1, 2), len(event_pool)))
            for item in must_cart:
                events.append({"item_id": item, "event_type": "cart", "days_ago": urng.randint(1, 7)})

            random_pool = [item for item in event_pool if item not in set(must_cart)] or event_pool
            while len(events) < n_events:
                item = urng.choice(random_pool)
                p = urng.random()
                if p < 0.72:
                    event_type = "view"
                    days_ago = urng.randint(0, 45)
                elif p < 0.90:
                    event_type = "cart"
                    days_ago = urng.randint(1, 35)
                else:
                    event_type = "purchase"
                    days_ago = urng.randint(8, 60)
                events.append({"item_id": item, "event_type": event_type, "days_ago": days_ago})

            carted = {ev["item_id"] for ev in events if ev["event_type"] == "cart"}
            purchased = {ev["item_id"] for ev in events if ev["event_type"] == "purchase"}
            for item in must_cart:
                events = [
                    ev for ev in events
                    if not (ev["item_id"] == item and ev["event_type"] == "purchase")
                ]
            purchased = {ev["item_id"] for ev in events if ev["event_type"] == "purchase"}
            viewed = {ev["item_id"] for ev in events if ev["event_type"] == "view"}
            self._history[u] = {"viewed": viewed, "carted": carted, "purchased": purchased}
            self._event_history[u] = sorted(events, key=lambda ev: ev["days_ago"], reverse=True)

    # ---- RecsysAdapter ----
    def list_user_ids(self) -> list[str]:
        return list(self._users)

    def get_top_k(self, user_id: str) -> list[str]:
        return list(self._top.get(user_id, []))

    def get_rich_signals(self, user_id: str) -> dict | None:
        return self._signals.get(user_id)

    # ---- CatalogSource ----
    def get_catalog(self) -> dict[str, dict]:
        return self._catalog

    def get_popularity(self) -> dict[str, dict]:
        return self._pop

    def get_user_history(self, user_id: str) -> dict[str, set]:
        h = self._history.get(user_id, {})
        return {k: set(v) for k, v in h.items()} or {"viewed": set(), "carted": set(), "purchased": set()}

    def get_event_history(self, user_id: str) -> list[dict]:
        return [dict(ev) for ev in self._event_history.get(user_id, [])]

    def get_user_context(self, user_id: str) -> dict:
        h = self._history.get(user_id, {})
        viewed = h.get("viewed", set())
        carted = h.get("carted", set())
        purchased = h.get("purchased", set())
        total = len(viewed) + len(carted) + len(purchased)
        top_cats = list({self._catalog[i]["category_l2"] for i in viewed if i in self._catalog})[:3]
        top_brands = list({self._catalog[i]["brand"] for i in viewed if i in self._catalog})[:3]
        carted_brands = list({self._catalog[i]["brand"] for i in carted if i in self._catalog})
        purchased_brands = list({self._catalog[i]["brand"] for i in purchased if i in self._catalog})
        prices = [float(self._catalog[i]["price"]) for i in viewed if i in self._catalog and self._catalog[i].get("price") is not None]
        p25 = _quantile(prices, 0.25)
        p75 = _quantile(prices, 0.75)
        return {
            "total_events": total,
            "recent14_events": int(total * 0.2),
            "top_categories": [c for c in top_cats if c],
            "top_brands": top_brands,
            "carted_brands": carted_brands,
            "purchased_brands": purchased_brands,
            "price_median": _quantile(prices, 0.5),
            "price_p25": p25,
            "price_p75": p75,
            "price_iqr": (p75 - p25) if p25 is not None and p75 is not None else None,
        }


# ---------- CSVAdapter (RecsysAdapter from submission CSV) ---------------


class CSVAdapter:
    """본체 submission CSV (user_id, item_id) 롱 포맷에서 top-K를 읽는 어댑터.

    rich signals는 CSV만으론 알 수 없으므로 None.
    """

    def __init__(self, submission_csv: str | Path, top_k: int = 10, max_users: int | None = None):
        import pandas as pd

        read_kwargs = {"dtype": {"user_id": str, "item_id": str}}
        if max_users is not None:
            read_kwargs["nrows"] = max(max_users, 0) * top_k
        df = pd.read_csv(submission_csv, **read_kwargs)
        cols = set(df.columns)
        if not {"user_id", "item_id"}.issubset(cols):
            raise ValueError(
                f"submission CSV에 user_id, item_id 컬럼 필요. 받은 컬럼: {sorted(cols)}"
            )
        self._top: dict[str, list[str]] = {}
        for uid, group in df.groupby("user_id", sort=False):
            self._top[str(uid)] = group["item_id"].astype(str).tolist()[:top_k]

    def list_user_ids(self) -> list[str]:
        return list(self._top.keys())

    def get_top_k(self, user_id: str) -> list[str]:
        return list(self._top.get(user_id, []))

    def get_rich_signals(self, user_id: str) -> dict | None:
        return None


# ---------- ParquetCatalogSource (CatalogSource from train.parquet) ------


class ParquetCatalogSource:
    """train.parquet에서 카탈로그·인기도·사용자 히스토리·컨텍스트를 집계.

    `user_ids`/`item_ids`를 넘기면 해당 Evidence Pack 빌드에 필요한 부분만
    materialize한다. 전체 빌드에서는 `cache_path`로 집계 결과를 재사용할 수 있다.
    """

    CACHE_VERSION = 3

    def __init__(
        self,
        train_parquet: str | Path,
        recent_days: int = 14,
        user_ids: set[str] | list[str] | None = None,
        item_ids: set[str] | list[str] | None = None,
        cache_path: str | Path | None = None,
        conversion_alpha: float = 50.0,
    ):
        self._cache_path = Path(cache_path) if cache_path else None
        use_cache = self._cache_path is not None and user_ids is None and item_ids is None
        if use_cache and self._cache_path.exists():
            self._load_cache(self._cache_path)
            return

        import pandas as pd

        cols = [
            "user_id", "item_id", "event_time", "category_code", "brand", "price", "event_type",
        ]
        user_filter = set(map(str, user_ids)) if user_ids is not None else None
        item_filter = set(map(str, item_ids)) if item_ids is not None else None
        partial = user_filter is not None or item_filter is not None

        if partial:
            user_df = self._prepare_events(
                self._read_filtered_parquet(train_parquet, cols, "user_id", user_filter or set())
            )
            item_df = self._prepare_events(
                self._read_filtered_parquet(train_parquet, cols, "item_id", item_filter or set())
            )
            item_side_df = pd.concat([item_df, user_df], ignore_index=True)
            prior_df = item_side_df
            max_time = self._parquet_max_time(train_parquet) or self._max_event_time(user_df, item_side_df)
        else:
            df = self._prepare_events(pd.read_parquet(train_parquet, columns=cols))
            user_df = df
            item_side_df = df
            prior_df = df
            max_time = df["event_time"].max()

        if max_time is None or pd.isna(max_time):
            max_time = pd.Timestamp.now(tz="UTC")
        cutoff = max_time - pd.Timedelta(days=recent_days)
        prev_cutoff = max_time - pd.Timedelta(days=recent_days * 2)

        self._build_item_side(item_side_df, prior_df, cutoff, prev_cutoff, conversion_alpha)
        self._build_user_side(user_df, cutoff)

        if use_cache:
            self._save_cache(self._cache_path)

    @staticmethod
    def _prepare_events(df):
        import pandas as pd

        for col in ("user_id", "item_id", "event_time", "category_code", "brand", "price", "event_type"):
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        df = df.copy()
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
        df["category_l2"] = df["category_code"].map(category_l2_of)
        try:
            from pipeline.brand_remap import remap_brand_series
            df["brand"] = remap_brand_series(df["brand"])
        except ImportError:
            pass
        return df

    @staticmethod
    def _read_filtered_parquet(train_parquet: str | Path, columns: list[str], field: str, values: set[str]):
        import pandas as pd

        if not values:
            return pd.DataFrame(columns=columns)
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(str(train_parquet), format="parquet")
            table = dataset.to_table(columns=columns, filter=ds.field(field).isin(list(values)))
            return table.to_pandas()
        except Exception:
            df = pd.read_parquet(train_parquet, columns=columns)
            return df[df[field].astype(str).isin(values)].copy()

    @staticmethod
    def _parquet_max_time(train_parquet: str | Path):
        import pandas as pd

        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(train_parquet)
            field_idx = pf.schema_arrow.get_field_index("event_time")
            if field_idx < 0:
                return None
            candidates = []
            for i in range(pf.num_row_groups):
                stats = pf.metadata.row_group(i).column(field_idx).statistics
                if stats is not None and stats.max is not None:
                    value = stats.max.decode() if isinstance(stats.max, bytes) else stats.max
                    candidates.append(pd.to_datetime(value, utc=True))
            return max(candidates) if candidates else None
        except Exception:
            return None

    @staticmethod
    def _max_event_time(*frames):
        import pandas as pd

        candidates = []
        for df in frames:
            if df is not None and not df.empty and "event_time" in df.columns:
                value = df["event_time"].max()
                if not pd.isna(value):
                    candidates.append(value)
        return max(candidates) if candidates else None

    def _build_item_side(self, item_df, all_df, cutoff, prev_cutoff, conversion_alpha: float) -> None:
        if item_df.empty:
            self._catalog = {}
            self._pop = {}
            return

        def _mode_or_none(s):
            s = s.dropna()
            return s.mode().iat[0] if not s.empty else None

        cat = item_df.groupby("item_id").agg(
            category_code=("category_code", _mode_or_none),
            category_l2=("category_l2", _mode_or_none),
            brand=("brand", _mode_or_none),
            price=("price", "mean"),
        )
        self._catalog = {
            str(k): {kk: v for kk, v in row.items()}
            for k, row in cat.to_dict("index").items()
        }

        item_ev = item_df.groupby(["item_id", "event_type"]).size().unstack(fill_value=0)
        for col in ("view", "cart", "purchase"):
            if col not in item_ev.columns:
                item_ev[col] = 0

        global_views = max(int((all_df["event_type"] == "view").sum()), 1)
        global_carts = int((all_df["event_type"] == "cart").sum())
        global_purchases = int((all_df["event_type"] == "purchase").sum())
        prior_v2c = global_carts / global_views
        prior_v2p = global_purchases / global_views

        pop_all = item_df.groupby("item_id").size()
        pop_recent = item_df[item_df["event_time"] >= cutoff].groupby("item_id").size()
        recent_views = item_df[
            (item_df["event_type"] == "view") & (item_df["event_time"] >= cutoff)
        ].groupby("item_id").size()
        prev_views = item_df[
            (item_df["event_type"] == "view")
            & (item_df["event_time"] >= prev_cutoff)
            & (item_df["event_time"] < cutoff)
        ].groupby("item_id").size()

        self._pop = {}
        for item, n in pop_all.items():
            item = str(item)
            views = int(item_ev.at[item, "view"]) if item in item_ev.index else 0
            carts = int(item_ev.at[item, "cart"]) if item in item_ev.index else 0
            purchases = int(item_ev.at[item, "purchase"]) if item in item_ev.index else 0
            recent_v = int(recent_views.get(item, 0))
            prev_v = int(prev_views.get(item, 0))
            self._pop[item] = {
                "item_pop_log1p": math.log1p(int(n)),
                "item_recent14_pop_log1p": math.log1p(int(pop_recent.get(item, 0))),
                "item_view_count": views,
                "item_cart_count": carts,
                "item_purchase_count": purchases,
                "item_v2c_smoothed": (carts + conversion_alpha * prior_v2c) / (views + conversion_alpha),
                "item_v2p_smoothed": (purchases + conversion_alpha * prior_v2p) / (views + conversion_alpha),
                "item_recent14_view_count": recent_v,
                "item_prev14_view_count": prev_v,
                "item_recent_trend_ratio": (recent_v + 1.0) / (prev_v + 1.0),
            }

    def _build_user_side(self, user_df, cutoff) -> None:
        if user_df.empty:
            self._viewed = {}
            self._carted = {}
            self._purchased = {}
            self._user_total = {}
            self._user_recent = {}
            self._top_cats = {}
            self._top_brands = {}
            self._carted_brands = {}
            self._purchased_brands = {}
            self._price_ctx = {}
            self._event_history = {}
            return

        max_time = user_df["event_time"].max()
        event_history: dict[str, list[dict]] = {}
        event_df = user_df[user_df["event_type"].isin(["view", "cart", "purchase"])]
        for uid, group in event_df.sort_values(["user_id", "event_time"]).groupby("user_id", sort=False):
            events = []
            for row in group.itertuples(index=False):
                days = int(max((max_time - row.event_time).days, 0))
                events.append({
                    "item_id": str(row.item_id),
                    "event_type": str(row.event_type),
                    "days_ago": days,
                })
            event_history[str(uid)] = events
        self._event_history = event_history

        view_events = user_df[user_df["event_type"] == "view"]
        viewed = view_events.groupby("user_id")["item_id"].apply(set).to_dict()
        carted = user_df[user_df["event_type"] == "cart"].groupby("user_id")["item_id"].apply(set).to_dict()
        purchased = user_df[user_df["event_type"] == "purchase"].groupby("user_id")["item_id"].apply(set).to_dict()
        self._viewed = {str(k): set(map(str, v)) for k, v in viewed.items()}
        self._carted = {str(k): set(map(str, v)) for k, v in carted.items()}
        self._purchased = {str(k): set(map(str, v)) for k, v in purchased.items()}

        self._user_total = {str(k): int(v) for k, v in user_df.groupby("user_id").size().items()}
        recent = user_df[user_df["event_time"] >= cutoff].groupby("user_id").size()
        self._user_recent = {str(k): int(v) for k, v in recent.items()}

        preference_df = view_events if not view_events.empty else user_df

        def _top_n(col: str, n: int = 3) -> dict[str, list[str]]:
            cnt = preference_df.dropna(subset=[col]).groupby(["user_id", col]).size().reset_index(name="cnt")
            if cnt.empty:
                return {}
            cnt = cnt.sort_values(["user_id", "cnt"], ascending=[True, False])
            out = cnt.groupby("user_id")[col].apply(lambda s: [str(x) for x in s.head(n).tolist()]).to_dict()
            return {str(k): v for k, v in out.items()}

        self._top_cats = _top_n("category_l2")
        self._top_brands = _top_n("brand")
        carted_df = user_df[user_df["event_type"] == "cart"]
        purchased_df = user_df[user_df["event_type"] == "purchase"]
        self._carted_brands = {
            str(k): [str(x) for x in v]
            for k, v in carted_df.groupby("user_id")["brand"].apply(lambda s: list(s.dropna().unique())[:5]).to_dict().items()
        }
        self._purchased_brands = {
            str(k): [str(x) for x in v]
            for k, v in purchased_df.groupby("user_id")["brand"].apply(lambda s: list(s.dropna().unique())[:5]).to_dict().items()
        }

        view_df = view_events.dropna(subset=["price"])
        price_stats = view_df.groupby("user_id")["price"].agg(
            price_median="median",
            price_p25=lambda x: x.quantile(0.25),
            price_p75=lambda x: x.quantile(0.75),
        )
        price_stats["price_iqr"] = price_stats["price_p75"] - price_stats["price_p25"]
        self._price_ctx = {
            str(k): {kk: (float(v) if v == v else None) for kk, v in row.items()}
            for k, row in price_stats.to_dict("index").items()
        }

    def _save_cache(self, path: Path) -> None:
        payload = {name: getattr(self, name) for name in (
            "_catalog", "_pop", "_viewed", "_carted", "_purchased", "_user_total",
            "_user_recent", "_top_cats", "_top_brands", "_carted_brands",
            "_purchased_brands", "_price_ctx", "_event_history",
        )}
        payload["_cache_version"] = self.CACHE_VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cache(self, path: Path) -> None:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("_cache_version") != self.CACHE_VERSION:
            raise ValueError(f"catalog cache version mismatch: {path}")
        for k, v in payload.items():
            if k != "_cache_version":
                setattr(self, k, v)

    def get_catalog(self) -> dict[str, dict]:
        return self._catalog

    def get_popularity(self) -> dict[str, dict]:
        return self._pop

    def get_user_history(self, user_id: str) -> dict[str, set]:
        return {
            "viewed": set(self._viewed.get(user_id, set())),
            "carted": set(self._carted.get(user_id, set())),
            "purchased": set(self._purchased.get(user_id, set())),
        }

    def get_event_history(self, user_id: str) -> list[dict]:
        return [dict(ev) for ev in self._event_history.get(user_id, [])]

    def get_user_context(self, user_id: str) -> dict:
        price_ctx = self._price_ctx.get(user_id, {})
        return {
            "total_events": int(self._user_total.get(user_id, 0)),
            "recent14_events": int(self._user_recent.get(user_id, 0)),
            "top_categories": [c for c in self._top_cats.get(user_id, []) if c],
            "top_brands": [b for b in self._top_brands.get(user_id, []) if b],
            "carted_brands": [b for b in self._carted_brands.get(user_id, []) if b],
            "purchased_brands": [b for b in self._purchased_brands.get(user_id, []) if b],
            "price_median": price_ctx.get("price_median"),
            "price_p25": price_ctx.get("price_p25"),
            "price_p75": price_ctx.get("price_p75"),
            "price_iqr": price_ctx.get("price_iqr"),
        }


if __name__ == "__main__":
    # 동작 데모 — Mock 어댑터만으로 끝까지 호출 가능
    a = MockAdapter(n_users=3, n_items=20, top_k=5, seed=0)
    print("users:", a.list_user_ids())
    u0 = a.list_user_ids()[0]
    print(f"top-K for {u0}:", a.get_top_k(u0))
    print("rich signals (1개 아이템):", list(a.get_rich_signals(u0).items())[:1])
    print("user_context:", a.get_user_context(u0))
