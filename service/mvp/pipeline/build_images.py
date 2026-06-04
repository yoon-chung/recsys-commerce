"""build_images.py — Pexels API로 카테고리별 stock photo URL 캐시 생성.

사용법:
    # .env에 PEXELS_API_KEY 설정 후
    python -m pipeline.build_images --catalog data/item_catalog.json --out data/item_images.json

동작:
    1. catalog의 모든 (L2, L3) 카테고리 조합 수집
    2. 각 카테고리 → Pexels 검색 키워드 매핑 → 10장 후보 fetch
    3. item_alias마다 해시로 결정적 1장 선택
    4. 결과: {item_alias: image_url} JSON 저장

비용: Pexels free tier 200 req/hr, 20K/month. 우리 사용 ~30 unique queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


# L3 (구체적) 우선, 없으면 L2로 폴백
CATEGORY_KEYWORDS: dict[str, str] = {
    # shoes 계열
    "shoes":     "shoes fashion",
    "slipons":   "slip-on shoes",
    "sandals":   "sandals fashion",
    "keds":      "sneakers white shoes",
    "ergot":     "casual shoes",     # remap한 brand
    # apparel
    "shirt":     "men shirt fashion",
    "tshirt":    "t-shirt apparel",
    "jeans":     "jeans denim fashion",
    "jumper":    "sweater jumper apparel",
    "jacket":    "jacket fashion",
    "costume":   "costume outfit",
    "scarf":     "scarf accessory",
    "underwear": "underwear apparel",
    "pajamas":   "pajamas sleepwear",
    "shorts":    "shorts apparel",
    "dress":     "dress fashion",
    "trousers":  "trousers pants",
    "skirt":     "skirt fashion",
    "swimwear":  "swimwear beachwear",
    "coat":      "coat outerwear",
    "blouse":    "blouse women fashion",
    # 기타 일반
    "apparel":   "apparel clothing fashion",
}

PEXELS_API = "https://api.pexels.com/v1/search"


def _resolve_keyword(category_l2: str | None, category_code: str | None) -> str:
    """L3 (마지막 segment) → L2 → fallback 순으로 keyword 매핑."""
    # category_code: "apparel.shoes.slipons" → L3 = "slipons"
    if category_code:
        parts = str(category_code).split(".")
        if len(parts) >= 3 and parts[2] in CATEGORY_KEYWORDS:
            return CATEGORY_KEYWORDS[parts[2]]
        if len(parts) >= 2 and parts[1] in CATEGORY_KEYWORDS:
            return CATEGORY_KEYWORDS[parts[1]]
    if category_l2:
        l2_last = str(category_l2).split(".")[-1]
        if l2_last in CATEGORY_KEYWORDS:
            return CATEGORY_KEYWORDS[l2_last]
    return CATEGORY_KEYWORDS["apparel"]


def fetch_pexels(keyword: str, api_key: str, per_page: int = 10) -> list[str]:
    """Pexels API 호출 → image URL 리스트 반환."""
    import requests

    r = requests.get(
        PEXELS_API,
        headers={"Authorization": api_key},
        params={"query": keyword, "per_page": per_page, "size": "small"},
        timeout=15,
    )
    if r.status_code != 200:
        print(f"  [WARN] {keyword} → HTTP {r.status_code}")
        return []
    photos = r.json().get("photos", [])
    return [p["src"]["medium"] for p in photos]


def _pick_for_alias(alias: str, urls: list[str]) -> str | None:
    """item_alias 해시로 결정적 선택."""
    if not urls:
        return None
    idx = int(hashlib.md5(alias.encode()).hexdigest(), 16) % len(urls)
    return urls[idx]


def build_images(
    catalog: dict,
    api_key: str,
    sleep_sec: float = 0.2,
) -> dict[str, str]:
    """카테고리별 stock photo fetch → item_alias마다 1 URL 매핑."""

    # 1. 카테고리 unique 키워드 수집
    keyword_for_alias: dict[str, str] = {}
    unique_keywords: set[str] = set()
    for alias, meta in catalog.items():
        kw = _resolve_keyword(meta.get("category_l2"), meta.get("category_code"))
        keyword_for_alias[alias] = kw
        unique_keywords.add(kw)

    print(f"unique keywords to fetch: {len(unique_keywords)}")

    # 2. keyword → URL 리스트 fetch (1회만)
    kw_to_urls: dict[str, list[str]] = {}
    for i, kw in enumerate(sorted(unique_keywords), 1):
        print(f"  [{i}/{len(unique_keywords)}] {kw}", end=" ... ", flush=True)
        urls = fetch_pexels(kw, api_key)
        kw_to_urls[kw] = urls
        print(f"got {len(urls)} urls")
        time.sleep(sleep_sec)  # rate limit 여유

    # 3. item_alias마다 결정적 선택
    image_map: dict[str, str] = {}
    for alias, kw in keyword_for_alias.items():
        url = _pick_for_alias(alias, kw_to_urls.get(kw, []))
        if url:
            image_map[alias] = url
    return image_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pexels stock photo URL cache 빌드")
    parser.add_argument("--catalog", default="data/item_catalog.json")
    parser.add_argument("--out", default="data/item_images.json")
    parser.add_argument("--api-key", default=None, help=".env PEXELS_API_KEY 사용 (생략 시 env에서 읽음)")
    args = parser.parse_args(argv)

    # .env 자동 로드 (있으면)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = args.api_key or os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("ERROR: PEXELS_API_KEY가 없습니다.")
        print("https://www.pexels.com/api/ 에서 발급 후 .env에 PEXELS_API_KEY=... 추가하세요.")
        return 1

    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"catalog: {len(catalog):,} items")

    image_map = build_images(catalog, api_key)
    print(f"image_map: {len(image_map):,} items mapped")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(image_map, f, ensure_ascii=False)
    print(f"→ {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())