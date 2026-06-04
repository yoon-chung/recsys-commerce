"""brand_remap.py — 데이터셋의 brand 라벨 → demo용 가공 fashion 브랜드명 매핑.

이 데이터(Rees46 계열)의 brand 라벨에 apple/samsung/sony 같은 electronics 네임이 섞여 있어
apparel.* 카테고리와 부조화. demo 시청자가 코드 버그로 오해할 위험.

해결: 인지도 높은 electronics 브랜드만 fictional fashion 브랜드명으로 1회 치환.
나머지(nike, adidas, chanel, defacto 등 fashion 브랜드 또는 무명 brand)는 그대로 통과.

이 매핑은 **offline 1회 적용**되며 id_alias / user_profile / build_catalog 모두 거침.
원본 brand는 보존되지 않음 (재현 가능성은 이 파일이 source of truth).
"""

from __future__ import annotations


# 가공 fashion 브랜드명 — 비슷한 음운 + 패션 톤
# 원본 → 가공  (모두 lowercase)
BRAND_REMAP: dict[str, str] = {
    # 전자제품 브랜드 → 가공 fashion 명
    "sony":       "solaris",
    "xiaomi":     "xanadu",
    "samsung":    "sandara",
    "apple":      "aurora",
    "huawei":     "havana",
    "intel":      "inteo",
    "lg":         "lumen",
    "amd":        "amada",
    "intex":      "inteno",
    "omron":      "orelle",
    "kingston":   "kinsley",
    "asrock":     "astra",
    "asus":       "aurio",
    "microsoft":  "mistral",
    "gigabyte":   "giselle",
    "bq":         "bristol",
    "prestigio":  "prestige",
    "lenovo":     "lenora",
    "yamaha":     "yanmar",
    "bosch":      "boscale",
    "gopro":      "galante",
    "garmin":     "garner",
    "kingmax":    "kinmax",
    "philips":    "philine",
    "dji":        "delos",
    "nikon":      "noir",
    "canon":      "candor",
    "panasonic":  "pasera",
    "epson":      "elson",
    "lego":       "lago",
    "kodak":      "korin",
    "razer":      "rayon",
    "ergo":       "ergot",
    "matrix":     "matera",
    "starline":   "starlit",   # 이름이 자동차 부품 같음
    "defender":   "delfin",     # 이름이 자동차 같음
}


def remap_brand(brand: str | None) -> str | None:
    """단일 brand 변환. None/빈값은 그대로."""
    if brand is None or brand == "" or brand != brand:  # NaN 방어
        return brand
    return BRAND_REMAP.get(str(brand).lower(), brand)


def remap_brand_series(series):
    """pandas Series 변환 (벡터화)."""
    return series.map(lambda b: remap_brand(b))


__all__ = ["BRAND_REMAP", "remap_brand", "remap_brand_series"]