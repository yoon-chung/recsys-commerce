"""brand_remap.py — 원본 brand 라벨을 그대로 통과시키는 no-op (의도적).

이 데이터셋은 brand-category 라벨이 부조화하게 매겨진 sample(`apple`이 `apparel.shoes`에 붙는 등).
초기엔 demo 시청자가 코드 버그로 오해할까봐 가공 fashion 브랜드로 치환했으나,
(a) 가상 브랜드 자체가 reviewer 의심을 부르고, (b) 데이터 messiness를 다루는 모습을 보이는 게
엔지니어링 시그널로 더 강하다고 판단해 매핑 비움. 함수 시그니처는 유지 (downstream import 호환).
"""

from __future__ import annotations


# 의도적으로 비움 — 원본 brand 그대로 통과.
# (과거 매핑은 git history 참조)
BRAND_REMAP: dict[str, str] = {
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