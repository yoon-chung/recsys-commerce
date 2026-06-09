"""evidence_pack — 추천 결과 + 카탈로그·히스토리를 Evidence Pack JSON으로 빌드."""

from __future__ import annotations

_EXPORTS = {
    "EvidencePack": (".schema", "EvidencePack"),
    "Recommendation": (".schema", "Recommendation"),
    "ItemSignals": (".schema", "ItemSignals"),
    "UserContext": (".schema", "UserContext"),
    "RecsysAdapter": (".adapter", "RecsysAdapter"),
    "CatalogSource": (".adapter", "CatalogSource"),
    "MockAdapter": (".adapter", "MockAdapter"),
    "CSVAdapter": (".adapter", "CSVAdapter"),
    "ParquetCatalogSource": (".adapter", "ParquetCatalogSource"),
    "build_evidence_pack": (".builder", "build_evidence_pack"),
    "iter_evidence_pack": (".builder", "iter_evidence_pack"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attr = _EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
