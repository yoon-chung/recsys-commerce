"""eval — 오프라인 회귀 채점."""

from __future__ import annotations

_EXPORTS = {
    "GoldenItem": (".golden_set", "GoldenItem"),
    "iter_golden_set": (".golden_set", "iter_golden_set"),
    "save_golden_set": (".golden_set", "save_golden_set"),
    "make_starter_golden_set": (".golden_set", "make_starter_golden_set"),
    "JudgeScore": (".judge", "JudgeScore"),
    "MockJudge": (".judge", "MockJudge"),
    "LLMJudge": (".judge", "LLMJudge"),
    "get_judge": (".judge", "get_judge"),
    "regression_run": (".judge", "regression_run"),
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
