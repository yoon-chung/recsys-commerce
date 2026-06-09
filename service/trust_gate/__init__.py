"""trust_gate — LLM 응답을 거르는 3중 게이트."""

from __future__ import annotations

_EXPORTS = {
    "check_response": (".hard_gate", "check_response"),
    "get_evidence_value": (".hard_gate", "get_evidence_value"),
    "self_check_consistency": (".self_check", "self_check_consistency"),
    "apply_self_check": (".self_check", "apply_self_check"),
    "Calibrator": (".calibration", "Calibrator"),
    "expected_calibration_error": (".calibration", "expected_calibration_error"),
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
