"""Calibration — confidence_raw가 실제 적중률에 맞도록 보정.

원 논문: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
핵심: 현대 모델은 confidence를 과신하는 경향 → temperature scaling으로 logit을 1/T로 나눠 완화.

여기서 우리는:
- ECE(Expected Calibration Error)로 보정 전/후 품질 측정.
- Temperature scaling을 1차원 최적화로 학습 (scipy.optimize).
- 학습된 T를 JSON으로 저장/로드 (UI 런타임은 transform만 사용).

라벨 정의: golden set에서 LLM의 응답이 실제로 "근거에 의해 supported"였는지 (1) 아닌지 (0).
즉 자동 hard_gate 통과 + (golden) 수작업 검토를 통과한 응답이 양의 라벨.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EPS = 1e-7


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def expected_calibration_error(
    confidences,
    labels,
    n_bins: int = 10,
) -> float:
    """ECE — bin별 |평균 정확도 - 평균 confidence|의 가중평균."""
    conf = _to_array(confidences)
    lab = _to_array(labels)
    if len(conf) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(lab[mask].mean() - conf[mask].mean())
    return float(ece)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_temperature(confidences: np.ndarray, labels: np.ndarray) -> float:
    """NLL을 최소화하는 T를 찾아 반환. scipy 없으면 grid search 폴백."""
    logits = _logit(confidences)

    def nll(T: float) -> float:
        if T <= 0:
            return 1e10
        p = _sigmoid(logits / T)
        p = np.clip(p, EPS, 1.0 - EPS)
        return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    try:
        from scipy.optimize import minimize_scalar  # type: ignore
        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        return float(result.x)
    except ImportError:
        # 폴백: 간단한 grid search
        grid = np.linspace(0.1, 5.0, 50)
        return float(grid[np.argmin([nll(T) for T in grid])])


class Calibrator:
    """Temperature scaling 캘리브레이터. fit → transform → save/load."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, confidences, labels) -> "Calibrator":
        conf = _to_array(confidences)
        lab = _to_array(labels)
        if len(conf) < 3:
            # 표본 너무 적으면 보정 안 함
            self.temperature = 1.0
            return self
        self.temperature = _fit_temperature(conf, lab)
        return self

    def transform(self, confidence) -> float:
        """단일 confidence (또는 배열)을 보정해 반환."""
        if isinstance(confidence, (list, tuple, np.ndarray)):
            arr = _to_array(confidence)
            return _sigmoid(_logit(arr) / self.temperature).tolist()
        return float(_sigmoid(_logit(np.array([confidence])) / self.temperature)[0])

    def ece(self, confidences, labels, n_bins: int = 10) -> float:
        return expected_calibration_error(confidences, labels, n_bins=n_bins)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"temperature": self.temperature}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        p = Path(path)
        if not p.exists():
            return cls(temperature=1.0)
        data = json.loads(p.read_text())
        return cls(temperature=float(data.get("temperature", 1.0)))


if __name__ == "__main__":
    # 데모: 합성 데이터로 fit → ECE 감소 확인
    rng = np.random.default_rng(42)
    n = 200
    # 과신 모델 시뮬: confidence가 실제보다 높음
    true_p = rng.beta(2, 5, size=n)         # 실제 P(correct)
    confidences = np.clip(true_p + 0.2, 0.01, 0.99)  # 과신
    labels = (rng.uniform(size=n) < true_p).astype(float)

    cal = Calibrator()
    ece_before = cal.ece(confidences, labels)
    cal.fit(confidences, labels)
    calibrated = np.asarray(cal.transform(confidences.tolist()))
    ece_after = cal.ece(calibrated, labels)
    print(f"T = {cal.temperature:.3f}")
    print(f"ECE before = {ece_before:.4f}")
    print(f"ECE after  = {ece_after:.4f}")
    print(f"sample: raw={confidences[0]:.3f} -> calibrated={calibrated[0]:.3f}")
