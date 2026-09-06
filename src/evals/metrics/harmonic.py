"""Harmonic-mean aggregations from OpenUnlearning (paper §4.3, §F.1, Table 3/6).

2D: Robustness = HM(R, Q); Overall = HM(Faithfulness, Robustness);
    Table 6 / model selection Agg = HM(Mem, Utility).
3D: Table 3 Agg = HM(Mem, Priv, Utility).
"""

from typing import Iterable


def harmonic_mean(values: Iterable[float], n: int | None = None) -> float:
    """Harmonic mean. A zero term yields 0 (Init Mem zeros Table 3 Agg)."""
    xs = [float(v) for v in values]
    if n is not None and len(xs) != n:
        raise ValueError(f"expected {n} terms for harmonic mean, got {len(xs)}")
    if not xs:
        raise ValueError("harmonic mean requires at least one value")
    if any(x < 0 for x in xs):
        raise ValueError("harmonic mean is undefined for negative values")
    if any(x == 0 for x in xs):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def hm_2d(a: float, b: float) -> float:
    """Two-term HM: Table 6 Agg = HM(Mem, Utility); Robustness = HM(R, Q)."""
    return harmonic_mean((a, b), n=2)


def hm_3d(a: float, b: float, c: float) -> float:
    """Three-term HM: Table 3 Agg = HM(Mem, Priv, Utility)."""
    return harmonic_mean((a, b, c), n=3)


def aggregate_method_scores(
    mem: float,
    utility: float,
    priv: float | None = None,
    mode: str = "table3",
) -> float:
    """Paper ranking aggregates.

    * ``table3`` / ``3d``: HM(Mem, Priv, Utility) — overall method ranking.
    * ``table6`` / ``2d``: HM(Mem, Utility) — model selection without privacy.
    """
    if mode in ("table3", "3d"):
        if priv is None:
            raise ValueError("table3 / 3d aggregation requires a privacy score")
        return hm_3d(mem, priv, utility)
    if mode in ("table6", "2d"):
        return hm_2d(mem, utility)
    raise ValueError(f"unknown aggregation mode: {mode}")
