"""Toy synthetic tests for the ``ks_statistic`` and ``w1_distance`` metrics.

WARNING: every number in this file comes from SYNTHETIC toy data, NOT from a real
checkpoint. Per the task's data-integrity rules, synthetic data must be clearly
labelled as toy — these tests only validate the code path and the statistical
formulas, they produce no paper numbers.

Covered by these tests:
    1. ``ks_statistic`` returns the KS statistic D, NOT the p-value.
    2. The asymptotic relation ``p ~= 2 * exp(-2 * n_eff * D^2)`` holds, where
       ``n_eff = n1 * n2 / (n1 + n2)`` is read from the actual sample sizes that
       participate in the test and printed to stdout.
    3. ``w1_distance`` matches the closed-form 1-Wasserstein distance (equal-length
       samples: ``mean(|sorted_a - sorted_b|)``).
    4. Frozen retain-model truth-ratio logs are loaded through the framework's
       ``reference_logs`` / ``${eval.tofu.retain_logs_path}`` mechanism (the reward
       needs only a single-sided forward pass on the forget set).
    5. The new metric configs reference handlers that are registered in
       ``METRICS_REGISTRY``.
    6. Missing ``reference_logs`` degrades to ``{"agg_value": None}`` instead of
       fabricating a value.

References (主方案_v3_ES-MU_2026-08-28.md):
    - §7 Prop 2': KS is a sup-type functional (piecewise constant, dD = 0 a.e.);
      W1 is an integral-type functional (piecewise linear, has a gradient).
    - §8: the reward uses the KS statistic D (not the p-value).
    - §9: C-W1 first-order control group.
    - §11 事实 1-2: p ~= 2*exp(-2*n_eff*D^2); p is sample-size sensitive, use D.
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, wasserstein_distance

from evals.metrics import METRICS_REGISTRY


def _value_by_index(arr: np.ndarray) -> dict:
    """Build the ``{"idx": {"score": float}}`` dict used by ``truth_ratio`` outputs."""
    return {str(i): {"score": float(s)} for i, s in enumerate(arr)}


def _metric_kwargs(forget_tr: np.ndarray, retain_tr: np.ndarray) -> dict:
    """Assemble the kwargs dict a metric receives from the evaluation framework."""
    return {
        "pre_compute": {
            "forget": {"value_by_index": _value_by_index(forget_tr)},
        },
        "reference_logs": {
            "retain_model_logs": {
                "retain": {"value_by_index": _value_by_index(retain_tr)},
            },
        },
    }


def test_ks_statistic_returns_d_not_p() -> None:
    """The metric must return the KS statistic D, never the p-value."""
    rng = np.random.default_rng(20260829)
    forget_tr = rng.normal(0.0, 1.0, 500)
    retain_tr = rng.normal(0.1, 1.0, 500)

    metric = METRICS_REGISTRY["ks_statistic"]
    result = metric.evaluate_metric(
        model=None, metric_name="ks_statistic", **_metric_kwargs(forget_tr, retain_tr)
    )

    fq = ks_2samp(forget_tr, retain_tr, method="asymp")
    assert isinstance(result["agg_value"], float)
    assert np.isclose(result["agg_value"], float(fq.statistic))
    # D and the p-value must not coincide (this is what the ES-MU signal needs).
    assert not np.isclose(result["agg_value"], float(fq.pvalue))


def test_ks_statistic_asymptotic_p_formula() -> None:
    """Check ``p ~= 2 * exp(-2 * n_eff * D^2)`` with n_eff = n1*n2/(n1+n2).

    The Smirnov expansion is p = 2 * sum_k (-1)^(k-1) exp(-2 k^2 lambda^2) with
    lambda = D * sqrt(n_eff). The leading term dominates only when lambda is large,
    so the location shift is chosen (0.1) to push lambda well past 1 -- with the
    previous shift of 0.02, lambda landed near 0.8-2.0 and the guard below was
    borderline, making this test flaky across numpy/scipy versions.

    The actual sample sizes participating in the test are printed before the assertion.
    """
    rng = np.random.default_rng(20260829)
    n1, n2 = 20000, 20000
    forget_tr = rng.normal(0.0, 1.0, n1)
    retain_tr = rng.normal(0.1, 1.0, n2)

    fq = ks_2samp(forget_tr, retain_tr, method="asymp")
    n_eff = n1 * n2 / (n1 + n2)
    p_asym = 2.0 * np.exp(-2.0 * n_eff * fq.statistic**2)

    print(
        f"n1={n1} n2={n2} n_eff={n_eff:.1f} "
        f"D={fq.statistic:.4f} p={fq.pvalue:.3e} p_asym={p_asym:.3e}"
    )
    # Validity guard: the leading term dominates only when D * sqrt(n_eff) is large.
    assert fq.statistic * np.sqrt(n_eff) > 1.0
    assert np.isclose(fq.pvalue, p_asym, rtol=0.2)


def test_w1_distance_value() -> None:
    """Equal-length samples: W1 == mean(|sorted_a - sorted_b|) == scipy's value."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 300)
    b = rng.normal(0.5, 1.0, 300)

    metric = METRICS_REGISTRY["w1_distance"]
    result = metric.evaluate_metric(
        model=None, metric_name="w1_distance", **_metric_kwargs(a, b)
    )

    expected = float(np.mean(np.abs(np.sort(a) - np.sort(b))))
    assert np.isclose(result["agg_value"], expected)
    assert np.isclose(result["agg_value"], float(wasserstein_distance(a, b)))


def test_ks_statistic_loads_frozen_retain_logs(tmp_path) -> None:
    """Load pre-frozen retain TR via the ${eval.tofu.retain_logs_path} JSON mechanism."""
    rng = np.random.default_rng(11)
    forget_tr = rng.normal(0.0, 1.0, 60)
    retain_tr = rng.normal(0.3, 1.0, 60)

    logs_path = tmp_path / "retain_logs.json"
    logs_path.write_text(
        json.dumps(
            {
                # Exactly the structure produced by the truth_ratio evaluator.
                "forget_truth_ratio": {
                    "value_by_index": _value_by_index(retain_tr),
                }
            }
        )
    )

    metric = METRICS_REGISTRY["ks_statistic"]
    kwargs = {
        "reference_logs": {
            "retain_model_logs": {
                "path": str(logs_path),
                "include": {"forget_truth_ratio": {"access_key": "retain"}},
            }
        }
    }
    prepared = metric.prepare_kwargs_evaluate_metric(
        model=None, metric_name="ks_statistic", cache={}, **kwargs
    )
    # The real pre_compute requires a model forward pass; inject the forget side
    # manually (toy data) since no model is loaded in these unit tests.
    prepared["pre_compute"] = {"forget": {"value_by_index": _value_by_index(forget_tr)}}
    result = metric.evaluate_metric(model=None, metric_name="ks_statistic", **prepared)

    fq = ks_2samp(forget_tr, retain_tr, method="asymp")
    assert np.isclose(result["agg_value"], float(fq.statistic))


def test_configs_register_handlers() -> None:
    """Each new metric yaml must reference a handler present in METRICS_REGISTRY."""
    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "eval" / "tofu_metrics"
    for name in ("forget_quality_D", "w1_distance"):
        cfg = OmegaConf.load(cfg_dir / f"{name}.yaml")
        handler = cfg.handler
        assert handler in METRICS_REGISTRY
        assert METRICS_REGISTRY[handler].name == handler


def test_missing_reference_logs_returns_none() -> None:
    """Without retain_model_logs both metrics return None instead of fabricating a value."""
    forget_tr = np.array([0.1, 0.2, 0.3])
    kwargs = {"pre_compute": {"forget": {"value_by_index": _value_by_index(forget_tr)}}}

    for name in ("ks_statistic", "w1_distance"):
        metric = METRICS_REGISTRY[name]
        result = metric.evaluate_metric(model=None, metric_name=name, **kwargs)
        assert result["agg_value"] is None
