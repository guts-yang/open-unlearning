from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from evals.metrics.base import unlearning_metric, logger


@unlearning_metric(name="ks_test")
def ks_test(model, **kwargs):
    """Compare two forget and retain model distributions with a 2-sample KS-test and report the p-value.
    Used in the TOFU benchmark as forget_quality when computed over the truth_ratio statistic."""
    forget_tr_stats = np.array(
        [
            evals["score"]
            for evals in kwargs["pre_compute"]["forget"]["value_by_index"].values()
        ]
    )
    reference_logs = kwargs.get("reference_logs", None)
    if reference_logs:
        reference_logs = reference_logs["retain_model_logs"]
        retain_tr_stats = np.array(
            [
                evals["score"]
                for evals in reference_logs["retain"]["value_by_index"].values()
            ]
        )
        fq = ks_2samp(forget_tr_stats, retain_tr_stats)
        pvalue = fq.pvalue
    else:
        logger.warning(
            "retain_model_logs not provided in reference_logs, setting forget_quality to None"
        )
        pvalue = None
    return {"agg_value": pvalue}


@unlearning_metric(name="privleak")
def privleak(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic.
    To be used for MIA AUC scores in ensuring consistency and reproducibility of the MUSE benchmark.
    This function is similar to the rel_diff function below, but due to the MUSE benchmark reporting AUC
    scores as (1-x) when the more conventional way is x, we do adjustments here to our MIA AUC scores.
    calculations in the reverse way,"""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"retain_model_logs evals not provided for privleak, using default retain auc of {kwargs['ref_value']}"
        )
        ref = kwargs["ref_value"]
    score = 1 - score
    ref = 1 - ref
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


@unlearning_metric(name="rel_diff")
def rel_diff(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic."""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"retain_model_logs evals not provided for privleak, using default retain auc of {kwargs['ref_value']}"
        )
        ref = kwargs["ref_value"]
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


def _collect_tr_arrays(
    metric_name: str, **kwargs: Any
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract the (forget, retain) truth-ratio arrays shared by the distribution metrics.

    Factored out of ``ks_statistic`` and ``w1_distance``, which are otherwise byte-identical
    apart from the final statistic. Deliberately NOT wired into ``ks_test``: that metric is
    left untouched so this change stays purely additive (repo rule: extend by addition only).

    Args:
        metric_name: only used to name the metric in the missing-logs warning.
        **kwargs: the metric kwargs contract supplied by the evaluation framework:
            kwargs["pre_compute"]["forget"]["value_by_index"] -> {"idx": {"score": float}}
            kwargs["reference_logs"]["retain_model_logs"]["retain"]["value_by_index"] -> same
            The retain side is loaded from the pre-frozen retain-model logs JSON
            (${eval.tofu.retain_logs_path}); the reward therefore needs only a single-sided
            forward pass on the forget set.

    Returns:
        (forget_tr, retain_tr) as 1-D float arrays. ``retain_tr`` is None when the pre-frozen
        retain logs are unavailable; callers must then return {"agg_value": None} rather than
        fabricating a value.
    """
    forget_tr = np.array(
        [
            evals["score"]
            for evals in kwargs["pre_compute"]["forget"]["value_by_index"].values()
        ]
    )
    reference_logs = kwargs.get("reference_logs", None)
    if not reference_logs:
        logger.warning(
            f"retain_model_logs not provided in reference_logs, setting {metric_name} to None"
        )
        return forget_tr, None
    reference_logs = reference_logs["retain_model_logs"]
    retain_tr = np.array(
        [
            evals["score"]
            for evals in reference_logs["retain"]["value_by_index"].values()
        ]
    )
    return forget_tr, retain_tr


@unlearning_metric(name="ks_statistic")
def ks_statistic(model: Any, **kwargs: Any) -> Dict[str, Any]:
    """KS test statistic D between the forget and retain truth-ratio distributions.

    Optimization signal of the ES-MU / CSS-MU population search. The full text (主方案 §11
    事实 1-2) consistently uses the KS statistic D instead of the p-value: p is extremely
    sensitive to the sample size and dp/dD -> 0, so p is never the optimization target.
    See 主方案 §8 for the reward definition `1 - D_KS / D_FO + ...`, and §7 Prop 2′ for the
    sup-type (piecewise constant, dD = 0 a.e.) characterization of KS.

    Input kwargs (same contract as ks_test, from the metric framework):
        kwargs["pre_compute"]["forget"]["value_by_index"]: dict[str, {"score": float}]
            per-sample truth ratios of the forget set (shape: one float per forget sample).
        kwargs["reference_logs"]["retain_model_logs"]["retain"]["value_by_index"]: same
            structure, loaded from the pre-frozen retain-model logs JSON, i.e.
            ${eval.tofu.retain_logs_path} (reward needs only a single-sided forward pass).

    Returns:
        {"agg_value": float}: the KS statistic D = max|F1 - F2|. NOT the p-value.
        {"agg_value": None}: when retain_model_logs is missing (no fabricated values).
    """
    forget_tr_stats, retain_tr_stats = _collect_tr_arrays("ks_statistic", **kwargs)
    if retain_tr_stats is None:
        return {"agg_value": None}
    # method= is explicit (never the default "auto") per GOTCHA 2. D itself does not depend on
    # the method; "asymp" is chosen so that any p-value read alongside D matches the validation
    # formula p ~ 2 * exp(-2 * n_eff * D^2) with n_eff = n1 * n2 / (n1 + n2).
    fq = ks_2samp(forget_tr_stats, retain_tr_stats, method="asymp")
    return {"agg_value": float(fq.statistic)}


@unlearning_metric(name="w1_distance")
def w1_distance(model: Any, **kwargs: Any) -> Dict[str, Any]:
    """First Wasserstein (earth-mover) distance between forget and retain truth-ratio distributions.

    W1 is an integral-type functional (piecewise linear, subdifferentiable a.e.), as opposed to
    the sup-type KS functional. This contrast underlies the C-W1 first-order comparison: 主方案
    §7 Prop 2′ and §9 (control group C-W1), to be used as a first-order counterpart of the
    zero-order KS signal in later stages.

    Input kwargs (same contract as ks_statistic):
        kwargs["pre_compute"]["forget"]["value_by_index"]: dict[str, {"score": float}]
            per-sample truth ratios of the forget set (shape: one float per forget sample).
        kwargs["reference_logs"]["retain_model_logs"]["retain"]["value_by_index"]: same
            structure, from the pre-frozen retain-model logs JSON.

    Returns:
        {"agg_value": float}: the 1-Wasserstein distance between the two TR samples.
        {"agg_value": None}: when retain_model_logs is missing (no fabricated values).
    """
    forget_tr_stats, retain_tr_stats = _collect_tr_arrays("w1_distance", **kwargs)
    if retain_tr_stats is None:
        return {"agg_value": None}
    return {"agg_value": float(wasserstein_distance(forget_tr_stats, retain_tr_stats))}
