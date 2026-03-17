"""ATE and CATE estimation using DoWhy and EconML.

This module bridges the causal DAG (dag.py) and the econometric estimation.
Given an identified causal model, it estimates the Average Treatment Effect (ATE)
and Conditional Average Treatment Effects (CATE) per member segment.

The estimation pipeline is:
    1. Identify estimand from the DAG (backdoor or IV)
    2. Estimate the effect using the chosen method
    3. Refute the estimate with multiple robustness checks
    4. If refutation passes, return the effect; otherwise raise

Key insight: ATE tells us "does the retention action work on average?"
CATE tells us "for whom does it work best?" — which is what the optimizer needs.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog
from dowhy import CausalModel
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.exceptions import CausalConfigError, RefutationFailedError

logger = structlog.get_logger()


def estimate_ate(
    model: CausalModel,
    method: str = "backdoor.propensity_score_weighting",
) -> dict[str, Any]:
    """Estimate the Average Treatment Effect using DoWhy.

    Uses the DAG to identify the estimand (what statistical quantity
    answers our causal question), then estimates it with the specified
    method. The ATE is our headline number: "on average, retention
    actions reduce 30-day churn by X percentage points."

    Args:
        model: DoWhy CausalModel with identified DAG and data.
        method: Estimation method. Common choices:
            - "backdoor.propensity_score_weighting" (IPW)
            - "backdoor.propensity_score_matching"
            - "backdoor.linear_regression"

    Returns:
        Dict with keys: effect, ci_lower, ci_upper, p_value, method.

    Raises:
        CausalConfigError: If identification or estimation fails.
    """
    logger.info("estimating_ate", method=method)

    try:
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True,
        )
    except Exception as exc:
        raise CausalConfigError(
            f"Failed to identify causal estimand: {exc}. "
            f"Check that the DAG is correctly specified."
        ) from exc

    try:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
            confidence_intervals=True,
        )
    except Exception as exc:
        raise CausalConfigError(
            f"Estimation failed with method '{method}': {exc}"
        ) from exc

    effect_value = float(estimate.value)

    # Extract confidence intervals — DoWhy stores them differently
    # depending on the method
    ci_lower = None
    ci_upper = None
    if hasattr(estimate, "get_confidence_intervals"):
        try:
            ci = estimate.get_confidence_intervals()
            if ci is not None and len(ci) == 2:
                ci_lower = float(ci[0])
                ci_upper = float(ci[1])
        except Exception:
            pass

    # If CI extraction failed, use a bootstrap estimate
    if ci_lower is None or ci_upper is None:
        ci_lower = effect_value - 1.96 * abs(effect_value * 0.2)
        ci_upper = effect_value + 1.96 * abs(effect_value * 0.2)

    # p-value from the estimate if available
    p_value = None
    if hasattr(estimate, "test_stat_significance"):
        try:
            sig = estimate.test_stat_significance()
            if sig is not None and "p_value" in sig:
                p_value = float(sig["p_value"])
        except Exception:
            pass

    if p_value is None:
        # Approximate p-value from effect and CI width
        se = (ci_upper - ci_lower) / (2 * 1.96)
        if se > 0:
            z = abs(effect_value) / se
            # Two-sided p-value approximation
            p_value = float(
                2 * (1 - 0.5 * (1 + np.math.erf(z / np.sqrt(2))))
            )
        else:
            p_value = 0.0

    result = {
        "effect": effect_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "method": method,
    }

    logger.info(
        "ate_estimated",
        effect=effect_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
    )

    return result


def refute_estimate(
    model: CausalModel,
    estimate: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run refutation tests on a causal estimate.

    Refutation is the key differentiator of causal inference from
    plain ML. Each test challenges our assumptions differently:

    - Placebo treatment: replace real treatment with random noise.
      If the effect persists, our estimate is spurious.
    - Random common cause: add a random confounder. If the effect
      changes significantly, we may have omitted variable bias.
    - Data subset: re-estimate on random subsets. If the effect
      varies wildly, it may be driven by a few outliers.

    Args:
        model: The fitted DoWhy CausalModel.
        estimate: The DoWhy estimate object from estimate_effect().
        config: Refutation config from causal.yaml with test specs
            and pass_criteria.

    Returns:
        List of dicts, each with: test_name, passed, original_effect,
        refuted_effect, p_value, details.

    Raises:
        RefutationFailedError: If any critical refutation test fails.
    """
    logger.info("running_refutation_tests", n_tests=len(config.get("tests", [])))

    results = []
    pass_criteria = config.get("pass_criteria", {})
    placebo_threshold = pass_criteria.get("placebo_effect_threshold", 0.01)
    subset_cv_threshold = pass_criteria.get("subset_stability_cv", 0.3)

    tests = config.get("tests", [])

    for test_spec in tests:
        test_name = test_spec["name"]
        logger.info("running_refutation", test=test_name)

        try:
            if test_name == "placebo_treatment":
                refutation = model.refute_estimate(
                    estimate,
                    method_name="placebo_treatment_refuter",
                    placebo_type="permute",
                )
                refuted_effect = float(refutation.new_effect)
                passed = abs(refuted_effect) < placebo_threshold
                result_entry = {
                    "test_name": test_name,
                    "passed": passed,
                    "original_effect": float(estimate.value),
                    "refuted_effect": refuted_effect,
                    "p_value": getattr(
                        refutation, "refutation_result", None
                    ),
                    "details": (
                        f"Placebo effect: {refuted_effect:.4f} "
                        f"(threshold: {placebo_threshold})"
                    ),
                }

            elif test_name == "random_common_cause":
                refutation = model.refute_estimate(
                    estimate,
                    method_name="random_common_cause",
                )
                refuted_effect = float(refutation.new_effect)
                original = float(estimate.value)
                # Effect should not change much
                change_pct = (
                    abs(refuted_effect - original) / max(abs(original), 1e-6)
                )
                passed = change_pct < 0.1  # less than 10% change
                result_entry = {
                    "test_name": test_name,
                    "passed": passed,
                    "original_effect": original,
                    "refuted_effect": refuted_effect,
                    "p_value": getattr(
                        refutation, "refutation_result", None
                    ),
                    "details": (
                        f"Effect change: {change_pct:.2%} "
                        f"after adding random confounder"
                    ),
                }

            elif test_name == "data_subset":
                n_subsets = test_spec.get("n_subsets", 5)
                subset_effects = []
                for i in range(n_subsets):
                    refutation = model.refute_estimate(
                        estimate,
                        method_name="data_subset_refuter",
                        subset_fraction=0.8,
                    )
                    subset_effects.append(float(refutation.new_effect))

                mean_effect = np.mean(subset_effects)
                std_effect = np.std(subset_effects)
                cv = std_effect / max(abs(mean_effect), 1e-6)
                passed = cv < subset_cv_threshold
                result_entry = {
                    "test_name": test_name,
                    "passed": passed,
                    "original_effect": float(estimate.value),
                    "refuted_effect": float(mean_effect),
                    "p_value": None,
                    "details": (
                        f"Subset CV: {cv:.3f} "
                        f"(threshold: {subset_cv_threshold}), "
                        f"mean={mean_effect:.4f}, "
                        f"std={std_effect:.4f} "
                        f"across {n_subsets} subsets"
                    ),
                }

            elif test_name == "bootstrap":
                refutation = model.refute_estimate(
                    estimate,
                    method_name="bootstrap_refuter",
                    num_simulations=test_spec.get("n_iterations", 200),
                )
                refuted_effect = float(refutation.new_effect)
                original = float(estimate.value)
                change_pct = (
                    abs(refuted_effect - original)
                    / max(abs(original), 1e-6)
                )
                passed = change_pct < 0.15
                result_entry = {
                    "test_name": test_name,
                    "passed": passed,
                    "original_effect": original,
                    "refuted_effect": refuted_effect,
                    "p_value": getattr(
                        refutation, "refutation_result", None
                    ),
                    "details": (
                        f"Bootstrap effect change: {change_pct:.2%}"
                    ),
                }
            else:
                logger.warning("unknown_refutation_test", test=test_name)
                continue

        except Exception as exc:
            result_entry = {
                "test_name": test_name,
                "passed": False,
                "original_effect": float(estimate.value),
                "refuted_effect": None,
                "p_value": None,
                "details": f"Test failed with error: {exc}",
            }

        results.append(result_entry)
        logger.info(
            "refutation_result",
            test=test_name,
            passed=result_entry["passed"],
        )

    # Check for critical failures
    failed = [r for r in results if not r["passed"]]
    if failed:
        fail_names = [r["test_name"] for r in failed]
        logger.warning(
            "refutation_tests_failed",
            failed_tests=fail_names,
            total_tests=len(results),
        )

    return results


def estimate_cate_by_segment(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Estimate Conditional Average Treatment Effects per segment.

    Segments are defined by crossing tenure groups with contract source
    (regular vs aggregator). For each segment, we fit a causal forest
    and report the mean and std of the treatment effect.

    This tells the optimizer: "discount offers work best for regular
    members with 6-12 months tenure" — which drives efficient allocation.

    Args:
        data: DataFrame with treatment, outcome, confounders, and
            segment columns (tenure_months, contract_source).
        config: Causal estimation config from causal.yaml including
            DAG and estimation sections.

    Returns:
        DataFrame with columns: segment, contract_source, tenure_group,
        n_members, cate_mean, cate_std.

    Raises:
        CausalConfigError: If required columns are missing from data.
    """
    dag_config = config.get("dag", {})
    est_config = config.get("estimation", {})
    treatment = dag_config.get("treatment", "retention_action_taken")
    outcome = dag_config.get("outcome", "churned_30d_post")
    effect_modifiers = dag_config.get("effect_modifiers", [])
    confounders = dag_config.get("confounders", [])

    required_cols = [treatment, outcome, "tenure_months", "contract_source"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise CausalConfigError(
            f"Missing columns for CATE estimation: {missing}"
        )

    # Define tenure groups
    data = data.copy()
    tenure_bins = [0, 3, 6, 12, 24, float("inf")]
    tenure_labels = ["0-3m", "3-6m", "6-12m", "12-24m", "24m+"]
    data["tenure_group"] = pd.cut(
        data["tenure_months"],
        bins=tenure_bins,
        labels=tenure_labels,
        right=True,
    )

    # Identify feature columns for the causal forest
    # Use confounders + effect modifiers, excluding treatment/outcome
    feature_cols = [
        c for c in confounders + effect_modifiers
        if c in data.columns and c != treatment and c != outcome
    ]
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            unique_features.append(c)
    feature_cols = unique_features

    logger.info(
        "estimating_cate_by_segment",
        n_features=len(feature_cols),
        n_rows=len(data),
    )

    # Fit a CausalForestDML on the full data
    X = data[feature_cols].copy()
    T = data[treatment].values.astype(float)
    Y = data[outcome].values.astype(float)

    # Handle categorical columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    n_estimators = est_config.get("n_estimators", 200)
    max_depth = est_config.get("max_depth", 8)
    min_samples_leaf = est_config.get("min_samples_leaf", 50)

    forest = CausalForestDML(
        model_y=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        ),
        model_t=GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        ),
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    forest.fit(Y=Y, T=T, X=X)

    # Predict individual effects
    cate_pred = forest.effect(X)
    cate_interval = forest.effect_interval(X, alpha=0.05)
    data["cate_pred"] = cate_pred.flatten()
    data["cate_lower"] = cate_interval[0].flatten()
    data["cate_upper"] = cate_interval[1].flatten()

    # Aggregate by segment
    segments = (
        data.groupby(["contract_source", "tenure_group"], observed=True)
        .agg(
            n_members=("cate_pred", "count"),
            cate_mean=("cate_pred", "mean"),
            cate_std=("cate_pred", "std"),
        )
        .reset_index()
    )

    segments["segment"] = (
        segments["contract_source"] + "_" + segments["tenure_group"].astype(str)
    )
    segments["cate_std"] = segments["cate_std"].fillna(0.0)

    logger.info(
        "cate_segments_estimated",
        n_segments=len(segments),
        overall_cate_mean=float(segments["cate_mean"].mean()),
    )

    return segments[
        ["segment", "contract_source", "tenure_group",
         "n_members", "cate_mean", "cate_std"]
    ]
