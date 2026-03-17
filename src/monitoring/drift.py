"""Data and model drift detection using Evidently and custom metrics.

This module monitors production data and model predictions for drift.
Drift detection is the first signal that the model may need retraining
-- it fires before performance degrades, giving the team a head start.

Key concepts:
- Data drift: input feature distributions have shifted (KS test)
- Prediction drift: model output distribution has shifted (PSI)
- Performance degradation: metrics dropped below thresholds
"""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml
from scipy import stats
from sklearn.metrics import brier_score_loss, precision_score, roc_auc_score

from src.exceptions import ModelPerformanceDegradationError

logger = structlog.get_logger()


def load_monitoring_config(
    config_path: str = "configs/monitoring.yaml",
) -> dict:
    """Load monitoring configuration from YAML.

    Args:
        config_path: Path to the monitoring config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: dict,
) -> dict[str, dict]:
    """Run KS test on configured features to detect data drift.

    The Kolmogorov-Smirnov test compares the empirical CDFs of the
    reference (training) and current (production) distributions.
    A low p-value means the distributions are statistically different.

    Args:
        reference: Reference (baseline/training) data.
        current: Current (production) data.
        config: Data drift configuration dict, expected to contain
            ``features_to_monitor`` and ``significance_level``.

    Returns:
        Dict mapping feature_name to
        ``{drifted: bool, p_value: float, statistic: float}``.
    """
    features = config.get("features_to_monitor", [])
    sig_level = config.get("significance_level", 0.05)

    results: dict[str, dict] = {}

    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            logger.warning(
                "drift_feature_missing",
                feature=feature,
                in_reference=feature in reference.columns,
                in_current=feature in current.columns,
            )
            continue

        ref_values = reference[feature].dropna().values
        cur_values = current[feature].dropna().values

        if len(ref_values) == 0 or len(cur_values) == 0:
            logger.warning(
                "drift_empty_feature",
                feature=feature,
                ref_count=len(ref_values),
                cur_count=len(cur_values),
            )
            continue

        ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
        is_drifted = p_value < sig_level

        results[feature] = {
            "drifted": is_drifted,
            "p_value": float(p_value),
            "statistic": float(ks_stat),
        }

        if is_drifted:
            logger.warning(
                "data_drift_detected",
                feature=feature,
                ks_statistic=round(ks_stat, 4),
                p_value=round(p_value, 6),
            )

    logger.info(
        "data_drift_check_complete",
        features_checked=len(results),
        features_drifted=sum(1 for r in results.values() if r["drifted"]),
    )

    return results


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI measures how much the distribution of a variable has shifted
    between two samples. Interpretation:
    - PSI < 0.1: no significant shift
    - 0.1 <= PSI < 0.2: moderate shift, monitor closely
    - PSI >= 0.2: significant shift, investigate and likely retrain

    Args:
        expected: Reference distribution values.
        actual: Current distribution values.
        bins: Number of bins for discretization.

    Returns:
        PSI value (non-negative float).
    """
    # Create bins from the expected distribution
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_percents = np.percentile(expected, breakpoints)
    # Ensure unique bin edges by adding tiny offsets
    expected_percents = np.unique(expected_percents)

    if len(expected_percents) < 2:
        # All values are the same in expected; PSI is 0 if actual is
        # also constant at the same value, otherwise undefined.
        return 0.0

    expected_counts = np.histogram(expected, bins=expected_percents)[0]
    actual_counts = np.histogram(actual, bins=expected_percents)[0]

    # Convert to proportions, avoiding division by zero
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    # Replace zeros with a small epsilon to avoid log(0)
    eps = 1e-6
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def detect_prediction_drift(
    reference_predictions: pd.Series,
    current_predictions: pd.Series,
    threshold: float = 0.2,
) -> dict:
    """Detect drift in model predictions using PSI.

    Prediction drift can signal concept drift even before ground-truth
    labels are available. If the model's output distribution shifts,
    something has changed in the relationship between features and target.

    Args:
        reference_predictions: Baseline prediction scores.
        current_predictions: Current production prediction scores.
        threshold: PSI threshold for flagging drift (default 0.2).

    Returns:
        Dict with ``psi`` (float) and ``is_drifted`` (bool).
    """
    ref_values = reference_predictions.dropna().values
    cur_values = current_predictions.dropna().values

    psi_value = compute_psi(ref_values, cur_values)
    is_drifted = psi_value > threshold

    logger.info(
        "prediction_drift_check",
        psi=round(psi_value, 4),
        threshold=threshold,
        is_drifted=is_drifted,
    )

    return {
        "psi": psi_value,
        "is_drifted": is_drifted,
    }


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str,
) -> str:
    """Generate an Evidently HTML drift report.

    Creates a comprehensive data drift report comparing reference and
    current datasets. The report includes per-feature drift tests,
    distribution plots, and summary statistics.

    Args:
        reference: Reference (baseline) DataFrame.
        current: Current (production) DataFrame.
        output_path: File path for the HTML report.

    Returns:
        Absolute path to the generated report file.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    report.save_html(output_path)

    logger.info(
        "drift_report_generated",
        path=output_path,
        ref_rows=len(reference),
        cur_rows=len(current),
    )

    return str(Path(output_path).resolve())


def check_model_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    thresholds: dict,
) -> dict:
    """Check model performance metrics against configured thresholds.

    Computes AUC-ROC, precision@k, and Brier score, then compares
    each against the threshold. Returns a detailed results dict so
    the caller can decide whether to trigger retraining.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted probabilities (0-1).
        thresholds: Dict mapping metric name to threshold value.
            Expected keys: ``auc_roc``, ``precision_at_k``,
            ``brier_score``. For ``precision_at_k``, also expects
            a ``k`` key specifying how many top predictions to
            evaluate.

    Returns:
        Dict with per-metric results:
        ``{metric_name: {value: float, threshold: float, passed: bool}}``.
    """
    results: dict[str, dict] = {}

    true_values = y_true.values
    pred_values = y_pred.values

    # AUC-ROC
    if "auc_roc" in thresholds:
        auc = roc_auc_score(true_values, pred_values)
        auc_threshold = thresholds["auc_roc"]
        results["auc_roc"] = {
            "value": float(auc),
            "threshold": auc_threshold,
            "passed": auc >= auc_threshold,
        }

    # Precision@k
    if "precision_at_k" in thresholds:
        k = thresholds.get("k", 100)
        top_k_idx = np.argsort(pred_values)[::-1][:k]
        top_k_true = true_values[top_k_idx]
        prec_at_k = float(top_k_true.sum()) / k if k > 0 else 0.0
        prec_threshold = thresholds["precision_at_k"]
        results["precision_at_k"] = {
            "value": prec_at_k,
            "threshold": prec_threshold,
            "k": k,
            "passed": prec_at_k >= prec_threshold,
        }

    # Brier score (lower is better)
    if "brier_score" in thresholds:
        brier = brier_score_loss(true_values, pred_values)
        brier_threshold = thresholds["brier_score"]
        results["brier_score"] = {
            "value": float(brier),
            "threshold": brier_threshold,
            "passed": brier <= brier_threshold,
        }

    logger.info(
        "model_performance_check",
        metrics={k: v["passed"] for k, v in results.items()},
    )

    return results
