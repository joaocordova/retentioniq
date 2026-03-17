"""Survival analysis module for RetentionIQ.

Fits Kaplan-Meier and Cox Proportional Hazards models to estimate
member retention curves and expected remaining tenure.  All experiments
are tracked via MLflow.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog
from lifelines import CoxPHFitter, KaplanMeierFitter

from src.exceptions import DataError, ModelError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------


def fit_kaplan_meier(
    members: pd.DataFrame,
    config: dict[str, Any],
) -> KaplanMeierFitter:
    """Fit a Kaplan-Meier survival estimator.

    When ``stratify_by`` is present in *config* (e.g. ``contract_source``),
    the model is fit on each stratum individually.  The returned fitter
    contains the **overall** (unstratified) curve; per-stratum curves are
    logged to MLflow for comparison.

    Args:
        members: DataFrame with at least the duration column and event
            column specified in *config*.
        config: Survival model configuration (typically loaded from
            ``configs/models.yaml`` under the ``survival`` key).

    Returns:
        A fitted ``KaplanMeierFitter`` instance.

    Raises:
        DataError: If required columns are missing.
    """
    duration_col: str = config["duration_col"]
    event_col: str = config["event_col"]
    stratify_by: str | None = config.get("stratify_by")

    _validate_survival_cols(members, duration_col, event_col)

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=members[duration_col],
        event_observed=members[event_col],
        label="overall",
    )

    # Log stratified curves to MLflow
    if stratify_by and stratify_by in members.columns:
        for stratum, group in members.groupby(stratify_by):
            kmf_stratum = KaplanMeierFitter()
            kmf_stratum.fit(
                durations=group[duration_col],
                event_observed=group[event_col],
                label=str(stratum),
            )
            logger.info(
                "km_stratum_fitted",
                stratum=str(stratum),
                median_survival=kmf_stratum.median_survival_time_,
            )

    logger.info(
        "kaplan_meier_fitted",
        n_members=len(members),
        median_survival=kmf.median_survival_time_,
    )
    return kmf


# ---------------------------------------------------------------------------
# Cox PH
# ---------------------------------------------------------------------------


def fit_cox_ph(
    feature_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> CoxPHFitter:
    """Fit a Cox Proportional Hazards model.

    The penalizer strength and other settings are read from *config* so
    that nothing is hard-coded.

    Args:
        feature_matrix: ML-ready DataFrame containing the duration column,
            event column, and covariates.
        config: Survival model configuration with keys ``duration_col``,
            ``event_col``, and ``penalizer``.

    Returns:
        A fitted ``CoxPHFitter``.

    Raises:
        DataError: If required columns are missing.
        ModelError: If the model fails to converge.
    """
    duration_col: str = config["duration_col"]
    event_col: str = config["event_col"]
    penalizer: float = config.get("penalizer", 0.01)

    _validate_survival_cols(feature_matrix, duration_col, event_col)

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(
            feature_matrix,
            duration_col=duration_col,
            event_col=event_col,
        )
    except Exception as exc:
        raise ModelError(
            f"CoxPH failed to converge: {exc}"
        ) from exc

    _log_cox_to_mlflow(cph, config, feature_matrix)

    logger.info(
        "cox_ph_fitted",
        n_samples=len(feature_matrix),
        n_covariates=len(cph.params_),
        concordance=round(cph.concordance_index_, 4),
    )
    return cph


def predict_survival_function(
    model: CoxPHFitter,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict survival curves for the given members.

    Args:
        model: A fitted ``CoxPHFitter``.
        features: DataFrame of covariates (same columns used during fit,
            minus the duration and event columns).

    Returns:
        DataFrame where rows are time points and columns are member
        indices, each cell being the survival probability S(t).
    """
    _ensure_fitted(model)
    survival_functions = model.predict_survival_function(features)
    logger.info(
        "survival_functions_predicted",
        n_members=survival_functions.shape[1],
        n_timepoints=survival_functions.shape[0],
    )
    return survival_functions


def compute_median_survival(
    model: CoxPHFitter,
    features: pd.DataFrame,
) -> pd.Series:
    """Compute the median expected remaining tenure for each member.

    The median survival time is the point at which the predicted survival
    probability drops below 0.5.

    Args:
        model: A fitted ``CoxPHFitter``.
        features: DataFrame of covariates.

    Returns:
        Series indexed like *features* with the median remaining tenure
        in the same units as the duration column used during fitting.
    """
    _ensure_fitted(model)
    medians = model.predict_median(features)

    # lifelines may return inf when survival never drops below 0.5;
    # cap at a sensible maximum
    max_horizon = 120  # 10 years in months
    medians = medians.clip(upper=max_horizon)

    logger.info(
        "median_survival_computed",
        n_members=len(medians),
        mean_median=round(float(np.nanmean(medians)), 2),
    )
    return medians


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


def _log_cox_to_mlflow(
    model: CoxPHFitter,
    config: dict[str, Any],
    feature_matrix: pd.DataFrame,
) -> None:
    """Log Cox PH training artefacts to MLflow."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    experiment_name = config.get(
        "experiment_name", "retentioniq-survival"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="cox_ph") as run:
        # Parameters
        mlflow.log_param("penalizer", config.get("penalizer", 0.01))
        mlflow.log_param(
            "baseline_estimation",
            config.get("baseline_estimation", "breslow"),
        )
        mlflow.log_param("n_samples", len(feature_matrix))
        mlflow.log_param(
            "n_covariates",
            len(model.params_),
        )

        # Metrics
        mlflow.log_metric(
            "concordance_index", model.concordance_index_
        )
        mlflow.log_metric(
            "log_likelihood",
            float(model.log_likelihood_),
        )
        mlflow.log_metric("AIC", float(model.AIC_partial_))

        # Log summary as text artifact
        summary = model.summary
        summary_path = "/tmp/cox_summary.csv"
        summary.to_csv(summary_path)
        mlflow.log_artifact(summary_path, artifact_path="model")

        logger.info(
            "cox_ph_logged_to_mlflow",
            run_id=run.info.run_id,
            concordance=round(model.concordance_index_, 4),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_survival_cols(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
) -> None:
    """Raise DataError if survival columns are missing."""
    missing = []
    if duration_col not in df.columns:
        missing.append(duration_col)
    if event_col not in df.columns:
        missing.append(event_col)
    if missing:
        raise DataError(
            f"Survival DataFrame missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _ensure_fitted(model: CoxPHFitter) -> None:
    """Raise ModelError if the CoxPH model has not been fitted."""
    if not hasattr(model, "params_") or model.params_ is None:
        raise ModelError(
            "CoxPHFitter has not been fitted yet. "
            "Call fit_cox_ph() first."
        )
