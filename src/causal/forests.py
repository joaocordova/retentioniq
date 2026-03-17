"""Causal forests for heterogeneous treatment effect estimation.

This module uses EconML's CausalForestDML to estimate individual-level
treatment effects (CATE). The key question: "for THIS specific member,
how much would a retention action reduce their churn probability?"

Unlike the ATE (a single number), causal forests give us a full distribution
of effects across the member population. This enables the optimizer to
allocate budget to members who will benefit most.

The DML (Double Machine Learning) framework is crucial here:
    1. First stage: predict outcome Y from confounders (nuisance model)
    2. First stage: predict treatment T from confounders (propensity model)
    3. Second stage: estimate effect on residuals (removes confounding bias)

This "orthogonalization" makes the CATE estimate robust to misspecification
of the nuisance models, as long as they converge at reasonable rates.

References:
    - Athey, S. & Imbens, G. (2019). Generalized Random Forests.
    - Chernozhukov et al. (2018). Double/Debiased Machine Learning.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.exceptions import CausalConfigError

logger = structlog.get_logger()


def _prepare_features(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix by encoding categoricals.

    Args:
        X: Raw feature DataFrame.

    Returns:
        Tuple of (encoded DataFrame, list of column names).
    """
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes
    # Fill NaN with column medians
    X = X.fillna(X.median(numeric_only=True))
    return X, list(X.columns)


def fit_causal_forest(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> CausalForestDML:
    """Fit an EconML CausalForestDML for heterogeneous treatment effects.

    The forest estimates individual-level CATE by combining DML
    (for bias removal) with random forests (for effect heterogeneity).
    Honest splitting ensures valid confidence intervals — each tree
    uses separate subsamples for splitting and estimation.

    Args:
        data: DataFrame with treatment, outcome, and feature columns.
            Must contain columns specified in config's DAG section.
        config: Full causal config dict with keys:
            - dag.treatment: treatment column name
            - dag.outcome: outcome column name
            - dag.confounders: list of confounder column names
            - dag.effect_modifiers: list of effect modifier names
            - estimation.n_estimators: number of trees
            - estimation.max_depth: max tree depth
            - estimation.min_samples_leaf: min samples per leaf
            - estimation.honest: whether to use honest splitting

    Returns:
        Fitted CausalForestDML model.

    Raises:
        CausalConfigError: If required columns are missing or config invalid.
    """
    dag_config = config.get("dag", {})
    est_config = config.get("estimation", {})

    treatment = dag_config.get("treatment")
    outcome = dag_config.get("outcome")
    confounders = dag_config.get("confounders", [])
    effect_modifiers = dag_config.get("effect_modifiers", [])

    if not treatment or not outcome:
        raise CausalConfigError(
            "Config must specify dag.treatment and dag.outcome"
        )

    # Build feature set: confounders + effect modifiers (deduplicated)
    feature_cols = []
    seen: set[str] = set()
    for col in confounders + effect_modifiers:
        if col not in seen and col in data.columns:
            feature_cols.append(col)
            seen.add(col)

    # Validate columns exist
    required = [treatment, outcome] + feature_cols
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise CausalConfigError(
            f"Missing columns in data: {missing}. "
            f"Available: {list(data.columns)}"
        )

    X = data[feature_cols].copy()
    T = data[treatment].values.astype(float)
    Y = data[outcome].values.astype(float)

    X, col_names = _prepare_features(X)

    n_estimators = est_config.get("n_estimators", 200)
    max_depth = est_config.get("max_depth", 8)
    min_samples_leaf = est_config.get("min_samples_leaf", 50)
    honest = est_config.get("honest", True)
    cv = est_config.get("cv", 5)

    logger.info(
        "fitting_causal_forest",
        n_samples=len(data),
        n_features=len(feature_cols),
        n_estimators=n_estimators,
        max_depth=max_depth,
        honest=honest,
    )

    # Build nuisance models
    model_y = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
    )
    model_t = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
    )

    forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        honest=honest,
        cv=cv,
        random_state=42,
    )

    forest.fit(Y=Y, T=T, X=X)

    # Store feature names for later use
    forest.feature_names_ = col_names

    logger.info(
        "causal_forest_fitted",
        n_estimators=n_estimators,
        n_features=len(feature_cols),
    )

    return forest


def predict_individual_effects(
    forest: CausalForestDML,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Predict individual-level treatment effects with uncertainty.

    For each member, returns the estimated CATE and a confidence
    interval. Members with higher CATE are better targets for
    retention actions — the optimizer uses these to allocate budget.

    Args:
        forest: Fitted CausalForestDML model.
        X: Feature matrix for prediction. Must contain the same
            columns used during fitting (confounders + modifiers).
            Should include a 'member_id' column if available.

    Returns:
        DataFrame with columns: member_id, cate_mean, cate_std,
        cate_lower, cate_upper.
    """
    # Extract member_id if present
    member_ids = None
    if "member_id" in X.columns:
        member_ids = X["member_id"].values
        X_pred = X.drop(columns=["member_id"]).copy()
    else:
        X_pred = X.copy()

    X_pred, _ = _prepare_features(X_pred)

    logger.info("predicting_individual_effects", n_members=len(X_pred))

    # Point estimates
    cate = forest.effect(X_pred)

    # Confidence intervals (95%)
    cate_interval = forest.effect_interval(X_pred, alpha=0.05)
    cate_lower = cate_interval[0].flatten()
    cate_upper = cate_interval[1].flatten()
    cate_mean = cate.flatten()

    # Approximate std from CI width (assuming normal)
    cate_std = (cate_upper - cate_lower) / (2 * 1.96)

    result = pd.DataFrame({
        "cate_mean": cate_mean,
        "cate_std": cate_std,
        "cate_lower": cate_lower,
        "cate_upper": cate_upper,
    })

    if member_ids is not None:
        result.insert(0, "member_id", member_ids)
    else:
        result.insert(0, "member_id", range(len(result)))

    logger.info(
        "individual_effects_predicted",
        n_members=len(result),
        mean_cate=float(result["cate_mean"].mean()),
        std_cate=float(result["cate_mean"].std()),
    )

    return result


def get_feature_importance(
    forest: CausalForestDML,
) -> pd.DataFrame:
    """Extract feature importance from the causal forest.

    These importances tell us which features drive HETEROGENEITY
    in treatment effects — not which features predict the outcome.
    A high importance means "the treatment effect varies a lot
    depending on this feature."

    For example, if tenure_months has high importance, it means
    retention actions work very differently for new vs. long-term
    members — the optimizer should differentiate.

    Args:
        forest: Fitted CausalForestDML model.

    Returns:
        DataFrame with columns: feature, importance, rank.
        Sorted by importance descending.
    """
    logger.info("extracting_feature_importance")

    # CausalForestDML exposes feature_importances_ after fitting
    importances = forest.feature_importances_

    feature_names = getattr(forest, "feature_names_", None)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    result = result.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    result["rank"] = range(1, len(result) + 1)

    logger.info(
        "feature_importance_extracted",
        top_feature=result.iloc[0]["feature"],
        top_importance=float(result.iloc[0]["importance"]),
    )

    return result


def summarize_effects_by_group(
    effects: pd.DataFrame,
    members: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate individual treatment effects by member segments.

    Rolls up the individual CATE predictions into segment-level
    summaries. Used for reporting ("aggregator members with <6 months
    tenure have 3x the treatment effect of regular long-tenure members")
    and for the optimizer's segment-level constraints.

    Args:
        effects: DataFrame from predict_individual_effects() with
            member_id, cate_mean, cate_std.
        members: Member data with member_id and grouping columns
            (e.g., contract_source, location_id, plan_type).
        group_cols: Columns to group by (e.g., ["contract_source",
            "tenure_group"]).

    Returns:
        DataFrame with group_cols + n_members, cate_mean, cate_std,
        cate_median, cate_q25, cate_q75.
    """
    # Validate group_cols exist in members
    missing = [c for c in group_cols if c not in members.columns]
    if missing:
        raise CausalConfigError(
            f"Group columns not found in members data: {missing}"
        )

    # Merge effects with member attributes
    merged = effects.merge(members, on="member_id", how="left")

    logger.info(
        "summarizing_effects_by_group",
        group_cols=group_cols,
        n_members=len(merged),
    )

    summary = (
        merged.groupby(group_cols, observed=True)
        .agg(
            n_members=("cate_mean", "count"),
            cate_mean=("cate_mean", "mean"),
            cate_std=("cate_mean", "std"),
            cate_median=("cate_mean", "median"),
            cate_q25=("cate_mean", lambda x: x.quantile(0.25)),
            cate_q75=("cate_mean", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    summary["cate_std"] = summary["cate_std"].fillna(0.0)

    logger.info(
        "effects_summarized",
        n_groups=len(summary),
        overall_mean=float(summary["cate_mean"].mean()),
    )

    return summary
