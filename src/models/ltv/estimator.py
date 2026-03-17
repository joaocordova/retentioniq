"""Customer Lifetime Value estimation for RetentionIQ.

Combines survival model predictions (expected remaining tenure) with
expected monthly revenue to produce a net-present-value LTV estimate
for every member.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from lifelines import CoxPHFitter

from src.exceptions import DataError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LTV estimation
# ---------------------------------------------------------------------------


def estimate_ltv(
    members: pd.DataFrame,
    survival_model: CoxPHFitter,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Estimate remaining customer lifetime value for each member.

    LTV = sum over t=1..T of (expected_monthly_revenue * S(t) / (1+r)^t)

    where S(t) is the predicted survival probability at month t, r is the
    monthly discount rate, and T is the maximum horizon.

    Args:
        members: DataFrame that must contain ``member_id``,
            ``monthly_price``, and all covariates required by
            *survival_model*.
        survival_model: A fitted ``CoxPHFitter`` used to predict S(t).
        config: LTV configuration dict with keys ``discount_rate`` and
            ``max_horizon_months``.

    Returns:
        DataFrame with columns ``member_id``, ``expected_remaining_tenure``,
        ``expected_monthly_revenue``, and ``ltv``.

    Raises:
        DataError: If required columns are missing.
    """
    required = {"member_id", "monthly_price"}
    missing = required - set(members.columns)
    if missing:
        raise DataError(
            f"Members DataFrame missing columns for LTV: "
            f"{sorted(missing)}"
        )

    discount_rate: float = config.get("discount_rate", 0.01)
    max_horizon: int = config.get("max_horizon_months", 36)

    # Prepare covariates for the survival model (exclude non-feature cols)
    feature_cols = [
        c for c in survival_model.params_.index
        if c in members.columns
    ]
    if not feature_cols:
        raise DataError(
            "None of the survival model covariates are present in "
            "the members DataFrame. Check column alignment."
        )

    covariates = members[feature_cols]

    # Predict full survival function: rows = time, cols = members
    survival_fn = survival_model.predict_survival_function(
        covariates,
        times=list(range(1, max_horizon + 1)),
    )
    # survival_fn shape: (max_horizon, n_members)

    monthly_revenue = members["monthly_price"].values  # (n_members,)

    # Discount factors: 1/(1+r)^t for t = 1..T
    time_points = np.arange(1, max_horizon + 1)
    discount_factors = 1.0 / (1.0 + discount_rate) ** time_points

    # Survival matrix: (T, n_members)
    S = survival_fn.values

    # NPV LTV = sum_t [ S(t) * monthly_revenue * discount(t) ]
    # Shape: (T, n_members) * (1, n_members) * (T, 1)
    discounted_survival = S * discount_factors[:, np.newaxis]
    ltv_values = (
        discounted_survival * monthly_revenue[np.newaxis, :]
    ).sum(axis=0)

    # Expected remaining tenure = sum of survival probabilities
    expected_tenure = S.sum(axis=0)

    result = pd.DataFrame(
        {
            "member_id": members["member_id"].values,
            "expected_remaining_tenure": np.round(
                expected_tenure, 2
            ),
            "expected_monthly_revenue": np.round(
                monthly_revenue, 2
            ),
            "ltv": np.round(ltv_values, 2),
        }
    )

    logger.info(
        "ltv_estimated",
        n_members=len(result),
        mean_ltv=round(float(result["ltv"].mean()), 2),
        median_ltv=round(float(result["ltv"].median()), 2),
        discount_rate=discount_rate,
        max_horizon=max_horizon,
    )
    return result


# ---------------------------------------------------------------------------
# LTV segmentation
# ---------------------------------------------------------------------------


def segment_ltv(
    ltv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add an ``ltv_segment`` column based on LTV percentiles.

    Segments:
    - ``high``:   top 25% (>= 75th percentile)
    - ``medium``: middle 50% (25th to 75th percentile)
    - ``low``:    bottom 25% (< 25th percentile)

    Args:
        ltv_df: DataFrame with at least a ``ltv`` column.

    Returns:
        Copy of *ltv_df* with an added ``ltv_segment`` column.

    Raises:
        DataError: If the ``ltv`` column is missing.
    """
    if "ltv" not in ltv_df.columns:
        raise DataError(
            "ltv_df must contain an 'ltv' column for segmentation."
        )

    df = ltv_df.copy()

    p25 = df["ltv"].quantile(0.25)
    p75 = df["ltv"].quantile(0.75)

    df["ltv_segment"] = np.where(
        df["ltv"] >= p75,
        "high",
        np.where(df["ltv"] >= p25, "medium", "low"),
    )

    segment_counts = df["ltv_segment"].value_counts().to_dict()
    logger.info(
        "ltv_segmented",
        n_members=len(df),
        p25=round(float(p25), 2),
        p75=round(float(p75), 2),
        segments=segment_counts,
    )
    return df
