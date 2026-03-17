"""Feature engineering for RetentionIQ.

Transforms raw member/visit data into ML-ready feature matrices.
All windowed aggregations respect temporal boundaries to prevent leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import yaml

from src.exceptions import DataError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Visit features
# ---------------------------------------------------------------------------


def compute_visit_features(
    visits: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute visit-based behavioural features per member.

    Calculates rolling visit counts at multiple windows (7d, 14d, 30d, 60d,
    90d), average visit duration over 30 days, days since last visit, and a
    frequency trend indicator (ratio of recent 14-day visits to prior 14-day
    visits).

    Args:
        visits: DataFrame with columns ``member_id``, ``visit_date``, and
            optionally ``visit_duration_minutes``.
        reference_date: The point-in-time for which features are computed.
            Only visits **before** this date are considered.

    Returns:
        DataFrame indexed by ``member_id`` with one row per member.
    """
    required_cols = {"member_id", "visit_date"}
    _validate_columns(visits, required_cols, "visits")

    visits = visits.copy()
    visits["visit_date"] = pd.to_datetime(visits["visit_date"])

    # Only look at visits strictly before reference date
    mask = visits["visit_date"] < reference_date
    visits = visits.loc[mask]

    members = visits["member_id"].unique()

    windows = {
        "visit_count_7d": 7,
        "visit_count_14d": 14,
        "visit_count_30d": 30,
        "visit_count_60d": 60,
        "visit_count_90d": 90,
    }

    result = pd.DataFrame({"member_id": members})

    for col_name, days in windows.items():
        cutoff = reference_date - pd.Timedelta(days=days)
        window_visits = visits.loc[visits["visit_date"] >= cutoff]
        counts = (
            window_visits.groupby("member_id")
            .size()
            .reset_index(name=col_name)
        )
        result = result.merge(counts, on="member_id", how="left")

    # Average visit duration in last 30 days
    cutoff_30 = reference_date - pd.Timedelta(days=30)
    recent_30 = visits.loc[visits["visit_date"] >= cutoff_30]
    if "visit_duration_minutes" in visits.columns:
        avg_dur = (
            recent_30.groupby("member_id")["visit_duration_minutes"]
            .mean()
            .reset_index(name="avg_visit_duration_30d")
        )
    else:
        avg_dur = pd.DataFrame(
            {"member_id": members, "avg_visit_duration_30d": np.nan}
        )
    result = result.merge(avg_dur, on="member_id", how="left")

    # Days since last visit
    last_visit = (
        visits.groupby("member_id")["visit_date"]
        .max()
        .reset_index(name="last_visit_date")
    )
    result = result.merge(last_visit, on="member_id", how="left")
    result["days_since_last_visit"] = (
        (reference_date - result["last_visit_date"]).dt.days
    )
    result.drop(columns=["last_visit_date"], inplace=True)

    # Visit frequency trend: ratio of last-14d to prior-14d visits
    cutoff_14 = reference_date - pd.Timedelta(days=14)
    cutoff_28 = reference_date - pd.Timedelta(days=28)

    recent_14 = visits.loc[visits["visit_date"] >= cutoff_14]
    prior_14 = visits.loc[
        (visits["visit_date"] >= cutoff_28)
        & (visits["visit_date"] < cutoff_14)
    ]

    cnt_recent = (
        recent_14.groupby("member_id")
        .size()
        .reset_index(name="_recent_14")
    )
    cnt_prior = (
        prior_14.groupby("member_id")
        .size()
        .reset_index(name="_prior_14")
    )
    result = result.merge(cnt_recent, on="member_id", how="left")
    result = result.merge(cnt_prior, on="member_id", how="left")

    result["_recent_14"] = result["_recent_14"].fillna(0)
    result["_prior_14"] = result["_prior_14"].fillna(0)
    result["visit_frequency_trend"] = result["_recent_14"] / np.maximum(
        result["_prior_14"], 1
    )
    result.drop(columns=["_recent_14", "_prior_14"], inplace=True)

    # Fill remaining NaNs with 0 for count columns
    for col in windows:
        result[col] = result[col].fillna(0).astype(int)

    logger.info(
        "visit_features_computed",
        n_members=len(result),
        reference_date=str(reference_date),
    )
    return result


# ---------------------------------------------------------------------------
# Tenure features
# ---------------------------------------------------------------------------


def compute_tenure_features(
    members: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute tenure-related features for each member.

    Args:
        members: DataFrame with columns ``member_id`` and ``join_date``.
        reference_date: Point-in-time reference for calculating tenure.

    Returns:
        DataFrame with ``member_id``, ``tenure_days``, ``tenure_months``,
        and ``is_first_90_days``.
    """
    required_cols = {"member_id", "join_date"}
    _validate_columns(members, required_cols, "members")

    df = members[["member_id", "join_date"]].copy()
    df["join_date"] = pd.to_datetime(df["join_date"])

    df["tenure_days"] = (reference_date - df["join_date"]).dt.days
    df["tenure_months"] = (df["tenure_days"] / 30.0).round(2)
    df["is_first_90_days"] = (df["tenure_days"] <= 90).astype(int)

    df.drop(columns=["join_date"], inplace=True)

    logger.info(
        "tenure_features_computed",
        n_members=len(df),
        reference_date=str(reference_date),
    )
    return df


# ---------------------------------------------------------------------------
# Contract features
# ---------------------------------------------------------------------------


def compute_contract_features(
    members: pd.DataFrame,
) -> pd.DataFrame:
    """One-hot encode contract and plan type features.

    Encodes ``contract_source`` and ``plan_type`` columns into binary
    indicator columns.

    Args:
        members: DataFrame with columns ``member_id``, ``contract_source``,
            and ``plan_type``.

    Returns:
        DataFrame with ``member_id`` and one-hot encoded columns.
    """
    required_cols = {"member_id", "contract_source", "plan_type"}
    _validate_columns(members, required_cols, "members")

    df = members[["member_id", "contract_source", "plan_type"]].copy()

    contract_dummies = pd.get_dummies(
        df["contract_source"], prefix="contract_source"
    )
    plan_dummies = pd.get_dummies(df["plan_type"], prefix="plan_type")

    result = pd.concat(
        [df[["member_id"]], contract_dummies, plan_dummies], axis=1
    )

    logger.info(
        "contract_features_computed",
        n_members=len(result),
        n_features=result.shape[1] - 1,
    )
    return result


# ---------------------------------------------------------------------------
# Build full feature matrix
# ---------------------------------------------------------------------------


def build_feature_matrix(
    member_360: pd.DataFrame,
    config_path: str = "configs/features.yaml",
) -> pd.DataFrame:
    """Build an ML-ready feature matrix from a 360-degree member view.

    Reads the feature configuration to determine which columns to exclude
    (PII, identifiers, leakage targets) and returns the remaining numeric
    features with NaNs filled.

    Args:
        member_360: Wide DataFrame containing pre-joined member, visit,
            tenure, and contract features.  Expected to already contain
            the individual feature columns.
        config_path: Path to the features YAML configuration file.

    Returns:
        DataFrame suitable for model training / scoring.
    """
    config = _load_config(config_path)
    exclude_cols: list[str] = config.get("exclude_from_models", [])

    cols_to_drop = [c for c in exclude_cols if c in member_360.columns]
    df = member_360.drop(columns=cols_to_drop)

    # Drop any remaining string/object columns that slipped through
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        logger.warning(
            "dropping_non_numeric_columns",
            columns=object_cols,
        )
        df = df.drop(columns=object_cols)

    # Ensure all boolean columns are int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    logger.info(
        "feature_matrix_built",
        shape=df.shape,
        dropped_cols=cols_to_drop,
    )
    return df


# ---------------------------------------------------------------------------
# Temporal train / validation / test split
# ---------------------------------------------------------------------------


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    train_months: int,
    val_months: int,
    test_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame temporally — never randomly.

    Data is ordered by *date_col* and then carved into three contiguous
    windows: train, validation, and test.

    Args:
        df: Input DataFrame containing a date column.
        date_col: Name of the datetime column used for ordering.
        train_months: Number of months for the training window.
        val_months: Number of months for the validation window.
        test_months: Number of months for the test window.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        DataError: If the date column is missing or the DataFrame is empty.
    """
    if date_col not in df.columns:
        raise DataError(
            f"Date column '{date_col}' not found in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    if df.empty:
        raise DataError("Cannot split an empty DataFrame.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    min_date = df[date_col].min()
    train_end = min_date + pd.DateOffset(months=train_months)
    val_end = train_end + pd.DateOffset(months=val_months)
    test_end = val_end + pd.DateOffset(months=test_months)

    train_df = df.loc[df[date_col] < train_end]
    val_df = df.loc[
        (df[date_col] >= train_end) & (df[date_col] < val_end)
    ]
    test_df = df.loc[
        (df[date_col] >= val_end) & (df[date_col] < test_end)
    ]

    logger.info(
        "temporal_split",
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        train_end=str(train_end.date()),
        val_end=str(val_end.date()),
        test_end=str(test_end.date()),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """Raise DataError if *required* columns are missing from *df*."""
    missing = required - set(df.columns)
    if missing:
        raise DataError(
            f"DataFrame '{name}' is missing required columns: "
            f"{sorted(missing)}"
        )


def _load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise DataError(f"Config file not found: {path}")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)
