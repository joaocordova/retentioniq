"""Data quality validation functions for each medallion layer.

Each function returns a list of failure messages. An empty list means
all checks passed. The functions are intentionally kept simple and
framework-agnostic so they can be invoked from Dagster assets, CLI
scripts, or test suites without pulling in the full Great Expectations
runtime.
"""

import pandas as pd
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Bronze validations
# ---------------------------------------------------------------------------


def validate_bronze_members(df: pd.DataFrame) -> list[str]:
    """Validate bronze-layer member data.

    Checks:
    * Required columns are present.
    * ``member_id`` is never null.
    * ``join_date`` is parseable as a date.
    * ``location_id`` is never null.
    * No fully-duplicated rows.

    Args:
        df: Bronze members DataFrame.

    Returns:
        List of failure messages (empty when all checks pass).
    """
    failures: list[str] = []
    required = [
        "member_id",
        "location_id",
        "join_date",
        "churned",
        "contract_source",
        "monthly_price",
    ]
    _check_required_columns(df, required, failures)

    if "member_id" in df.columns:
        n_null = int(df["member_id"].isna().sum())
        if n_null > 0:
            failures.append(
                f"member_id has {n_null} null values"
            )

        n_dup = int(df["member_id"].duplicated().sum())
        if n_dup > 0:
            failures.append(
                f"member_id has {n_dup} duplicate values"
            )

    if "location_id" in df.columns:
        n_null = int(df["location_id"].isna().sum())
        if n_null > 0:
            failures.append(
                f"location_id has {n_null} null values"
            )

    if "join_date" in df.columns:
        parsed = pd.to_datetime(df["join_date"], errors="coerce")
        n_bad = int(parsed.isna().sum())
        original_nulls = int(df["join_date"].isna().sum())
        unparseable = n_bad - original_nulls
        if unparseable > 0:
            failures.append(
                f"join_date has {unparseable} unparseable values"
            )

    _log_result("bronze_members", failures)
    return failures


def validate_bronze_visits(df: pd.DataFrame) -> list[str]:
    """Validate bronze-layer visit data.

    Args:
        df: Bronze visits DataFrame.

    Returns:
        List of failure messages.
    """
    failures: list[str] = []
    required = [
        "member_id",
        "location_id",
        "visit_date",
        "visit_duration_minutes",
    ]
    _check_required_columns(df, required, failures)

    if "visit_date" in df.columns:
        parsed = pd.to_datetime(df["visit_date"], errors="coerce")
        n_bad = int(parsed.isna().sum()) - int(
            df["visit_date"].isna().sum()
        )
        if n_bad > 0:
            failures.append(
                f"visit_date has {n_bad} unparseable values"
            )

    if "visit_duration_minutes" in df.columns:
        n_neg = int(
            (df["visit_duration_minutes"] < 0).sum()
        )
        if n_neg > 0:
            failures.append(
                f"visit_duration_minutes has {n_neg} negative values"
            )

    _log_result("bronze_visits", failures)
    return failures


def validate_bronze_retention_actions(
    df: pd.DataFrame,
) -> list[str]:
    """Validate bronze-layer retention-action data.

    Args:
        df: Bronze retention-actions DataFrame.

    Returns:
        List of failure messages.
    """
    failures: list[str] = []
    required = [
        "member_id",
        "location_id",
        "action_date",
        "action_type",
        "cost",
    ]
    _check_required_columns(df, required, failures)

    if "cost" in df.columns:
        n_neg = int((df["cost"] < 0).sum())
        if n_neg > 0:
            failures.append(
                f"cost has {n_neg} negative values"
            )

    _log_result("bronze_retention_actions", failures)
    return failures


# ---------------------------------------------------------------------------
# Silver validations
# ---------------------------------------------------------------------------


def validate_silver_members(df: pd.DataFrame) -> list[str]:
    """Validate silver-layer member data (business-rule checks).

    Checks:
    * ``join_date`` is datetime and never null.
    * ``cancel_date`` (when present) >= ``join_date``.
    * ``contract_source`` is one of the valid values.
    * ``monthly_price`` is positive where not null.
    * ``churned`` is boolean and consistent with ``cancel_date``.

    Args:
        df: Silver members DataFrame.

    Returns:
        List of failure messages.
    """
    failures: list[str] = []

    # join_date must be a datetime and never null
    if "join_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["join_date"]):
            failures.append(
                "join_date should be datetime, "
                f"got {df['join_date'].dtype}"
            )
        n_null = int(df["join_date"].isna().sum())
        if n_null > 0:
            failures.append(
                f"join_date has {n_null} null values after cleaning"
            )

    # cancel_date >= join_date
    if (
        "cancel_date" in df.columns
        and "join_date" in df.columns
    ):
        valid_cancel = df["cancel_date"].notna()
        if valid_cancel.any():
            bad_order = df.loc[
                valid_cancel,
                "cancel_date",
            ] < df.loc[valid_cancel, "join_date"]
            n_bad = int(bad_order.sum())
            if n_bad > 0:
                failures.append(
                    f"{n_bad} members have cancel_date before "
                    f"join_date"
                )

    # contract_source must be valid
    valid_sources = {"regular", "aggregator"}
    if "contract_source" in df.columns:
        invalid = ~df["contract_source"].isin(valid_sources)
        n_inv = int(invalid.sum())
        if n_inv > 0:
            failures.append(
                f"contract_source has {n_inv} invalid values"
            )

    # monthly_price must be positive where not null
    if "monthly_price" in df.columns:
        non_null_prices = df["monthly_price"].dropna()
        n_neg = int((non_null_prices <= 0).sum())
        if n_neg > 0:
            failures.append(
                f"monthly_price has {n_neg} non-positive values"
            )

    # churned consistency
    if "churned" in df.columns and "cancel_date" in df.columns:
        # Churned members must have a cancel_date
        churned_no_date = (
            df["churned"].astype(bool)
            & df["cancel_date"].isna()
        )
        n_inconsistent = int(churned_no_date.sum())
        if n_inconsistent > 0:
            failures.append(
                f"{n_inconsistent} members marked churned but "
                f"have no cancel_date"
            )

    _log_result("silver_members", failures)
    return failures


def validate_silver_visits(df: pd.DataFrame) -> list[str]:
    """Validate silver-layer visit data.

    Args:
        df: Silver visits DataFrame.

    Returns:
        List of failure messages.
    """
    failures: list[str] = []

    if "visit_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(
            df["visit_date"]
        ):
            failures.append(
                "visit_date should be datetime after cleaning"
            )

    if "visit_duration_minutes" in df.columns:
        below = int(
            (df["visit_duration_minutes"] < 5).sum()
        )
        above = int(
            (df["visit_duration_minutes"] > 300).sum()
        )
        if below > 0:
            failures.append(
                f"{below} visits still have duration < 5 min"
            )
        if above > 0:
            failures.append(
                f"{above} visits still have duration > 300 min"
            )

    _log_result("silver_visits", failures)
    return failures


# ---------------------------------------------------------------------------
# Gold validations
# ---------------------------------------------------------------------------


def validate_gold_member_360(df: pd.DataFrame) -> list[str]:
    """Validate gold-layer member-360 feature table.

    Checks:
    * Expected columns are present.
    * Visit counts are non-negative.
    * ``tenure_days`` is non-negative.
    * ``days_since_last_visit`` is >= -1 (sentinel for no visits).
    * No extreme outliers in visit counts (> 500 in 30d).

    Args:
        df: Gold member-360 DataFrame.

    Returns:
        List of failure messages.
    """
    failures: list[str] = []

    expected_cols = [
        "member_id",
        "location_id",
        "tenure_days",
        "visit_count_30d",
        "days_since_last_visit",
        "retention_action_count",
    ]
    _check_required_columns(df, expected_cols, failures)

    # Non-negative visit counts
    visit_cols = [
        c for c in df.columns if c.startswith("visit_count_")
    ]
    for col in visit_cols:
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            failures.append(
                f"{col} has {n_neg} negative values"
            )

    # tenure_days non-negative
    if "tenure_days" in df.columns:
        n_neg = int((df["tenure_days"] < 0).sum())
        if n_neg > 0:
            failures.append(
                f"tenure_days has {n_neg} negative values"
            )

    # days_since_last_visit sentinel check
    if "days_since_last_visit" in df.columns:
        n_bad = int((df["days_since_last_visit"] < -1).sum())
        if n_bad > 0:
            failures.append(
                f"days_since_last_visit has {n_bad} values below -1"
            )

    # Outlier check: > 500 visits in 30d is suspicious
    if "visit_count_30d" in df.columns:
        n_outlier = int((df["visit_count_30d"] > 500).sum())
        if n_outlier > 0:
            failures.append(
                f"visit_count_30d has {n_outlier} extreme outliers "
                f"(>500)"
            )

    # retention_action_count non-negative
    if "retention_action_count" in df.columns:
        n_neg = int(
            (df["retention_action_count"] < 0).sum()
        )
        if n_neg > 0:
            failures.append(
                f"retention_action_count has {n_neg} negative values"
            )

    _log_result("gold_member_360", failures)
    return failures


def validate_gold_location_aggregates(
    df: pd.DataFrame,
) -> list[str]:
    """Validate gold-layer location-aggregate table.

    Args:
        df: Gold location-aggregates DataFrame.

    Returns:
        List of failure messages.
    """
    failures: list[str] = []
    expected_cols = [
        "location_id",
        "active_members",
        "churn_rate",
        "mrr",
    ]
    _check_required_columns(df, expected_cols, failures)

    if "churn_rate" in df.columns:
        out_of_range = (
            (df["churn_rate"] < 0) | (df["churn_rate"] > 1)
        )
        n_bad = int(out_of_range.sum())
        if n_bad > 0:
            failures.append(
                f"churn_rate has {n_bad} values outside [0, 1]"
            )

    if "mrr" in df.columns:
        n_neg = int((df["mrr"] < 0).sum())
        if n_neg > 0:
            failures.append(f"mrr has {n_neg} negative values")

    _log_result("gold_location_aggregates", failures)
    return failures


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_required_columns(
    df: pd.DataFrame,
    required: list[str],
    failures: list[str],
) -> None:
    """Append a failure message if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        failures.append(f"Missing columns: {missing}")


def _log_result(suite: str, failures: list[str]) -> None:
    """Log the outcome of a validation suite."""
    if failures:
        logger.warning(
            "validation_failed",
            suite=suite,
            n_failures=len(failures),
            first=failures[0],
        )
    else:
        logger.info("validation_passed", suite=suite)
