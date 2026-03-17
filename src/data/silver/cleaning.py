"""Silver layer: cleaning, deduplication, and standardisation.

Transforms bronze-layer DataFrames into clean, analysis-ready tables.
Handles date parsing, null treatment, re-enrollment logic, and
filtering of invalid records based on rules from ``configs/data.yaml``.
"""

from typing import Any

import pandas as pd
import structlog
import yaml
from dagster import AssetExecutionContext, AssetIn, asset

from src.exceptions import DataError

logger = structlog.get_logger()

# Valid retention-action types (canonical list).
_VALID_ACTION_TYPES: set[str] = {
    "sms_reengagement",
    "phone_call",
    "discount_offer",
    "pt_session",
}


# ---------------------------------------------------------------------------
# Public cleaning functions
# ---------------------------------------------------------------------------


def clean_members(
    df: pd.DataFrame,
    rules: dict[str, Any],
) -> pd.DataFrame:
    """Clean and standardise the raw members table.

    Processing steps:
        1. Parse ``join_date`` and ``cancel_date`` to datetime.
        2. Drop exact-duplicate rows.
        3. Enforce null thresholds from *rules*.
        4. Validate ``contract_source`` against the allowed list.
        5. Apply re-enrollment logic: if a member cancels and
           re-enrolls within ``reenrollment_window_days``, the
           cancellation is reversed (same tenure).

    Args:
        df: Raw members DataFrame (bronze layer).
        rules: Silver-layer rules dict from ``configs/data.yaml``.

    Returns:
        Cleaned members DataFrame.

    Raises:
        DataError: When null thresholds are exceeded.
    """
    logger.info("clean_members_start", rows=len(df))
    out = df.copy()

    # 1. Parse dates ----------------------------------------------------------
    out["join_date"] = pd.to_datetime(out["join_date"], errors="coerce")
    out["cancel_date"] = pd.to_datetime(
        out["cancel_date"], errors="coerce"
    )

    # 2. Deduplicate ----------------------------------------------------------
    before = len(out)
    out = out.drop_duplicates(subset=["member_id"], keep="last")
    logger.info(
        "clean_members_dedup",
        dropped=before - len(out),
    )

    # 3. Null thresholds ------------------------------------------------------
    null_thresholds: dict[str, float] = rules.get(
        "null_thresholds", {}
    )
    _enforce_null_thresholds(out, null_thresholds)

    # 4. Validate contract_source ---------------------------------------------
    valid_sources: list[str] = rules.get("valid_contract_sources", [])
    if valid_sources:
        invalid_mask = ~out["contract_source"].isin(valid_sources)
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            logger.warning(
                "clean_members_invalid_contract_source",
                n_invalid=n_invalid,
            )
            out.loc[invalid_mask, "contract_source"] = "regular"

    # 5. Validate plan_type ---------------------------------------------------
    valid_plans: list[str] = rules.get("valid_plan_types", [])
    if valid_plans:
        invalid_plan_mask = (
            out["plan_type"].notna() & ~out["plan_type"].isin(valid_plans)
        )
        n_bad_plans = int(invalid_plan_mask.sum())
        if n_bad_plans > 0:
            logger.warning(
                "clean_members_invalid_plan_type",
                n_invalid=n_bad_plans,
            )
            out.loc[invalid_plan_mask, "plan_type"] = "basic"

    # 6. Re-enrollment logic --------------------------------------------------
    window_days = rules.get("reenrollment_window_days", 30)
    out = _apply_reenrollment_logic(out, window_days)

    logger.info("clean_members_done", rows=len(out))
    return out


def clean_visits(
    df: pd.DataFrame,
    rules: dict[str, Any],
) -> pd.DataFrame:
    """Clean the raw visits table.

    Processing steps:
        1. Parse ``visit_date`` to datetime.
        2. Drop duplicates.
        3. Filter visits with invalid durations (below
           ``min_visit_duration_minutes`` or above
           ``max_visit_duration_minutes``).

    Args:
        df: Raw visits DataFrame (bronze layer).
        rules: Silver-layer rules dict from ``configs/data.yaml``.

    Returns:
        Cleaned visits DataFrame.
    """
    logger.info("clean_visits_start", rows=len(df))
    out = df.copy()

    # 1. Parse dates ----------------------------------------------------------
    out["visit_date"] = pd.to_datetime(
        out["visit_date"], errors="coerce"
    )

    # 2. Deduplicate ----------------------------------------------------------
    before = len(out)
    out = out.drop_duplicates()
    logger.info("clean_visits_dedup", dropped=before - len(out))

    # 3. Duration filter ------------------------------------------------------
    min_dur = rules.get("min_visit_duration_minutes", 5)
    max_dur = rules.get("max_visit_duration_minutes", 300)

    valid_mask = out["visit_duration_minutes"].between(
        min_dur, max_dur
    )
    n_filtered = int((~valid_mask).sum())
    if n_filtered > 0:
        logger.warning(
            "clean_visits_invalid_duration",
            n_filtered=n_filtered,
            min=min_dur,
            max=max_dur,
        )
    out = out.loc[valid_mask].reset_index(drop=True)

    # 4. Drop rows where visit_date could not be parsed -----------------------
    out = out.dropna(subset=["visit_date"]).reset_index(drop=True)

    logger.info("clean_visits_done", rows=len(out))
    return out


def clean_retention_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw retention actions table.

    Processing steps:
        1. Parse ``action_date`` to datetime.
        2. Drop duplicates.
        3. Validate ``action_type`` against the canonical list.

    Args:
        df: Raw retention-actions DataFrame (bronze layer).

    Returns:
        Cleaned retention-actions DataFrame.
    """
    logger.info("clean_retention_actions_start", rows=len(df))
    out = df.copy()

    # 1. Parse dates ----------------------------------------------------------
    out["action_date"] = pd.to_datetime(
        out["action_date"], errors="coerce"
    )

    # 2. Deduplicate ----------------------------------------------------------
    before = len(out)
    out = out.drop_duplicates()
    logger.info(
        "clean_retention_actions_dedup",
        dropped=before - len(out),
    )

    # 3. Validate action types ------------------------------------------------
    invalid_mask = ~out["action_type"].isin(_VALID_ACTION_TYPES)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        logger.warning(
            "clean_retention_actions_invalid_type",
            n_invalid=n_invalid,
            invalid_types=out.loc[
                invalid_mask, "action_type"
            ].unique().tolist(),
        )
        out = out.loc[~invalid_mask].reset_index(drop=True)

    # 4. Drop unparseable dates -----------------------------------------------
    out = out.dropna(subset=["action_date"]).reset_index(drop=True)

    logger.info("clean_retention_actions_done", rows=len(out))
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _enforce_null_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    """Raise ``DataError`` if any column exceeds its null threshold."""
    for col, max_null_frac in thresholds.items():
        if col not in df.columns:
            continue
        null_frac = float(df[col].isna().mean())
        if null_frac > max_null_frac:
            raise DataError(
                f"Column '{col}' has {null_frac:.2%} nulls, "
                f"exceeding threshold of {max_null_frac:.2%}"
            )


def _apply_reenrollment_logic(
    df: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """Collapse cancel-then-reenroll sequences within *window_days*.

    When the same ``member_id`` appears multiple times and the gap
    between one row's ``cancel_date`` and the next row's ``join_date``
    is <= *window_days*, we treat it as continuous tenure: keep the
    earliest ``join_date`` and the latest ``cancel_date``.
    """
    if "member_id" not in df.columns:
        return df

    # Sort so earliest join comes first
    df = df.sort_values(
        ["member_id", "join_date"]
    ).reset_index(drop=True)

    dup_mask = df.duplicated(subset=["member_id"], keep=False)
    if not dup_mask.any():
        return df

    unique_rows = df.loc[~dup_mask].copy()
    dup_rows = df.loc[dup_mask].copy()

    merged: list[pd.DataFrame] = []
    for _mid, group in dup_rows.groupby("member_id"):
        group = group.sort_values("join_date").reset_index(drop=True)
        keep_idx = 0
        result = group.iloc[[keep_idx]].copy()

        for i in range(1, len(group)):
            prev_cancel = group.loc[
                group.index[keep_idx], "cancel_date"
            ]
            curr_join = group.loc[group.index[i], "join_date"]

            if (
                pd.notna(prev_cancel)
                and pd.notna(curr_join)
                and (curr_join - prev_cancel).days <= window_days
            ):
                # Merge: keep earliest join, latest cancel
                new_cancel = group.loc[
                    group.index[i], "cancel_date"
                ]
                result.iloc[0, result.columns.get_loc("cancel_date")] = (
                    new_cancel
                )
                # Update churned flag
                if pd.isna(new_cancel):
                    result.iloc[
                        0,
                        result.columns.get_loc("churned"),
                    ] = False
                else:
                    result.iloc[
                        0,
                        result.columns.get_loc("churned"),
                    ] = True
            else:
                keep_idx = i
                result = pd.concat(
                    [result, group.iloc[[i]]],
                    ignore_index=True,
                )

        merged.append(result)

    if merged:
        merged_df = pd.concat(merged, ignore_index=True)
        out = pd.concat(
            [unique_rows, merged_df], ignore_index=True
        )
    else:
        out = unique_rows

    logger.info(
        "reenrollment_logic_applied",
        window_days=window_days,
        rows_before=len(df),
        rows_after=len(out),
    )
    return out


def _load_silver_rules(
    config_path: str = "configs/data.yaml",
) -> dict[str, Any]:
    """Load silver-layer rules from the config file."""
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    return config["silver"]["rules"]


# ---------------------------------------------------------------------------
# Dagster assets
# ---------------------------------------------------------------------------


@asset(
    group_name="silver",
    ins={"bronze_members": AssetIn()},
    description="Cleaned and standardised member data.",
)
def silver_members(
    context: AssetExecutionContext,
    bronze_members: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the silver-layer members table."""
    rules = _load_silver_rules()
    df = clean_members(bronze_members, rules)
    context.log.info(f"silver_members: {len(df)} rows")
    return df


@asset(
    group_name="silver",
    ins={"bronze_visits": AssetIn()},
    description="Cleaned visit data with invalid durations removed.",
)
def silver_visits(
    context: AssetExecutionContext,
    bronze_visits: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the silver-layer visits table."""
    rules = _load_silver_rules()
    df = clean_visits(bronze_visits, rules)
    context.log.info(f"silver_visits: {len(df)} rows")
    return df


@asset(
    group_name="silver",
    ins={"bronze_retention_actions": AssetIn()},
    description=(
        "Cleaned retention actions with validated action types."
    ),
)
def silver_retention_actions(
    context: AssetExecutionContext,
    bronze_retention_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the silver-layer retention-actions table."""
    df = clean_retention_actions(bronze_retention_actions)
    context.log.info(
        f"silver_retention_actions: {len(df)} rows"
    )
    return df
