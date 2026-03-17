"""Gold layer: business-ready aggregated tables.

Builds the analytical tables consumed by the ML, causal-inference,
optimisation, and agent layers:

* **member_360** -- one row per member with visit behaviour,
  tenure, retention-action history, and contract metadata.
* **location_aggregates** -- one row per location with KPIs.
* **cohort_table** -- monthly enrolment cohorts with retention curves.
"""

from typing import Any

import pandas as pd
import structlog
import yaml
from dagster import AssetExecutionContext, AssetIn, asset

logger = structlog.get_logger()

# Reference date for "current" calculations.
_REFERENCE_DATE = pd.Timestamp("2026-03-17")


# ---------------------------------------------------------------------------
# Public builder functions
# ---------------------------------------------------------------------------


def build_member_360(
    members: pd.DataFrame,
    visits: pd.DataFrame,
    actions: pd.DataFrame,
    visit_windows: list[int] | None = None,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a member-level 360-degree view.

    For each member the output contains:
    * ``visit_count_<N>d`` -- number of visits in the last *N* days.
    * ``days_since_last_visit`` -- calendar days since the most
      recent visit (relative to *reference_date*).
    * ``tenure_days`` -- days between join and cancel (or
      *reference_date* if still active).
    * ``retention_action_count`` -- number of retention actions
      received.
    * ``last_action_type`` -- most recent action type (or ``None``).
    * Demographic / contract columns carried through.

    Args:
        members: Silver-layer members DataFrame (dates already
            parsed as datetime64).
        visits: Silver-layer visits DataFrame.
        actions: Silver-layer retention-actions DataFrame.
        visit_windows: List of day-windows to compute, e.g.
            ``[7, 14, 30, 60, 90]``. Defaults to the gold config.
        reference_date: "Today" for relative calculations.

    Returns:
        DataFrame with one row per member.
    """
    ref = reference_date or _REFERENCE_DATE
    if visit_windows is None:
        visit_windows = [7, 14, 30, 60, 90]

    logger.info(
        "build_member_360_start",
        members=len(members),
        visits=len(visits),
        actions=len(actions),
    )

    # Ensure datetime types ------------------------------------------------
    members = members.copy()
    members["join_date"] = pd.to_datetime(
        members["join_date"], errors="coerce"
    )
    members["cancel_date"] = pd.to_datetime(
        members["cancel_date"], errors="coerce"
    )

    visits = visits.copy()
    visits["visit_date"] = pd.to_datetime(
        visits["visit_date"], errors="coerce"
    )

    actions = actions.copy()
    actions["action_date"] = pd.to_datetime(
        actions["action_date"], errors="coerce"
    )

    # -- Tenure ------------------------------------------------------------
    members["tenure_days"] = (
        members["cancel_date"].fillna(ref) - members["join_date"]
    ).dt.days

    # -- Visit aggregations ------------------------------------------------
    visit_agg = _aggregate_visits(visits, visit_windows, ref)

    # -- Retention-action aggregations ------------------------------------
    action_agg = _aggregate_actions(actions)

    # -- Merge -------------------------------------------------------------
    out = members.merge(visit_agg, on="member_id", how="left")
    out = out.merge(action_agg, on="member_id", how="left")

    # Fill NaN visit/action counts with 0
    fill_cols = [
        f"visit_count_{w}d" for w in visit_windows
    ] + ["retention_action_count"]
    for col in fill_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    out["days_since_last_visit"] = out[
        "days_since_last_visit"
    ].fillna(-1).astype(int)

    logger.info("build_member_360_done", rows=len(out))
    return out


def build_location_aggregates(
    member_360: pd.DataFrame,
) -> pd.DataFrame:
    """Build location-level KPI aggregates.

    Columns produced:
    * ``active_members`` -- count of members not yet churned.
    * ``churned_members`` -- count of churned members.
    * ``churn_rate`` -- churned / total.
    * ``avg_visits_30d`` -- mean 30-day visit count across members.
    * ``mrr`` -- monthly recurring revenue (sum of ``monthly_price``
      for active members).
    * ``aggregator_pct`` -- fraction of members sourced through
      aggregators.

    Args:
        member_360: Gold-layer member-360 DataFrame.

    Returns:
        DataFrame with one row per location.
    """
    logger.info(
        "build_location_aggregates_start",
        rows=len(member_360),
    )

    df = member_360.copy()

    # Ensure boolean churned
    df["churned"] = df["churned"].astype(bool)

    agg = df.groupby("location_id").agg(
        total_members=("member_id", "count"),
        active_members=(
            "churned",
            lambda s: int((~s).sum()),
        ),
        churned_members=(
            "churned",
            lambda s: int(s.sum()),
        ),
        avg_visits_30d=(
            "visit_count_30d",
            "mean",
        ),
        avg_tenure_days=("tenure_days", "mean"),
        mrr=(
            "monthly_price",
            lambda s: s.loc[~df.loc[s.index, "churned"]].sum(),
        ),
        aggregator_count=(
            "contract_source",
            lambda s: int((s == "aggregator").sum()),
        ),
    ).reset_index()

    agg["churn_rate"] = (
        agg["churned_members"] / agg["total_members"]
    ).round(4)
    agg["aggregator_pct"] = (
        agg["aggregator_count"] / agg["total_members"]
    ).round(4)

    # Round numeric columns
    agg["avg_visits_30d"] = agg["avg_visits_30d"].round(2)
    agg["avg_tenure_days"] = agg["avg_tenure_days"].round(1)
    agg["mrr"] = agg["mrr"].round(2)

    agg = agg.drop(columns=["aggregator_count"])

    logger.info(
        "build_location_aggregates_done",
        locations=len(agg),
    )
    return agg


def build_cohort_table(
    members: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    min_cohort_size: int = 50,
) -> pd.DataFrame:
    """Build monthly enrolment cohorts with retention curves.

    Each row represents a (cohort_month, period) pair with the
    fraction of the original cohort still active at each monthly
    period after enrolment.

    Args:
        members: Silver- or gold-layer members DataFrame.
        reference_date: "Today" for right-censoring.
        min_cohort_size: Minimum number of enrolments for a cohort
            to be included.

    Returns:
        DataFrame with columns ``cohort_month``, ``period``,
        ``cohort_size``, ``retained``, ``retention_rate``.
    """
    ref = reference_date or _REFERENCE_DATE
    logger.info("build_cohort_table_start", members=len(members))

    df = members.copy()
    df["join_date"] = pd.to_datetime(
        df["join_date"], errors="coerce"
    )
    df["cancel_date"] = pd.to_datetime(
        df["cancel_date"], errors="coerce"
    )

    # Assign each member to a cohort month
    df["cohort_month"] = df["join_date"].dt.to_period("M")

    # Maximum observable months from each cohort to the reference date
    cohorts = df.groupby("cohort_month")["member_id"].count()
    cohorts = cohorts[cohorts >= min_cohort_size]

    rows: list[dict[str, Any]] = []
    for cohort_period, cohort_size in cohorts.items():
        cohort_members = df[df["cohort_month"] == cohort_period]
        cohort_start = cohort_period.to_timestamp()
        max_months = max(
            1,
            (ref.year - cohort_start.year) * 12
            + (ref.month - cohort_start.month),
        )

        for period in range(max_months + 1):
            cutoff = cohort_start + pd.DateOffset(months=period)
            # A member is "retained" if they have not cancelled
            # before the cutoff
            retained = int(
                (
                    cohort_members["cancel_date"].isna()
                    | (cohort_members["cancel_date"] >= cutoff)
                ).sum()
            )
            rows.append(
                {
                    "cohort_month": str(cohort_period),
                    "period": period,
                    "cohort_size": int(cohort_size),
                    "retained": retained,
                    "retention_rate": round(
                        retained / int(cohort_size), 4
                    ),
                }
            )

    result = pd.DataFrame(rows)
    logger.info(
        "build_cohort_table_done",
        cohorts=len(cohorts),
        rows=len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_visits(
    visits: pd.DataFrame,
    windows: list[int],
    ref: pd.Timestamp,
) -> pd.DataFrame:
    """Compute per-member visit counts for each window and recency."""
    if visits.empty:
        cols = {
            f"visit_count_{w}d": pd.Series(dtype="int64")
            for w in windows
        }
        cols["days_since_last_visit"] = pd.Series(dtype="int64")
        cols["member_id"] = pd.Series(dtype="object")
        return pd.DataFrame(cols)

    agg_frames: list[pd.DataFrame] = []

    for w in windows:
        cutoff = ref - pd.Timedelta(days=w)
        window_visits = visits[visits["visit_date"] >= cutoff]
        counts = (
            window_visits.groupby("member_id")
            .size()
            .reset_index(name=f"visit_count_{w}d")
        )
        agg_frames.append(counts)

    # Days since last visit
    last_visit = (
        visits.groupby("member_id")["visit_date"]
        .max()
        .reset_index()
    )
    last_visit["days_since_last_visit"] = (
        ref - last_visit["visit_date"]
    ).dt.days
    last_visit = last_visit.drop(columns=["visit_date"])

    # Merge all aggregations
    result = agg_frames[0]
    for frame in agg_frames[1:]:
        result = result.merge(frame, on="member_id", how="outer")
    result = result.merge(last_visit, on="member_id", how="outer")

    return result


def _aggregate_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-member retention-action counts and last type."""
    if actions.empty:
        return pd.DataFrame(
            {
                "member_id": pd.Series(dtype="object"),
                "retention_action_count": pd.Series(
                    dtype="int64"
                ),
                "last_action_type": pd.Series(dtype="object"),
            }
        )

    counts = (
        actions.groupby("member_id")
        .size()
        .reset_index(name="retention_action_count")
    )

    last_action = (
        actions.sort_values("action_date")
        .groupby("member_id")["action_type"]
        .last()
        .reset_index()
        .rename(columns={"action_type": "last_action_type"})
    )

    return counts.merge(last_action, on="member_id", how="left")


def _load_gold_config(
    config_path: str = "configs/data.yaml",
) -> dict[str, Any]:
    """Load gold-layer config from the YAML file."""
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    return config["gold"]


# ---------------------------------------------------------------------------
# Dagster assets
# ---------------------------------------------------------------------------


@asset(
    group_name="gold",
    ins={
        "silver_members": AssetIn(),
        "silver_visits": AssetIn(),
        "silver_retention_actions": AssetIn(),
    },
    description="Member-level 360-degree view with visit behaviour.",
)
def gold_member_360(
    context: AssetExecutionContext,
    silver_members: pd.DataFrame,
    silver_visits: pd.DataFrame,
    silver_retention_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the gold-layer member-360 table."""
    gold_cfg = _load_gold_config()
    windows = gold_cfg["member_360"]["visit_windows_days"]
    df = build_member_360(
        silver_members,
        silver_visits,
        silver_retention_actions,
        visit_windows=windows,
    )
    context.log.info(f"gold_member_360: {len(df)} rows")
    return df


@asset(
    group_name="gold",
    ins={"gold_member_360": AssetIn()},
    description="Location-level KPI aggregates.",
)
def gold_location_aggregates(
    context: AssetExecutionContext,
    gold_member_360: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the gold-layer location-aggregates table."""
    df = build_location_aggregates(gold_member_360)
    context.log.info(
        f"gold_location_aggregates: {len(df)} locations"
    )
    return df


@asset(
    group_name="gold",
    ins={"silver_members": AssetIn()},
    description="Monthly enrolment cohorts with retention curves.",
)
def gold_cohort_table(
    context: AssetExecutionContext,
    silver_members: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the gold-layer cohort table."""
    gold_cfg = _load_gold_config()
    min_size = gold_cfg["cohort"]["min_cohort_size"]
    df = build_cohort_table(silver_members, min_cohort_size=min_size)
    context.log.info(
        f"gold_cohort_table: {len(df)} rows"
    )
    return df
