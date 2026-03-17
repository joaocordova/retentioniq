"""Feast feature store integration for RetentionIQ.

Provides FeatureView definitions and helper functions for pushing /
retrieving features from the online store.  The implementation is
designed to work with Feast's Python SDK while keeping a clean
interface for the rest of the system.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import pandas as pd
import structlog
from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Float64, Int64

from src.exceptions import DataError, ModelError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Entity definitions
# ---------------------------------------------------------------------------

member_entity = Entity(
    name="member",
    join_keys=["member_id"],
    description="A gym member identified by member_id.",
)

# ---------------------------------------------------------------------------
# Data sources — default to local parquet; override via env for production
# ---------------------------------------------------------------------------

_DATA_DIR = os.getenv(
    "FEAST_DATA_DIR", "data/gold"
)

visit_behavior_source = FileSource(
    path=f"{_DATA_DIR}/visit_features.parquet",
    timestamp_field="event_timestamp",
)

tenure_source = FileSource(
    path=f"{_DATA_DIR}/tenure_features.parquet",
    timestamp_field="event_timestamp",
)

financial_source = FileSource(
    path=f"{_DATA_DIR}/financial_features.parquet",
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

visit_behavior_fv = FeatureView(
    name="visit_behavior",
    entities=[member_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="visit_count_7d", dtype=Int64),
        Field(name="visit_count_14d", dtype=Int64),
        Field(name="visit_count_30d", dtype=Int64),
        Field(name="visit_count_60d", dtype=Int64),
        Field(name="visit_count_90d", dtype=Int64),
        Field(name="avg_visit_duration_30d", dtype=Float32),
        Field(name="days_since_last_visit", dtype=Int64),
        Field(name="visit_frequency_trend", dtype=Float64),
    ],
    source=visit_behavior_source,
    online=True,
    description="Rolling visit behaviour features at multiple windows.",
)

tenure_fv = FeatureView(
    name="tenure",
    entities=[member_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="tenure_days", dtype=Int64),
        Field(name="tenure_months", dtype=Float32),
        Field(name="is_first_90_days", dtype=Int64),
    ],
    source=tenure_source,
    online=True,
    description="Membership tenure features.",
)

financial_fv = FeatureView(
    name="financial",
    entities=[member_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="monthly_price", dtype=Float64),
        Field(name="total_revenue", dtype=Float64),
        Field(name="payment_delay_flag", dtype=Int64),
    ],
    source=financial_source,
    online=True,
    description="Revenue and payment features.",
)

# ---------------------------------------------------------------------------
# Feature service — bundles all views for single-call retrieval
# ---------------------------------------------------------------------------

retention_feature_service = FeatureService(
    name="retention_features",
    features=[visit_behavior_fv, tenure_fv, financial_fv],
    description=(
        "All features used by churn, survival, and LTV models."
    ),
)

# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

_FEATURE_VIEW_REGISTRY: dict[str, FeatureView] = {
    "visit_behavior": visit_behavior_fv,
    "tenure": tenure_fv,
    "financial": financial_fv,
}


def _get_store() -> FeatureStore:
    """Return a Feast FeatureStore using the repo path from env."""
    repo_path = os.getenv("FEAST_REPO_PATH", "feature_repo")
    return FeatureStore(repo_path=repo_path)


def get_online_features(
    entity_keys: list[dict[str, Any]],
    feature_refs: list[str],
) -> pd.DataFrame:
    """Retrieve features from the Feast online store.

    Args:
        entity_keys: List of entity key dicts, e.g.
            ``[{"member_id": 42}, {"member_id": 99}]``.
        feature_refs: Feature references in ``view:feature`` format, e.g.
            ``["visit_behavior:visit_count_7d"]``.

    Returns:
        DataFrame with entity keys and requested feature columns.

    Raises:
        ModelError: If the Feast store is unreachable or the query fails.
    """
    if not entity_keys:
        raise DataError("entity_keys must be a non-empty list.")
    if not feature_refs:
        raise DataError("feature_refs must be a non-empty list.")

    try:
        store = _get_store()
        response = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_keys,
        )
        df = response.to_df()
        logger.info(
            "online_features_retrieved",
            n_entities=len(entity_keys),
            n_features=len(feature_refs),
        )
        return df
    except Exception as exc:
        raise ModelError(
            f"Failed to retrieve online features from Feast: {exc}"
        ) from exc


def push_features_to_store(
    df: pd.DataFrame,
    feature_view_name: str,
) -> None:
    """Push a DataFrame of features into the Feast online store.

    The DataFrame must contain the entity key column (``member_id``) and
    an ``event_timestamp`` column, plus any feature columns defined in
    the corresponding FeatureView.

    Args:
        df: DataFrame to push.
        feature_view_name: Name of the target FeatureView
            (``visit_behavior``, ``tenure``, or ``financial``).

    Raises:
        DataError: If the feature_view_name is unknown or required columns
            are missing.
        ModelError: If the push operation fails.
    """
    if feature_view_name not in _FEATURE_VIEW_REGISTRY:
        raise DataError(
            f"Unknown feature view '{feature_view_name}'. "
            f"Valid names: {list(_FEATURE_VIEW_REGISTRY.keys())}"
        )

    required = {"member_id", "event_timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise DataError(
            f"DataFrame is missing required columns for push: "
            f"{sorted(missing)}"
        )

    try:
        store = _get_store()
        store.push(
            push_source_name=f"{feature_view_name}_push",
            df=df,
        )
        logger.info(
            "features_pushed",
            feature_view=feature_view_name,
            n_rows=len(df),
        )
    except Exception as exc:
        raise ModelError(
            f"Failed to push features to Feast view "
            f"'{feature_view_name}': {exc}"
        ) from exc
