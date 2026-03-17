"""Feature engineering and store integration for RetentionIQ."""

from src.features.engineering import (
    build_feature_matrix,
    compute_contract_features,
    compute_tenure_features,
    compute_visit_features,
    temporal_train_test_split,
)
from src.features.store import (
    get_online_features,
    push_features_to_store,
    retention_feature_service,
    visit_behavior_fv,
    tenure_fv,
    financial_fv,
    member_entity,
)

__all__ = [
    # engineering
    "compute_visit_features",
    "compute_tenure_features",
    "compute_contract_features",
    "build_feature_matrix",
    "temporal_train_test_split",
    # store
    "get_online_features",
    "push_features_to_store",
    "retention_feature_service",
    "visit_behavior_fv",
    "tenure_fv",
    "financial_fv",
    "member_entity",
]
