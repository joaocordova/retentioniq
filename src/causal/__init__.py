"""Causal inference layer for RetentionIQ.

Provides DAG definition, ATE/CATE estimation, causal forests,
and refutation testing for retention treatment effects.
"""

from src.causal.dag import CausalDAGConfig, build_causal_model, load_dag_config
from src.causal.effects import estimate_ate, estimate_cate_by_segment, refute_estimate
from src.causal.forests import (
    fit_causal_forest,
    get_feature_importance,
    predict_individual_effects,
    summarize_effects_by_group,
)

__all__ = [
    "CausalDAGConfig",
    "build_causal_model",
    "estimate_ate",
    "estimate_cate_by_segment",
    "fit_causal_forest",
    "get_feature_importance",
    "load_dag_config",
    "predict_individual_effects",
    "refute_estimate",
    "summarize_effects_by_group",
]
