"""Agent layer for RetentionIQ.

Provides a LangGraph-based multi-agent system with memory,
tools, guardrails, and evaluation for retention queries.
"""

from src.agents.graph import AgentState, build_retention_graph, run_agent
from src.agents.guardrails import (
    GuardrailConfig,
    load_guardrail_config,
    mask_pii_in_dataframe,
    mask_pii_in_text,
    validate_agent_output,
)
from src.agents.memory import AgentMemoryStore
from src.agents.tools import (
    compare_cohorts,
    estimate_treatment_effect,
    get_churn_score,
    query_location_metrics,
    query_member_data,
    run_optimizer,
)

__all__ = [
    "AgentMemoryStore",
    "AgentState",
    "GuardrailConfig",
    "build_retention_graph",
    "compare_cohorts",
    "estimate_treatment_effect",
    "get_churn_score",
    "load_guardrail_config",
    "mask_pii_in_dataframe",
    "mask_pii_in_text",
    "query_location_metrics",
    "query_member_data",
    "run_agent",
    "run_optimizer",
    "validate_agent_output",
]
