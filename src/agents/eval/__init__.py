"""Agent evaluation framework for RetentionIQ.

Provides scenario-based evaluation of agent performance
including task success, tool selection, PII safety, and latency.
"""

from src.agents.eval.evaluator import (
    EvalScenario,
    ScenarioResult,
    check_pii_leakage,
    evaluate_agent,
    load_scenarios,
)

__all__ = [
    "EvalScenario",
    "ScenarioResult",
    "check_pii_leakage",
    "evaluate_agent",
    "load_scenarios",
]
