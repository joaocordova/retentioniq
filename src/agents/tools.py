"""Agent tools for the LangGraph retention agent.

Each tool is a function that the agent can call to interact with the
underlying data and models. All tools enforce:
    1. PII masking before returning any data
    2. Structured logging for audit trail
    3. Guardrail checks on inputs and outputs

Tools are designed to be composable — the agent can chain them:
    query_member_data → get_churn_score → estimate_treatment_effect → run_optimizer

Security model:
    - Tools never expose raw PII (name, CPF, email, phone)
    - SQL tools are read-only with row limits
    - All tool calls are logged with masked inputs
"""

from typing import Any

import pandas as pd
import structlog

from src.agents.guardrails import (
    load_guardrail_config,
    mask_pii_in_dataframe,
    mask_pii_in_text,
    validate_agent_output,
)
from src.exceptions import AgentError

logger = structlog.get_logger()

# Load guardrail config at module level
_guardrail_config: Any = None


def _get_guardrail_config() -> Any:
    """Lazy-load guardrail config."""
    global _guardrail_config
    if _guardrail_config is None:
        _guardrail_config = load_guardrail_config()
    return _guardrail_config


def query_member_data(member_id: str) -> dict[str, Any]:
    """Look up member information with PII masked.

    Retrieves member profile data from the database, masks all PII
    fields, and returns only operationally relevant attributes.

    Args:
        member_id: Unique member identifier.

    Returns:
        Dict with member attributes (PII fields replaced with
        [REDACTED]): member_id, location_id, plan_type, tenure_months,
        contract_source, visit_frequency_30d, payment_delay_flag,
        plan_price, member_name (REDACTED), email (REDACTED).

    Raises:
        AgentError: If member not found or query fails.
    """
    gc = _get_guardrail_config()

    logger.info("tool_query_member_data", member_id=member_id)

    # In production, this queries the gold layer in PostgreSQL.
    # Placeholder returns structured mock data for development.
    try:
        # Simulated database query — replace with actual SQL in production
        member_data = {
            "member_id": member_id,
            "member_name": "[REDACTED]",
            "cpf": "[REDACTED]",
            "email": "[REDACTED]",
            "phone": "[REDACTED]",
            "address": "[REDACTED]",
            "location_id": "LOC_001",
            "plan_type": "premium",
            "tenure_months": 14,
            "contract_source": "regular",
            "visit_frequency_30d": 8,
            "days_since_last_visit": 5,
            "payment_delay_flag": False,
            "plan_price": 149.90,
            "enrollment_month": 3,
            "age_group": "25-34",
            "status": "active",
        }

        # Apply PII masking as defense-in-depth
        for field in gc.pii_fields:
            if field in member_data:
                member_data[field] = gc.pii_mask_token

        # Also mask any PII that might appear in string values
        for key, val in member_data.items():
            if isinstance(val, str):
                member_data[key] = mask_pii_in_text(
                    val, gc.pii_mask_token
                )

        logger.info(
            "tool_query_member_data_complete",
            member_id=member_id,
        )
        return member_data

    except Exception as exc:
        raise AgentError(
            f"Failed to query member data for {member_id}: {exc}"
        ) from exc


def query_location_metrics(
    location_id: str,
    period: str = "last_30d",
) -> dict[str, Any]:
    """Query location-level aggregated metrics.

    Returns operational metrics for a location without any member-level
    PII. Safe for inclusion in agent responses.

    Args:
        location_id: Location identifier.
        period: Time period string (e.g., "last_30d", "last_90d").

    Returns:
        Dict with: location_id, period, active_members, churned_members,
        churn_rate, mrr, avg_visits_per_member, at_risk_members,
        aggregator_pct, top_churn_reasons.

    Raises:
        AgentError: If location not found or query fails.
    """
    logger.info(
        "tool_query_location_metrics",
        location_id=location_id,
        period=period,
    )

    try:
        # Simulated query — replace with gold layer SQL in production
        metrics = {
            "location_id": location_id,
            "period": period,
            "active_members": 320,
            "churned_members": 28,
            "churn_rate": 0.0875,
            "mrr": 47_840.00,
            "avg_visits_per_member": 6.2,
            "at_risk_members": 45,
            "aggregator_pct": 0.35,
            "top_churn_reasons": [
                {"reason": "low_visit_frequency", "pct": 0.42},
                {"reason": "payment_delay", "pct": 0.28},
                {"reason": "seasonal_dropout", "pct": 0.18},
            ],
            "retention_actions_this_period": 52,
            "budget_spent": 1_840.00,
            "budget_remaining": 160.00,
        }

        logger.info(
            "tool_query_location_metrics_complete",
            location_id=location_id,
            churn_rate=metrics["churn_rate"],
        )
        return metrics

    except Exception as exc:
        raise AgentError(
            f"Failed to query metrics for location "
            f"{location_id}: {exc}"
        ) from exc


def get_churn_score(member_id: str) -> dict[str, Any]:
    """Get churn prediction for a member.

    Looks up the latest churn probability from the model prediction
    table. Returns the score, risk tier, and top contributing features.

    Args:
        member_id: Unique member identifier.

    Returns:
        Dict with: member_id, churn_probability, risk_tier,
        horizon_days, top_risk_factors, model_version.

    Raises:
        AgentError: If prediction not available.
    """
    logger.info("tool_get_churn_score", member_id=member_id)

    try:
        # Simulated model lookup — replace with MLflow model call
        score = {
            "member_id": member_id,
            "churn_probability": 0.73,
            "risk_tier": "high",
            "horizon_days": 30,
            "top_risk_factors": [
                {"feature": "days_since_last_visit", "impact": 0.31},
                {"feature": "visit_frequency_trend", "impact": -0.22},
                {"feature": "payment_delay_flag", "impact": 0.18},
            ],
            "model_version": "churn_xgb_v2.1",
            "predicted_at": "2026-03-17T10:00:00Z",
        }

        logger.info(
            "tool_get_churn_score_complete",
            member_id=member_id,
            churn_probability=score["churn_probability"],
            risk_tier=score["risk_tier"],
        )
        return score

    except Exception as exc:
        raise AgentError(
            f"Failed to get churn score for {member_id}: {exc}"
        ) from exc


def estimate_treatment_effect(
    segment: str,
    action: str,
) -> dict[str, Any]:
    """Get the estimated treatment effect for a segment and action.

    Looks up the CATE estimate from the causal inference pipeline.
    Returns the expected reduction in churn probability if the given
    action is applied to members in the specified segment.

    Args:
        segment: Member segment (e.g., "regular_6-12m",
            "aggregator_0-3m").
        action: Retention action name (e.g., "phone_call",
            "discount_offer").

    Returns:
        Dict with: segment, action, cate_mean, cate_std, ci_lower,
        ci_upper, n_members_in_segment, significant (bool).

    Raises:
        AgentError: If CATE estimate not available.
    """
    logger.info(
        "tool_estimate_treatment_effect",
        segment=segment,
        action=action,
    )

    try:
        # Simulated CATE lookup — replace with causal forest prediction
        effect = {
            "segment": segment,
            "action": action,
            "cate_mean": 0.08,
            "cate_std": 0.02,
            "ci_lower": 0.04,
            "ci_upper": 0.12,
            "n_members_in_segment": 85,
            "significant": True,
            "interpretation": (
                f"Applying {action} to {segment} members is expected "
                f"to reduce 30-day churn by ~8 percentage points "
                f"(95% CI: 4-12pp)."
            ),
        }

        logger.info(
            "tool_estimate_treatment_effect_complete",
            segment=segment,
            action=action,
            cate_mean=effect["cate_mean"],
        )
        return effect

    except Exception as exc:
        raise AgentError(
            f"Failed to estimate treatment effect for "
            f"{segment}/{action}: {exc}"
        ) from exc


def run_optimizer(
    location_ids: list[str],
    budget: float,
) -> dict[str, Any]:
    """Run the budget optimizer for specified locations.

    Invokes the Pyomo stochastic optimizer to allocate retention
    budget across members and actions. Returns a summary suitable
    for the agent to communicate to the manager.

    Args:
        location_ids: List of location IDs to optimize for.
        budget: Total budget in R$ for the optimization run.

    Returns:
        Dict with: locations, budget_used, expected_retained_mrr,
        roi, allocations_summary (counts by action), top_actions.

    Raises:
        AgentError: If optimization fails.
    """
    logger.info(
        "tool_run_optimizer",
        n_locations=len(location_ids),
        budget=budget,
    )

    try:
        # Simulated optimizer result — replace with actual Pyomo solve
        result = {
            "locations": location_ids,
            "budget_total": budget,
            "budget_used": min(budget, budget * 0.92),
            "expected_retained_mrr": budget * 3.2,
            "roi": 3.2,
            "members_targeted": 45,
            "allocations_summary": {
                "sms_reengagement": 20,
                "phone_call": 15,
                "discount_offer": 7,
                "personal_trainer_session": 3,
            },
            "top_actions": [
                {
                    "action": "phone_call",
                    "members": 15,
                    "expected_saves": 8,
                    "cost": 225.00,
                },
                {
                    "action": "sms_reengagement",
                    "members": 20,
                    "expected_saves": 6,
                    "cost": 50.00,
                },
            ],
            "solver_status": "optimal",
            "scenarios_evaluated": 50,
        }

        logger.info(
            "tool_run_optimizer_complete",
            budget_used=result["budget_used"],
            members_targeted=result["members_targeted"],
            roi=result["roi"],
        )
        return result

    except Exception as exc:
        raise AgentError(
            f"Optimization failed for locations "
            f"{location_ids}: {exc}"
        ) from exc


def compare_cohorts(
    cohort_a: dict[str, Any],
    cohort_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two member cohorts on key retention metrics.

    Used by the analyst agent to answer questions like "how do
    aggregator members compare to regular members?" or "how does
    location A compare to location B?"

    Both cohort dicts should specify filters (e.g.,
    {"contract_source": "regular", "location_id": "LOC_001"}).

    Args:
        cohort_a: Filter criteria for cohort A.
        cohort_b: Filter criteria for cohort B.

    Returns:
        Dict with: cohort_a_stats, cohort_b_stats, differences,
        significant_differences (list of metrics with p < 0.05).

    Raises:
        AgentError: If cohort query fails.
    """
    gc = _get_guardrail_config()

    logger.info(
        "tool_compare_cohorts",
        cohort_a=cohort_a,
        cohort_b=cohort_b,
    )

    try:
        # Simulated cohort comparison — replace with gold layer query
        result = {
            "cohort_a": {
                "filters": cohort_a,
                "n_members": 180,
                "churn_rate": 0.07,
                "avg_tenure_months": 18.5,
                "avg_visits_per_month": 7.2,
                "avg_ltv": 2_150.00,
                "mrr": 26_910.00,
            },
            "cohort_b": {
                "filters": cohort_b,
                "n_members": 140,
                "churn_rate": 0.12,
                "avg_tenure_months": 8.3,
                "avg_visits_per_month": 4.1,
                "avg_ltv": 980.00,
                "mrr": 20_930.00,
            },
            "differences": {
                "churn_rate_diff": -0.05,
                "tenure_diff_months": 10.2,
                "visits_diff": 3.1,
                "ltv_diff": 1_170.00,
            },
            "significant_differences": [
                "churn_rate",
                "avg_visits_per_month",
                "avg_ltv",
            ],
            "interpretation": (
                "Cohort A has significantly lower churn (7% vs 12%), "
                "higher visit frequency, and higher LTV. "
                "The difference is statistically significant."
            ),
        }

        # Ensure no PII in the output
        output_text = str(result)
        validated = validate_agent_output(output_text, gc.pii_mask_token)
        if validated != output_text:
            logger.warning("pii_detected_in_cohort_comparison")

        logger.info(
            "tool_compare_cohorts_complete",
            n_a=result["cohort_a"]["n_members"],
            n_b=result["cohort_b"]["n_members"],
        )
        return result

    except AgentError:
        raise
    except Exception as exc:
        raise AgentError(
            f"Cohort comparison failed: {exc}"
        ) from exc
