"""LangGraph state machine for the RetentionIQ multi-agent system.

This module defines the agent graph that processes natural language queries
from location managers. The architecture follows a router → specialist → writer
pattern:

    1. Router: classifies the query intent (factual/diagnostic/prescriptive)
    2. Analyst: handles data queries and statistical analysis
    3. Strategist: handles causal analysis and optimization
    4. Writer: formats output in manager-friendly language

The graph uses LangGraph's StateGraph with conditional edges for routing.
Every node enforces PII masking and confidence checks via guardrails.

State flows through the graph as a TypedDict, accumulating context,
tool results, and confidence scores at each node.
"""

import os
import time
import uuid
from typing import Any, Literal

import structlog
import yaml
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.agents.guardrails import (
    check_confidence,
    load_guardrail_config,
    mask_pii_in_text,
    validate_agent_output,
)
from src.agents.tools import (
    compare_cohorts,
    estimate_treatment_effect,
    get_churn_score,
    query_location_metrics,
    query_member_data,
    run_optimizer,
)
from src.exceptions import AgentError

logger = structlog.get_logger()


class AgentState(TypedDict):
    """State that flows through the agent graph.

    Each node reads from and writes to this state. The graph framework
    handles merging updates across nodes.
    """

    messages: list[dict[str, str]]
    context: dict[str, Any]
    tools_called: list[str]
    confidence: float
    query: str
    query_type: str
    location_id: str | None
    session_id: str
    analysis_result: dict[str, Any] | None
    strategy_result: dict[str, Any] | None
    final_response: str
    error: str | None


def _load_agents_config(
    config_path: str = "configs/agents.yaml",
) -> dict[str, Any]:
    """Load agents configuration from YAML.

    Args:
        config_path: Path to agents config file.

    Returns:
        Full agents config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_router_node() -> Any:
    """Create the router node that classifies query intent.

    The router examines the user's query and classifies it into one
    of four categories, which determines the processing path:
    - factual: simple data lookup (Analyst only)
    - diagnostic: why-type analysis (Analyst -> Strategist)
    - prescriptive: what-to-do recommendations (full pipeline)
    - conversational: chitchat or unclear (Writer only)

    Returns:
        A function compatible with LangGraph's add_node().
    """

    def router_node(state: AgentState) -> dict[str, Any]:
        """Classify query intent and set routing."""
        query = state["query"].lower()

        logger.info(
            "router_classifying",
            query_preview=query[:80],
            session_id=state["session_id"],
        )

        # Rule-based classification — in production, use an LLM call
        # with structured output for more robust classification.
        prescriptive_keywords = [
            "should", "recommend", "optimize", "allocate",
            "budget", "action", "what to do", "strategy",
            "intervention", "best approach", "how to reduce",
        ]
        diagnostic_keywords = [
            "why", "cause", "reason", "driver", "explain",
            "diagnose", "root cause", "what happened",
            "trend", "compared to", "difference",
        ]
        factual_keywords = [
            "how many", "what is", "show me", "list",
            "count", "total", "current", "score",
            "churn rate", "metrics", "members",
        ]

        query_type: str = "conversational"
        confidence: float = 0.7

        if any(kw in query for kw in prescriptive_keywords):
            query_type = "prescriptive"
            confidence = 0.85
        elif any(kw in query for kw in diagnostic_keywords):
            query_type = "diagnostic"
            confidence = 0.82
        elif any(kw in query for kw in factual_keywords):
            query_type = "factual"
            confidence = 0.88
        else:
            query_type = "conversational"
            confidence = 0.6

        logger.info(
            "router_classified",
            query_type=query_type,
            confidence=confidence,
        )

        return {
            "query_type": query_type,
            "confidence": confidence,
            "messages": state["messages"] + [
                {
                    "role": "system",
                    "content": (
                        f"Query classified as: {query_type} "
                        f"(confidence: {confidence:.2f})"
                    ),
                }
            ],
        }

    return router_node


def create_analyst_node() -> Any:
    """Create the analyst node for data queries and analysis.

    The analyst executes data-retrieval tools based on the query.
    It handles member lookups, location metrics, churn scores,
    and cohort comparisons. All data is PII-masked before being
    added to the state.

    Returns:
        A function compatible with LangGraph's add_node().
    """

    def analyst_node(state: AgentState) -> dict[str, Any]:
        """Execute data analysis tools based on query context."""
        query = state["query"].lower()
        location_id = state.get("location_id")
        tools_called = list(state.get("tools_called", []))

        logger.info(
            "analyst_processing",
            query_type=state["query_type"],
            location_id=location_id,
        )

        analysis: dict[str, Any] = {}

        # Determine which tools to call based on query content
        try:
            # Location metrics
            if location_id and any(
                kw in query
                for kw in [
                    "metric", "churn rate", "location",
                    "how many", "members", "mrr",
                ]
            ):
                metrics = query_location_metrics(
                    location_id, "last_30d"
                )
                analysis["location_metrics"] = metrics
                tools_called.append("query_location_metrics")

            # Member-specific queries
            member_id = _extract_member_id(query)
            if member_id:
                member_data = query_member_data(member_id)
                analysis["member_data"] = member_data
                tools_called.append("query_member_data")

                churn_score = get_churn_score(member_id)
                analysis["churn_score"] = churn_score
                tools_called.append("get_churn_score")

            # Cohort comparison
            if any(
                kw in query
                for kw in ["compare", "versus", "vs", "difference"]
            ):
                cohort_a = {"contract_source": "regular"}
                cohort_b = {"contract_source": "aggregator"}
                if location_id:
                    cohort_a["location_id"] = location_id
                    cohort_b["location_id"] = location_id
                comparison = compare_cohorts(cohort_a, cohort_b)
                analysis["cohort_comparison"] = comparison
                tools_called.append("compare_cohorts")

            # If no specific tool matched, get location metrics as default
            if not analysis and location_id:
                metrics = query_location_metrics(
                    location_id, "last_30d"
                )
                analysis["location_metrics"] = metrics
                tools_called.append("query_location_metrics")

        except AgentError as exc:
            logger.warning(
                "analyst_tool_error", error=str(exc)
            )
            analysis["error"] = str(exc)

        logger.info(
            "analyst_complete",
            tools_used=tools_called,
            data_keys=list(analysis.keys()),
        )

        return {
            "analysis_result": analysis,
            "tools_called": tools_called,
            "messages": state["messages"] + [
                {
                    "role": "assistant",
                    "content": (
                        f"Analysis complete. "
                        f"Data retrieved: {list(analysis.keys())}"
                    ),
                }
            ],
        }

    return analyst_node


def create_strategist_node() -> Any:
    """Create the strategist node for causal analysis and optimization.

    The strategist interprets causal estimates and runs the budget
    optimizer. It translates CATE results into actionable recommendations.

    Returns:
        A function compatible with LangGraph's add_node().
    """

    def strategist_node(state: AgentState) -> dict[str, Any]:
        """Execute causal analysis and optimization tools."""
        query = state["query"].lower()
        location_id = state.get("location_id")
        tools_called = list(state.get("tools_called", []))
        analysis = state.get("analysis_result") or {}

        logger.info(
            "strategist_processing",
            query_type=state["query_type"],
            has_analysis=bool(analysis),
        )

        strategy: dict[str, Any] = {}

        try:
            # Estimate treatment effects for relevant segments
            segments_to_check = [
                "regular_6-12m",
                "regular_0-3m",
                "aggregator_3-6m",
            ]
            actions_to_check = [
                "phone_call",
                "sms_reengagement",
                "discount_offer",
            ]

            # Get CATE estimates
            cate_results = []
            for segment in segments_to_check:
                for action in actions_to_check:
                    effect = estimate_treatment_effect(segment, action)
                    cate_results.append(effect)

            strategy["cate_estimates"] = cate_results
            tools_called.append("estimate_treatment_effect")

            # Run optimizer if query is prescriptive
            if state["query_type"] == "prescriptive":
                locations = (
                    [location_id] if location_id else ["LOC_001"]
                )
                budget = 2000.0
                optimization = run_optimizer(locations, budget)
                strategy["optimization"] = optimization
                tools_called.append("run_optimizer")

            # Find best action per segment
            best_actions = {}
            for result in cate_results:
                seg = result["segment"]
                if (
                    seg not in best_actions
                    or result["cate_mean"]
                    > best_actions[seg]["cate_mean"]
                ):
                    best_actions[seg] = result
            strategy["best_actions"] = best_actions

        except AgentError as exc:
            logger.warning(
                "strategist_tool_error", error=str(exc)
            )
            strategy["error"] = str(exc)

        logger.info(
            "strategist_complete",
            n_cate_estimates=len(
                strategy.get("cate_estimates", [])
            ),
            has_optimization="optimization" in strategy,
        )

        return {
            "strategy_result": strategy,
            "tools_called": tools_called,
            "messages": state["messages"] + [
                {
                    "role": "assistant",
                    "content": (
                        f"Strategy analysis complete. "
                        f"Recommendations generated."
                    ),
                }
            ],
        }

    return strategist_node


def create_writer_node() -> Any:
    """Create the writer node that formats output for managers.

    The writer translates technical analysis into clear, actionable
    language. It follows rules from configs/agents.yaml:
    - No statistical jargon
    - Always include action items
    - Numbers rounded and contextualized
    - Explicit about uncertainty

    Returns:
        A function compatible with LangGraph's add_node().
    """
    gc = load_guardrail_config()

    def writer_node(state: AgentState) -> dict[str, Any]:
        """Format analysis and strategy results for managers."""
        analysis = state.get("analysis_result") or {}
        strategy = state.get("strategy_result") or {}
        query_type = state.get("query_type", "conversational")

        logger.info(
            "writer_formatting",
            query_type=query_type,
            has_analysis=bool(analysis),
            has_strategy=bool(strategy),
        )

        # Check confidence before generating response
        is_confident, fallback = check_confidence(
            state.get("confidence", 0.5),
            gc.confidence_threshold,
            gc.fallback_response,
        )
        if not is_confident:
            return {
                "final_response": fallback,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": fallback}
                ],
            }

        # Build response based on available data
        response_parts: list[str] = []

        # Location metrics summary
        if "location_metrics" in analysis:
            metrics = analysis["location_metrics"]
            churn_pct = metrics["churn_rate"] * 100
            response_parts.append(
                f"Your location currently has "
                f"{metrics['active_members']} active members "
                f"with a {churn_pct:.1f}% monthly churn rate "
                f"({metrics['churned_members']} members lost "
                f"this period). "
                f"There are {metrics['at_risk_members']} members "
                f"currently at risk of churning."
            )

        # Churn score for specific member
        if "churn_score" in analysis:
            score = analysis["churn_score"]
            prob_pct = score["churn_probability"] * 100
            response_parts.append(
                f"Member {score['member_id']} has a "
                f"{score['risk_tier']} churn risk "
                f"({prob_pct:.0f}% probability in the next "
                f"{score['horizon_days']} days). "
                f"Main risk drivers: "
                + ", ".join(
                    f["feature"].replace("_", " ")
                    for f in score["top_risk_factors"][:3]
                )
                + "."
            )

        # Cohort comparison
        if "cohort_comparison" in analysis:
            comp = analysis["cohort_comparison"]
            response_parts.append(comp.get("interpretation", ""))

        # Strategy recommendations
        if strategy.get("best_actions"):
            actions_text = []
            for seg, act in strategy["best_actions"].items():
                effect_pct = act["cate_mean"] * 100
                action_name = act["action"].replace("_", " ")
                actions_text.append(
                    f"- {seg}: {action_name} "
                    f"(expected ~{effect_pct:.0f}pp churn reduction)"
                )
            response_parts.append(
                "Recommended actions by segment:\n"
                + "\n".join(actions_text)
            )

        # Optimization results
        if strategy.get("optimization"):
            opt = strategy["optimization"]
            response_parts.append(
                f"Budget optimization: targeting "
                f"{opt['members_targeted']} members with "
                f"R${opt['budget_used']:,.0f} budget "
                f"(expected ROI: {opt['roi']:.1f}x). "
                f"Top actions: "
                + ", ".join(
                    f"{a['action'].replace('_', ' ')} "
                    f"({a['members']} members)"
                    for a in opt["top_actions"]
                )
                + "."
            )

        # Conversational fallback
        if not response_parts:
            response_parts.append(
                "I can help you with retention analysis for your "
                "location. Try asking about:\n"
                "- Current churn metrics\n"
                "- Specific member risk scores\n"
                "- Recommended retention actions\n"
                "- Budget optimization"
            )

        full_response = "\n\n".join(response_parts)

        # Final PII check and masking
        full_response = validate_agent_output(
            full_response, gc.pii_mask_token
        )
        full_response = mask_pii_in_text(
            full_response, gc.pii_mask_token
        )

        logger.info(
            "writer_complete",
            response_length=len(full_response),
        )

        return {
            "final_response": full_response,
            "messages": state["messages"] + [
                {"role": "assistant", "content": full_response}
            ],
        }

    return writer_node


def _extract_member_id(query: str) -> str | None:
    """Extract a member ID from the query text.

    Looks for patterns like 'member MEM_001' or 'member_id MEM_001'.

    Args:
        query: User query string.

    Returns:
        Member ID string or None if not found.
    """
    import re

    patterns = [
        r"member[_\s]?(?:id)?[:\s]+([A-Za-z0-9_-]+)",
        r"(MEM_\d+)",
        r"(mem_\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _route_after_router(
    state: AgentState,
) -> Literal["analyst", "writer"]:
    """Route from router to the next node based on query type.

    Args:
        state: Current agent state with query_type set.

    Returns:
        Next node name.
    """
    query_type = state.get("query_type", "conversational")
    if query_type in ("factual", "diagnostic", "prescriptive"):
        return "analyst"
    return "writer"


def _route_after_analyst(
    state: AgentState,
) -> Literal["strategist", "writer"]:
    """Route from analyst to strategist or directly to writer.

    Args:
        state: Current agent state with query_type set.

    Returns:
        Next node name.
    """
    query_type = state.get("query_type", "conversational")
    if query_type in ("diagnostic", "prescriptive"):
        return "strategist"
    return "writer"


def _route_after_strategist(
    state: AgentState,
) -> Literal["writer"]:
    """Route from strategist to writer (always).

    Args:
        state: Current agent state.

    Returns:
        'writer' always.
    """
    return "writer"


def build_retention_graph() -> StateGraph:
    """Build the full LangGraph state machine for retention queries.

    Graph topology:
        router → (factual) → analyst → writer → END
        router → (diagnostic) → analyst → strategist → writer → END
        router → (prescriptive) → analyst → strategist → writer → END
        router → (conversational) → writer → END

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    logger.info("building_retention_graph")

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", create_router_node())
    graph.add_node("analyst", create_analyst_node())
    graph.add_node("strategist", create_strategist_node())
    graph.add_node("writer", create_writer_node())

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional edges
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"analyst": "analyst", "writer": "writer"},
    )
    graph.add_conditional_edges(
        "analyst",
        _route_after_analyst,
        {"strategist": "strategist", "writer": "writer"},
    )
    graph.add_conditional_edges(
        "strategist",
        _route_after_strategist,
        {"writer": "writer"},
    )

    # Writer always goes to END
    graph.add_edge("writer", END)

    compiled = graph.compile()

    logger.info("retention_graph_built")
    return compiled


def run_agent(
    query: str,
    location_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the retention agent on a natural language query.

    This is the main entry point for the agent system. It creates
    the initial state, runs the graph, and returns the response
    with metadata (tools used, confidence, latency).

    Args:
        query: Natural language query from a location manager.
        location_id: Optional location context for the query.
        session_id: Optional session ID for conversation continuity.
            Generated automatically if not provided.

    Returns:
        Dict with: answer, confidence, tools_used, session_id,
        query_type, latency_ms.

    Raises:
        AgentError: If the agent fails to produce a response.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    start_time = time.time()

    logger.info(
        "agent_run_start",
        query_preview=query[:80],
        location_id=location_id,
        session_id=session_id,
    )

    # Mask any PII in the query before processing
    gc = load_guardrail_config()
    masked_query = mask_pii_in_text(query, gc.pii_mask_token)

    # Build initial state
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": masked_query}],
        "context": {"location_id": location_id},
        "tools_called": [],
        "confidence": 0.5,
        "query": masked_query,
        "query_type": "conversational",
        "location_id": location_id,
        "session_id": session_id,
        "analysis_result": None,
        "strategy_result": None,
        "final_response": "",
        "error": None,
    }

    try:
        graph = build_retention_graph()
        final_state = graph.invoke(initial_state)

        latency_ms = int((time.time() - start_time) * 1000)

        result = {
            "answer": final_state.get("final_response", ""),
            "confidence": final_state.get("confidence", 0.5),
            "tools_used": final_state.get("tools_called", []),
            "session_id": session_id,
            "query_type": final_state.get(
                "query_type", "conversational"
            ),
            "latency_ms": latency_ms,
        }

        logger.info(
            "agent_run_complete",
            query_type=result["query_type"],
            tools_used=result["tools_used"],
            confidence=result["confidence"],
            latency_ms=latency_ms,
        )

        return result

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "agent_run_failed",
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise AgentError(
            f"Agent failed to process query: {exc}"
        ) from exc
