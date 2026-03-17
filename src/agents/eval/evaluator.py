"""Agent evaluation framework for RetentionIQ.

This module provides systematic evaluation of the agent's performance
across multiple dimensions:
    - Task success: does the agent answer the question correctly?
    - Tool selection: does it use the right tools for the job?
    - PII safety: does any PII leak through to the response?
    - Latency: does it respond within acceptable time?

Evaluation runs on a set of predefined scenarios that cover the
four query types (factual, diagnostic, prescriptive, conversational).
Scenarios are loaded from YAML for easy maintenance by the team.

Usage:
    scenarios = load_scenarios("configs/eval_scenarios.yaml")
    results = evaluate_agent(run_agent, scenarios)
    print(results["task_success_rate"])
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog
import yaml

from src.agents.guardrails import CPF_PATTERN, EMAIL_PATTERN, PHONE_PATTERN

logger = structlog.get_logger()


@dataclass
class EvalScenario:
    """A single evaluation scenario for the agent.

    Each scenario specifies a query, the expected behavior, and
    criteria for success.

    Attributes:
        query: The natural language question to ask the agent.
        expected_tools: List of tool names the agent should use.
        expected_answer_contains: Substrings that must appear in
            the agent's response for the scenario to pass.
        category: Query category (factual/diagnostic/prescriptive/
            conversational).
        location_id: Optional location context for the query.
        description: Human-readable description of what this tests.
    """

    query: str
    expected_tools: list[str]
    expected_answer_contains: list[str]
    category: str
    location_id: str | None = None
    description: str = ""


@dataclass
class ScenarioResult:
    """Result of evaluating a single scenario.

    Attributes:
        scenario: The evaluated scenario.
        task_success: Whether the response contained expected content.
        tool_accuracy: Fraction of expected tools that were used.
        pii_leaked: Whether PII was detected in the response.
        latency_ms: Response time in milliseconds.
        actual_tools: List of tools the agent actually used.
        response_preview: First 200 chars of the agent's response.
        error: Error message if the agent failed.
    """

    scenario: EvalScenario
    task_success: bool = False
    tool_accuracy: float = 0.0
    pii_leaked: bool = False
    latency_ms: int = 0
    actual_tools: list[str] = field(default_factory=list)
    response_preview: str = ""
    error: str | None = None


def load_scenarios(path: str) -> list[EvalScenario]:
    """Load evaluation scenarios from a YAML file.

    The YAML file should have a top-level 'scenarios' key containing
    a list of scenario definitions.

    Args:
        path: Path to the YAML file with scenarios.

    Returns:
        List of EvalScenario objects.

    Raises:
        FileNotFoundError: If the scenarios file doesn't exist.
        ValueError: If the YAML structure is invalid.
    """
    logger.info("loading_eval_scenarios", path=path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None or "scenarios" not in raw:
        raise ValueError(
            f"Scenarios file must have a 'scenarios' key. "
            f"Got: {list(raw.keys()) if raw else 'empty file'}"
        )

    scenarios = []
    for item in raw["scenarios"]:
        scenario = EvalScenario(
            query=item["query"],
            expected_tools=item.get("expected_tools", []),
            expected_answer_contains=item.get(
                "expected_answer_contains", []
            ),
            category=item.get("category", "factual"),
            location_id=item.get("location_id"),
            description=item.get("description", ""),
        )
        scenarios.append(scenario)

    logger.info(
        "eval_scenarios_loaded",
        n_scenarios=len(scenarios),
        categories=[s.category for s in scenarios],
    )

    return scenarios


def check_pii_leakage(response: str) -> bool:
    """Check if a response contains PII patterns.

    Scans for Brazilian PII patterns (CPF, email, phone). This is
    a critical safety check — any PII in the response means the
    guardrails failed.

    Args:
        response: Agent response text to check.

    Returns:
        True if PII is detected (BAD), False if clean (GOOD).
    """
    if CPF_PATTERN.search(response):
        logger.critical("pii_leak_detected", pii_type="cpf")
        return True
    if EMAIL_PATTERN.search(response):
        logger.critical("pii_leak_detected", pii_type="email")
        return True
    if PHONE_PATTERN.search(response):
        logger.critical("pii_leak_detected", pii_type="phone")
        return True

    # Also check for common PII-like patterns
    # Names after "member name:" or similar
    name_pattern = re.compile(
        r"(?:name|nome)[:\s]+[A-Z][a-z]+ [A-Z][a-z]+",
        re.IGNORECASE,
    )
    if name_pattern.search(response):
        logger.warning(
            "possible_pii_leak", pii_type="name_pattern"
        )
        return True

    return False


def _evaluate_single_scenario(
    agent_fn: Callable[..., dict[str, Any]],
    scenario: EvalScenario,
) -> ScenarioResult:
    """Evaluate a single scenario against the agent.

    Args:
        agent_fn: Agent function with signature
            (query, location_id, session_id) -> dict.
        scenario: The scenario to evaluate.

    Returns:
        ScenarioResult with all metrics.
    """
    result = ScenarioResult(scenario=scenario)

    start_time = time.time()

    try:
        agent_response = agent_fn(
            query=scenario.query,
            location_id=scenario.location_id,
            session_id=f"eval_{id(scenario)}",
        )

        result.latency_ms = int(
            (time.time() - start_time) * 1000
        )
        answer = agent_response.get("answer", "")
        result.response_preview = answer[:200]
        result.actual_tools = agent_response.get("tools_used", [])

        # Check task success: all expected substrings present
        if scenario.expected_answer_contains:
            answer_lower = answer.lower()
            matches = sum(
                1
                for expected in scenario.expected_answer_contains
                if expected.lower() in answer_lower
            )
            result.task_success = (
                matches == len(scenario.expected_answer_contains)
            )
        else:
            # If no expected content specified, just check non-empty
            result.task_success = len(answer.strip()) > 0

        # Check tool selection accuracy
        if scenario.expected_tools:
            actual_set = set(result.actual_tools)
            expected_set = set(scenario.expected_tools)
            if expected_set:
                correct = len(actual_set & expected_set)
                result.tool_accuracy = correct / len(expected_set)
            else:
                result.tool_accuracy = 1.0
        else:
            result.tool_accuracy = 1.0

        # Check PII leakage
        result.pii_leaked = check_pii_leakage(answer)

    except Exception as exc:
        result.latency_ms = int(
            (time.time() - start_time) * 1000
        )
        result.error = str(exc)
        result.task_success = False
        logger.error(
            "eval_scenario_failed",
            scenario=scenario.description,
            error=str(exc),
        )

    return result


def evaluate_agent(
    agent_fn: Callable[..., dict[str, Any]],
    scenarios: list[EvalScenario],
) -> dict[str, Any]:
    """Evaluate the agent across all scenarios.

    Runs each scenario, collects results, and computes aggregate
    metrics. This is the primary evaluation entry point.

    Args:
        agent_fn: Agent function with signature
            (query: str, location_id: str | None, session_id: str)
            -> dict[str, Any]. Must return a dict with 'answer'
            and 'tools_used' keys.
        scenarios: List of evaluation scenarios to run.

    Returns:
        Dict with aggregate metrics:
            - task_success_rate: fraction of scenarios that passed
            - tool_selection_accuracy: average tool accuracy
            - avg_latency: average response time in ms
            - pii_leak_rate: fraction of responses with PII
            - results_by_category: breakdown by query type
            - scenario_results: list of per-scenario results
    """
    logger.info(
        "evaluating_agent",
        n_scenarios=len(scenarios),
    )

    scenario_results: list[ScenarioResult] = []

    for scenario in scenarios:
        logger.info(
            "eval_running_scenario",
            description=scenario.description,
            category=scenario.category,
        )
        result = _evaluate_single_scenario(agent_fn, scenario)
        scenario_results.append(result)

    # Aggregate metrics
    n = len(scenario_results)
    if n == 0:
        return {
            "task_success_rate": 0.0,
            "tool_selection_accuracy": 0.0,
            "avg_latency": 0,
            "pii_leak_rate": 0.0,
            "results_by_category": {},
            "scenario_results": [],
        }

    task_success_rate = (
        sum(1 for r in scenario_results if r.task_success) / n
    )
    tool_selection_accuracy = (
        sum(r.tool_accuracy for r in scenario_results) / n
    )
    avg_latency = (
        sum(r.latency_ms for r in scenario_results) / n
    )
    pii_leak_rate = (
        sum(1 for r in scenario_results if r.pii_leaked) / n
    )

    # Breakdown by category
    categories: dict[str, list[ScenarioResult]] = {}
    for r in scenario_results:
        cat = r.scenario.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    results_by_category = {}
    for cat, cat_results in categories.items():
        cat_n = len(cat_results)
        results_by_category[cat] = {
            "n_scenarios": cat_n,
            "task_success_rate": (
                sum(1 for r in cat_results if r.task_success)
                / cat_n
            ),
            "tool_selection_accuracy": (
                sum(r.tool_accuracy for r in cat_results)
                / cat_n
            ),
            "avg_latency": (
                sum(r.latency_ms for r in cat_results)
                / cat_n
            ),
            "pii_leak_rate": (
                sum(1 for r in cat_results if r.pii_leaked)
                / cat_n
            ),
        }

    # Serialize scenario results for reporting
    serialized_results = []
    for r in scenario_results:
        serialized_results.append({
            "description": r.scenario.description,
            "category": r.scenario.category,
            "task_success": r.task_success,
            "tool_accuracy": r.tool_accuracy,
            "pii_leaked": r.pii_leaked,
            "latency_ms": r.latency_ms,
            "actual_tools": r.actual_tools,
            "response_preview": r.response_preview,
            "error": r.error,
        })

    evaluation = {
        "task_success_rate": task_success_rate,
        "tool_selection_accuracy": tool_selection_accuracy,
        "avg_latency": int(avg_latency),
        "pii_leak_rate": pii_leak_rate,
        "n_scenarios": n,
        "results_by_category": results_by_category,
        "scenario_results": serialized_results,
    }

    logger.info(
        "evaluation_complete",
        task_success_rate=task_success_rate,
        tool_selection_accuracy=tool_selection_accuracy,
        avg_latency_ms=int(avg_latency),
        pii_leak_rate=pii_leak_rate,
    )

    return evaluation
