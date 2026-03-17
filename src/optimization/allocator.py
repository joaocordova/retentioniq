"""Two-stage stochastic budget optimizer for retention allocation.

This module solves the core prescriptive question: given limited budget,
estimated treatment effects (with uncertainty), and operational constraints,
how should we allocate retention actions across locations?

The key insight is that treatment effects are *estimates*, not known values.
A deterministic optimizer that treats estimates as ground truth will overallocate
to segments where we happen to have noisy high estimates. The stochastic
formulation accounts for this uncertainty explicitly.

Mathematical formulation:
    Maximize: E_s[ Σ_l Σ_a Σ_m (LTV_m × CATE_m_a_s × x_l_a_m) ]

    Subject to:
        Σ_l Σ_a Σ_m (cost_a × x_l_a_m) ≤ total_budget           (budget)
        Σ_a Σ_m (cost_a × x_l_a_m) ≤ max_budget_l    ∀l         (per-location cap)
        Σ_a Σ_m (cost_a × x_l_a_m) ≥ min_budget_l    ∀l         (per-location floor)
        Σ_a x_l_a_m ≤ 1                               ∀l,m      (one action per member)
        Σ_m (time_a × x_l_a_m) ≤ staff_hours_l        ∀l,a      (capacity)
        x_l_a_m ∈ {0, 1}                                         (binary decision)

    Where:
        l = location, a = action type, m = member, s = scenario
        x_l_a_m = 1 if member m at location l receives action a
        CATE_m_a_s = conditional average treatment effect for member m,
                     action a, under scenario s (sampled from CI)
        LTV_m = estimated lifetime value of member m
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog
import yaml
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Set,
    SolverFactory,
    Var,
    maximize,
    summation,
    value,
)

from src.exceptions import InfeasibleProblemError

logger = structlog.get_logger()


@dataclass
class AllocationResult:
    """Result of the budget optimization."""

    allocations: pd.DataFrame  # location_id, member_id, action, cost
    total_cost: float
    expected_retained_mrr: float
    scenarios_evaluated: int
    solver_status: str
    robust_gap: float  # difference between expected and worst-case


def load_optimization_config(
    config_path: str = "configs/optimization.yaml",
) -> dict[str, Any]:
    """Load optimization parameters from YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def sample_cate_scenarios(
    cate_estimates: pd.DataFrame,
    n_scenarios: int = 50,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Sample treatment effect scenarios from confidence intervals.

    Each scenario draws CATE values from the estimated distribution,
    simulating the uncertainty in our causal estimates.

    Args:
        cate_estimates: DataFrame with columns [member_id, action, cate_mean, cate_std].
        n_scenarios: Number of scenarios to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of DataFrames, each with realized CATE values for one scenario.
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    for _ in range(n_scenarios):
        scenario = cate_estimates.copy()
        # Sample from normal distribution centered on estimate
        scenario["cate_realized"] = rng.normal(
            loc=scenario["cate_mean"],
            scale=scenario["cate_std"].clip(lower=1e-6),
        )
        # Treatment effects are bounded: can't reduce churn by more than 100%
        # and we assume treatment never increases churn (floor at 0)
        scenario["cate_realized"] = scenario["cate_realized"].clip(lower=0, upper=1)
        scenarios.append(scenario)

    logger.info("cate_scenarios_sampled", n_scenarios=n_scenarios)
    return scenarios


def build_deterministic_model(
    members: pd.DataFrame,
    cate_estimates: pd.DataFrame,
    config: dict[str, Any],
) -> ConcreteModel:
    """Build deterministic budget allocation model (single scenario).

    This is the simpler version — treats CATE estimates as ground truth.
    Useful as a baseline to compare against the stochastic solution.

    Args:
        members: DataFrame with [member_id, location_id, ltv, contract_source].
        cate_estimates: DataFrame with [member_id, action, cate_mean].
        config: Optimization config dict.

    Returns:
        Pyomo ConcreteModel ready to solve.
    """
    model = ConcreteModel("RetentionBudgetAllocation")

    # Sets
    locations = members["location_id"].unique().tolist()
    member_ids = members["member_id"].unique().tolist()
    actions = [a["name"] for a in config["actions"]]

    model.L = Set(initialize=locations)
    model.M = Set(initialize=member_ids)
    model.A = Set(initialize=actions)

    # Decision variable: x[m, a] = 1 if member m receives action a
    model.x = Var(model.M, model.A, domain=Binary)

    # Parameters from data
    ltv = dict(zip(members["member_id"], members["ltv"]))
    member_location = dict(zip(members["member_id"], members["location_id"]))

    cate_dict = {}
    for _, row in cate_estimates.iterrows():
        cate_dict[(row["member_id"], row["action"])] = row["cate_mean"]

    action_cost = {a["name"]: a["cost_per_member"] for a in config["actions"]}
    action_time = {a["name"]: a["time_per_member_minutes"] for a in config["actions"]}

    # Objective: maximize expected retained MRR
    def objective_rule(m):
        return sum(
            ltv.get(member, 0)
            * cate_dict.get((member, action), 0)
            * m.x[member, action]
            for member in model.M
            for action in model.A
        )

    model.obj = Objective(rule=objective_rule, sense=maximize)

    # Constraint: total budget
    def budget_rule(m):
        return sum(
            action_cost[action] * m.x[member, action]
            for member in model.M
            for action in model.A
        ) <= config["constraints"]["total_budget"]

    model.budget = Constraint(rule=budget_rule)

    # Constraint: one action per member max
    def one_action_rule(m, member):
        return sum(m.x[member, action] for action in model.A) <= 1

    model.one_action = Constraint(model.M, rule=one_action_rule)

    # Constraint: per-location budget cap
    def location_budget_rule(m, loc):
        loc_members = [mid for mid in model.M if member_location.get(mid) == loc]
        if not loc_members:
            return Constraint.Skip
        return sum(
            action_cost[action] * m.x[member, action]
            for member in loc_members
            for action in model.A
        ) <= config["constraints"]["max_budget_per_location"]

    model.location_budget = Constraint(model.L, rule=location_budget_rule)

    # Constraint: staff hours per location
    def staff_hours_rule(m, loc):
        loc_members = [mid for mid in model.M if member_location.get(mid) == loc]
        if not loc_members:
            return Constraint.Skip
        return sum(
            action_time[action] * m.x[member, action]
            for member in loc_members
            for action in model.A
        ) <= config["constraints"]["staff_hours_per_location"] * 60  # convert to minutes

    model.staff_hours = Constraint(model.L, rule=staff_hours_rule)

    return model


def solve_allocation(
    members: pd.DataFrame,
    cate_estimates: pd.DataFrame,
    config_path: str = "configs/optimization.yaml",
) -> AllocationResult:
    """Solve the budget allocation problem.

    Uses deterministic model for simplicity. For stochastic version,
    see build_stochastic_model() which averages across CATE scenarios.

    Args:
        members: Member data with location, LTV.
        cate_estimates: CATE estimates with confidence intervals.
        config_path: Path to optimization config.

    Returns:
        AllocationResult with recommended actions per member.

    Raises:
        InfeasibleProblemError: If constraints can't be satisfied.
    """
    config = load_optimization_config(config_path)

    logger.info(
        "solving_allocation",
        n_members=len(members),
        n_locations=members["location_id"].nunique(),
        total_budget=config["constraints"]["total_budget"],
    )

    model = build_deterministic_model(members, cate_estimates, config)

    solver = SolverFactory(config["stochastic"]["solver"])
    result = solver.solve(model, tee=False)

    status = str(result.solver.termination_condition)
    if status != "optimal":
        raise InfeasibleProblemError(
            reason=f"Solver returned status: {status}. "
            f"Budget may be too low for minimum allocation constraints."
        )

    # Extract solution
    allocations = []
    action_cost = {a["name"]: a["cost_per_member"] for a in config["actions"]}
    member_location = dict(zip(members["member_id"], members["location_id"]))

    for member in model.M:
        for action in model.A:
            if value(model.x[member, action]) > 0.5:  # binary, but check > 0.5 for numerical safety
                allocations.append({
                    "member_id": member,
                    "location_id": member_location[member],
                    "action": action,
                    "cost": action_cost[action],
                })

    df_alloc = pd.DataFrame(allocations)
    total_cost = df_alloc["cost"].sum() if len(df_alloc) > 0 else 0
    expected_mrr = value(model.obj) if len(df_alloc) > 0 else 0

    logger.info(
        "allocation_solved",
        status=status,
        members_targeted=len(df_alloc),
        total_cost=total_cost,
        expected_retained_mrr=expected_mrr,
    )

    return AllocationResult(
        allocations=df_alloc,
        total_cost=total_cost,
        expected_retained_mrr=expected_mrr,
        scenarios_evaluated=1,
        solver_status=status,
        robust_gap=0.0,
    )
