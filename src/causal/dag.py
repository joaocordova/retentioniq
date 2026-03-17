"""Causal DAG definition for the retention problem.

This module defines the causal graph that encodes our domain assumptions
about what causes churn and how retention actions affect it. The DAG is
the foundation of all causal claims — no estimation without a declared model.

References:
    - Pearl, J. (2009). Causality. Cambridge University Press.
    - Hernán, M. & Robins, J. (2020). Causal Inference: What If.
"""

from dataclasses import dataclass

import dowhy
import yaml
from dowhy import CausalModel

from src.exceptions import CausalConfigError


@dataclass(frozen=True)
class CausalDAGConfig:
    """Immutable configuration for a causal DAG."""

    treatment: str
    outcome: str
    confounders: list[str]
    effect_modifiers: list[str]
    instruments: list[str]


def load_dag_config(config_path: str = "configs/causal.yaml") -> CausalDAGConfig:
    """Load causal DAG configuration from YAML.

    Args:
        config_path: Path to the causal config file.

    Returns:
        CausalDAGConfig with validated fields.

    Raises:
        CausalConfigError: If required fields are missing or invalid.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    dag_config = raw.get("dag")
    if dag_config is None:
        raise CausalConfigError("Missing 'dag' section in causal config")

    required_fields = ["treatment", "outcome", "confounders"]
    for field in required_fields:
        if field not in dag_config:
            raise CausalConfigError(f"Missing required field: dag.{field}")

    return CausalDAGConfig(
        treatment=dag_config["treatment"],
        outcome=dag_config["outcome"],
        confounders=dag_config["confounders"],
        effect_modifiers=dag_config.get("effect_modifiers", []),
        instruments=dag_config.get("instruments", []),
    )


def build_causal_model(
    data: "pd.DataFrame",
    config: CausalDAGConfig,
    segment: str | None = None,
) -> CausalModel:
    """Build a DoWhy CausalModel from configuration and data.

    The model encodes our assumptions as a DAG. DoWhy will use this
    to identify the estimand (what we need to estimate) and validate
    our assumptions via refutation tests.

    Args:
        data: DataFrame with all variables referenced in the DAG.
        config: DAG configuration specifying treatment, outcome, confounders.
        segment: Optional segment filter ('regular' or 'aggregator').
            If provided, uses segment-specific confounder adjustments.

    Returns:
        DoWhy CausalModel ready for identification and estimation.

    Raises:
        CausalConfigError: If data columns don't match DAG variables.
    """
    # Validate that all DAG variables exist in the data
    all_vars = [config.treatment, config.outcome] + config.confounders + config.effect_modifiers
    missing = [v for v in all_vars if v not in data.columns]
    if missing:
        raise CausalConfigError(
            f"Variables in DAG not found in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    # Build the graph specification string (GML format for DoWhy)
    # Every confounder has edges to both treatment and outcome
    edges = []
    for confounder in config.confounders:
        edges.append(f'"{confounder}" -> "{config.treatment}"')
        edges.append(f'"{confounder}" -> "{config.outcome}"')

    # Treatment causes outcome
    edges.append(f'"{config.treatment}" -> "{config.outcome}"')

    # Effect modifiers also affect outcome
    for modifier in config.effect_modifiers:
        edges.append(f'"{modifier}" -> "{config.outcome}"')

    graph_str = "digraph { " + "; ".join(edges) + " }"

    model = CausalModel(
        data=data,
        treatment=config.treatment,
        outcome=config.outcome,
        graph=graph_str,
        effect_modifiers=config.effect_modifiers if config.effect_modifiers else None,
    )

    return model
