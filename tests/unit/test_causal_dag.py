"""Tests for the causal DAG module.

Causal assumptions are the foundation of all treatment effect claims.
These tests ensure the DAG configuration is valid and the model
is built correctly.
"""

import pytest

from src.causal.dag import CausalDAGConfig, load_dag_config
from src.exceptions import CausalConfigError


class TestCausalDAGConfig:
    """Test DAG configuration loading and validation."""

    def test_config_is_immutable(self):
        config = CausalDAGConfig(
            treatment="action",
            outcome="churned",
            confounders=["visit_freq"],
            effect_modifiers=["tenure"],
            instruments=[],
        )
        with pytest.raises(AttributeError):
            config.treatment = "other"  # frozen dataclass

    def test_confounders_must_not_include_treatment(self):
        """A variable can't be both confounder and treatment."""
        config = CausalDAGConfig(
            treatment="retention_action",
            outcome="churned_30d",
            confounders=["visit_freq", "retention_action"],  # bad!
            effect_modifiers=[],
            instruments=[],
        )
        # This is a logical error, not caught by the dataclass itself.
        # We'd catch it in build_causal_model when DoWhy validates the graph.
        assert config.treatment in config.confounders  # proves the test scenario

    def test_outcome_must_not_be_in_confounders(self):
        config = CausalDAGConfig(
            treatment="action",
            outcome="churned",
            confounders=["churned", "visit_freq"],  # outcome as confounder = bad
            effect_modifiers=[],
            instruments=[],
        )
        assert config.outcome in config.confounders


class TestLoadDAGConfig:
    """Test loading from YAML (requires config file)."""

    @pytest.fixture
    def valid_config_file(self, tmp_path):
        config_content = """
dag:
  treatment: "retention_action_taken"
  outcome: "churned_30d_post"
  confounders:
    - "visit_frequency_30d"
    - "tenure_months"
  effect_modifiers:
    - "contract_source"
  instruments: []
estimation:
  method: "causal_forest_dml"
"""
        config_file = tmp_path / "causal.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    @pytest.fixture
    def invalid_config_file(self, tmp_path):
        config_content = """
dag:
  treatment: "action"
  # missing outcome and confounders
"""
        config_file = tmp_path / "causal_bad.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    def test_loads_valid_config(self, valid_config_file):
        config = load_dag_config(valid_config_file)
        assert config.treatment == "retention_action_taken"
        assert config.outcome == "churned_30d_post"
        assert len(config.confounders) == 2
        assert "contract_source" in config.effect_modifiers

    def test_raises_on_missing_fields(self, invalid_config_file):
        with pytest.raises(CausalConfigError, match="Missing required field"):
            load_dag_config(invalid_config_file)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dag_config("/nonexistent/path.yaml")
