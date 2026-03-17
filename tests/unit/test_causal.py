"""Unit tests for the causal inference layer.

These tests verify DAG configuration, model building, and CATE
scenario sampling. Causal assumptions are critical — incorrect
assumptions lead to biased treatment effect estimates, which lead
to bad retention allocation decisions.
"""

import numpy as np
import pandas as pd
import pytest

from src.causal.dag import CausalDAGConfig, build_causal_model, load_dag_config
from src.exceptions import CausalConfigError
from src.optimization.allocator import sample_cate_scenarios


# ---------------------------------------------------------------------------
# DAG configuration loading
# ---------------------------------------------------------------------------


class TestDAGConfigLoading:
    """Test that DAG configuration is correctly parsed from YAML."""

    @pytest.fixture
    def valid_config_file(self, tmp_path: object) -> str:
        """Write a valid causal config YAML to a temp file."""
        content = """\
dag:
  treatment: "retention_action_taken"
  outcome: "churned_30d_post"
  confounders:
    - "visit_frequency_30d"
    - "days_since_last_visit"
    - "tenure_months"
  effect_modifiers:
    - "contract_source"
    - "plan_type"
  instruments: []
estimation:
  method: "causal_forest_dml"
"""
        config_file = tmp_path / "causal.yaml"
        config_file.write_text(content)
        return str(config_file)

    @pytest.fixture
    def missing_outcome_config(self, tmp_path: object) -> str:
        """Config missing the required 'outcome' field."""
        content = """\
dag:
  treatment: "retention_action_taken"
  confounders:
    - "visit_frequency_30d"
"""
        config_file = tmp_path / "causal_bad.yaml"
        config_file.write_text(content)
        return str(config_file)

    @pytest.fixture
    def missing_dag_section_config(self, tmp_path: object) -> str:
        """Config with no 'dag' section at all."""
        content = """\
estimation:
  method: "linear"
"""
        config_file = tmp_path / "no_dag.yaml"
        config_file.write_text(content)
        return str(config_file)

    @pytest.mark.unit
    def test_loads_valid_config(self, valid_config_file: str) -> None:
        """Valid YAML should produce a correct CausalDAGConfig."""
        config = load_dag_config(valid_config_file)
        assert config.treatment == "retention_action_taken"
        assert config.outcome == "churned_30d_post"
        assert len(config.confounders) == 3
        assert "visit_frequency_30d" in config.confounders
        assert "contract_source" in config.effect_modifiers
        assert config.instruments == []

    @pytest.mark.unit
    def test_raises_on_missing_outcome(
        self, missing_outcome_config: str
    ) -> None:
        """Missing required field should raise CausalConfigError."""
        with pytest.raises(CausalConfigError, match="Missing required field"):
            load_dag_config(missing_outcome_config)

    @pytest.mark.unit
    def test_raises_on_missing_dag_section(
        self, missing_dag_section_config: str
    ) -> None:
        """Missing 'dag' section should raise CausalConfigError."""
        with pytest.raises(CausalConfigError, match="Missing 'dag' section"):
            load_dag_config(missing_dag_section_config)

    @pytest.mark.unit
    def test_raises_on_missing_file(self) -> None:
        """Non-existent config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dag_config("/nonexistent/path/causal.yaml")

    @pytest.mark.unit
    def test_config_is_frozen(self) -> None:
        """CausalDAGConfig should be immutable (frozen dataclass)."""
        config = CausalDAGConfig(
            treatment="action",
            outcome="churned",
            confounders=["visit_freq"],
            effect_modifiers=["tenure"],
            instruments=[],
        )
        with pytest.raises(AttributeError):
            config.treatment = "other"

    @pytest.mark.unit
    def test_optional_fields_default_empty(
        self, tmp_path: object
    ) -> None:
        """effect_modifiers and instruments default to empty lists."""
        content = """\
dag:
  treatment: "action"
  outcome: "churned"
  confounders:
    - "visit_freq"
"""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(content)
        config = load_dag_config(str(config_file))
        assert config.effect_modifiers == []
        assert config.instruments == []


# ---------------------------------------------------------------------------
# DAG building
# ---------------------------------------------------------------------------


class TestDAGBuilding:
    """Test building a DoWhy CausalModel from config and data."""

    @pytest.fixture
    def dag_config(self) -> CausalDAGConfig:
        """A simple DAG config for testing."""
        return CausalDAGConfig(
            treatment="retention_action_taken",
            outcome="churned_30d_post",
            confounders=["visit_frequency_30d", "tenure_months"],
            effect_modifiers=["contract_source"],
            instruments=[],
        )

    @pytest.fixture
    def valid_causal_data(self) -> pd.DataFrame:
        """Data with all required DAG variables."""
        rng = np.random.default_rng(42)
        n = 200
        return pd.DataFrame({
            "retention_action_taken": rng.choice([0, 1], n),
            "churned_30d_post": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "visit_frequency_30d": rng.poisson(4, n).astype(float),
            "tenure_months": rng.uniform(1, 24, n),
            "contract_source": rng.choice(
                ["regular", "aggregator"], n, p=[0.7, 0.3]
            ),
        })

    @pytest.mark.unit
    def test_builds_model_with_valid_data(
        self,
        valid_causal_data: pd.DataFrame,
        dag_config: CausalDAGConfig,
    ) -> None:
        """CausalModel should be created without errors on valid data."""
        model = build_causal_model(valid_causal_data, dag_config)
        assert model is not None

    @pytest.mark.unit
    def test_raises_on_missing_variable(
        self, dag_config: CausalDAGConfig
    ) -> None:
        """Data missing a DAG variable should raise CausalConfigError."""
        data = pd.DataFrame({
            "retention_action_taken": [0, 1],
            "churned_30d_post": [0, 1],
            # Missing: visit_frequency_30d, tenure_months, contract_source
        })
        with pytest.raises(CausalConfigError, match="not found in data"):
            build_causal_model(data, dag_config)

    @pytest.mark.unit
    def test_missing_single_confounder_raises(
        self, dag_config: CausalDAGConfig
    ) -> None:
        """Even one missing confounder should cause failure."""
        data = pd.DataFrame({
            "retention_action_taken": [0, 1, 0],
            "churned_30d_post": [0, 1, 0],
            "visit_frequency_30d": [3.0, 5.0, 2.0],
            # Missing: tenure_months
            "contract_source": ["regular", "aggregator", "regular"],
        })
        with pytest.raises(CausalConfigError, match="tenure_months"):
            build_causal_model(data, dag_config)

    @pytest.mark.unit
    def test_extra_columns_in_data_ok(
        self,
        valid_causal_data: pd.DataFrame,
        dag_config: CausalDAGConfig,
    ) -> None:
        """Extra columns in the data should not cause issues."""
        data = valid_causal_data.copy()
        data["extra_column"] = 42
        data["another_extra"] = "hello"
        model = build_causal_model(data, dag_config)
        assert model is not None


# ---------------------------------------------------------------------------
# CATE scenario sampling
# ---------------------------------------------------------------------------


class TestCATEScenarioSampling:
    """Test treatment effect scenario generation for the stochastic optimizer.

    Scenarios sample from the confidence interval of CATE estimates
    to model uncertainty in treatment effects.
    """

    @pytest.mark.unit
    def test_correct_number_of_scenarios(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Should generate exactly n_scenarios scenarios."""
        scenarios = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=20, seed=42
        )
        assert len(scenarios) == 20

    @pytest.mark.unit
    def test_scenario_shape_matches_input(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Each scenario should have same rows as input estimates."""
        scenarios = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=5, seed=42
        )
        for scenario in scenarios:
            assert len(scenario) == len(sample_cate_estimates)

    @pytest.mark.unit
    def test_scenario_has_cate_realized_column(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Each scenario should add a 'cate_realized' column."""
        scenarios = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=3, seed=42
        )
        for scenario in scenarios:
            assert "cate_realized" in scenario.columns

    @pytest.mark.unit
    def test_cate_realized_bounded_0_1(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Realized CATE should be clipped to [0, 1]."""
        scenarios = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=50, seed=42
        )
        for scenario in scenarios:
            assert scenario["cate_realized"].min() >= 0.0
            assert scenario["cate_realized"].max() <= 1.0

    @pytest.mark.unit
    def test_scenarios_differ_from_each_other(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Different scenarios should have different realized values.

        This validates that sampling is actually introducing variation.
        """
        scenarios = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=10, seed=42
        )
        values_0 = scenarios[0]["cate_realized"].values
        values_1 = scenarios[1]["cate_realized"].values
        # Scenarios should not be identical (probability of this
        # is essentially zero with continuous sampling)
        assert not np.allclose(values_0, values_1)

    @pytest.mark.unit
    def test_seed_produces_reproducible_results(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Same seed should produce identical scenarios."""
        scenarios_a = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=5, seed=123
        )
        scenarios_b = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=5, seed=123
        )
        for sa, sb in zip(scenarios_a, scenarios_b):
            np.testing.assert_array_equal(
                sa["cate_realized"].values,
                sb["cate_realized"].values,
            )

    @pytest.mark.unit
    def test_different_seeds_produce_different_results(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Different seeds should produce different scenarios."""
        scenarios_a = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=5, seed=1
        )
        scenarios_b = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=5, seed=999
        )
        assert not np.allclose(
            scenarios_a[0]["cate_realized"].values,
            scenarios_b[0]["cate_realized"].values,
        )

    @pytest.mark.unit
    def test_original_columns_preserved(
        self, sample_cate_estimates: pd.DataFrame
    ) -> None:
        """Original columns (member_id, action, cate_mean, cate_std)
        should still be present in each scenario."""
        scenarios = sample_cate_scenarios(
            sample_cate_estimates, n_scenarios=3, seed=42
        )
        for scenario in scenarios:
            assert "member_id" in scenario.columns
            assert "action" in scenario.columns
            assert "cate_mean" in scenario.columns
            assert "cate_std" in scenario.columns
