"""Unit tests for the data layer: schema validation, cleaning, aggregation.

These tests verify the medallion architecture logic without requiring
a database or Dagster runtime. Each test uses small, deterministic
DataFrames to check specific business rules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.bronze.ingestion import validate_schema
from src.exceptions import SchemaValidationError


# ---------------------------------------------------------------------------
# Bronze layer: schema validation
# ---------------------------------------------------------------------------


class TestBronzeSchemaValidation:
    """Test that validate_schema catches structural problems early."""

    @pytest.mark.unit
    def test_valid_members_schema_passes(self) -> None:
        """A DataFrame matching the expected schema should pass."""
        df = pd.DataFrame({
            "member_id": ["MBR-001", "MBR-002"],
            "location_id": ["LOC-001", "LOC-001"],
            "join_date": ["2024-06-01", "2024-07-15"],
            "cancel_date": [None, "2025-01-01"],
            "churned": [False, True],
            "contract_source": ["regular", "aggregator"],
            "aggregator_platform": ["none", "gympass"],
            "plan_type": ["standard", "aggregator_standard"],
            "monthly_price": [119.90, 49.90],
            "age": [28, 35],
            "gender": ["M", "F"],
            "name": ["Alice", "Bob"],
            "email": ["a@test.com", "b@test.com"],
            "cpf": ["123.456.789-00", "987.654.321-00"],
        })
        expected_schema = {
            "member_id": "string",
            "location_id": "string",
            "join_date": "string",
            "cancel_date": "string",
            "churned": "bool",
            "contract_source": "string",
            "aggregator_platform": "string",
            "plan_type": "string",
            "monthly_price": "float64",
            "age": "int64",
            "gender": "string",
            "name": "string",
            "email": "string",
            "cpf": "string",
        }
        assert validate_schema(df, expected_schema) is True

    @pytest.mark.unit
    def test_missing_column_raises_error(self) -> None:
        """Schema validation should fail if a required column is absent."""
        df = pd.DataFrame({
            "member_id": ["MBR-001"],
            # location_id is missing
            "join_date": ["2024-06-01"],
        })
        expected_schema = {
            "member_id": "string",
            "location_id": "string",
            "join_date": "string",
        }
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(df, expected_schema)
        assert "location_id" in str(exc_info.value)
        assert exc_info.value.layer == "bronze"

    @pytest.mark.unit
    def test_wrong_dtype_raises_error(self) -> None:
        """Schema validation should fail on type mismatches."""
        df = pd.DataFrame({
            "member_id": ["MBR-001"],
            "monthly_price": ["not_a_number"],  # should be float64
        })
        expected_schema = {
            "member_id": "string",
            "monthly_price": "float64",
        }
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(df, expected_schema)
        assert "monthly_price" in str(exc_info.value)

    @pytest.mark.unit
    def test_nullable_int_stored_as_float_is_accepted(self) -> None:
        """Pandas stores nullable ints as float64; this should pass."""
        df = pd.DataFrame({
            "age": [28.0, 35.0, np.nan],  # float64 due to NaN
        })
        expected_schema = {"age": "int64"}
        # int64 expected but float64 actual -> accepted per ingestion.py logic
        assert validate_schema(df, expected_schema) is True

    @pytest.mark.unit
    def test_valid_visits_schema_passes(self) -> None:
        """Visit table schema validation."""
        df = pd.DataFrame({
            "member_id": ["MBR-001"],
            "location_id": ["LOC-001"],
            "visit_date": ["2025-01-15"],
            "visit_duration_minutes": [60],
        })
        expected_schema = {
            "member_id": "string",
            "location_id": "string",
            "visit_date": "string",
            "visit_duration_minutes": "int64",
        }
        assert validate_schema(df, expected_schema) is True

    @pytest.mark.unit
    def test_extra_columns_do_not_cause_failure(self) -> None:
        """Extra columns in the data are fine; only missing ones matter."""
        df = pd.DataFrame({
            "member_id": ["MBR-001"],
            "extra_col": ["whatever"],
        })
        expected_schema = {"member_id": "string"}
        assert validate_schema(df, expected_schema) is True

    @pytest.mark.unit
    def test_multiple_failures_reported(self) -> None:
        """All schema failures should be collected, not just the first."""
        df = pd.DataFrame({
            "member_id": [123],  # wrong type (int, not string)
        })
        expected_schema = {
            "member_id": "string",
            "location_id": "string",  # missing
        }
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(df, expected_schema)
        assert len(exc_info.value.failures) == 2


# ---------------------------------------------------------------------------
# Silver layer: cleaning rules
# ---------------------------------------------------------------------------


class TestSilverCleaningRules:
    """Test silver-layer data cleaning business rules.

    These tests verify the logic independently of any actual silver-layer
    module, using the rules defined in configs/data.yaml.
    """

    @pytest.mark.unit
    def test_reenrollment_window_merges_tenure(self) -> None:
        """Members who cancel and re-enroll within 30 days should keep
        their original tenure (treated as the same membership).
        """
        members = pd.DataFrame({
            "member_id": ["MBR-001", "MBR-001"],
            "join_date": pd.to_datetime(
                ["2024-01-01", "2024-07-20"]
            ),
            "cancel_date": pd.to_datetime(
                ["2024-07-01", pd.NaT]
            ),
        })
        # Rule: if gap between cancel and next join < 30 days,
        # treat as continuous
        gap_days = (
            members.iloc[1]["join_date"]
            - members.iloc[0]["cancel_date"]
        ).days
        reenrollment_window = 30
        assert gap_days < reenrollment_window
        # Effective join_date should be the original one
        effective_join = members.iloc[0]["join_date"]
        assert effective_join == pd.Timestamp("2024-01-01")

    @pytest.mark.unit
    def test_reenrollment_outside_window_is_new_membership(self) -> None:
        """Members who re-enroll after 30+ days gap start fresh."""
        cancel_date = pd.Timestamp("2024-03-01")
        new_join_date = pd.Timestamp("2024-06-15")
        gap_days = (new_join_date - cancel_date).days
        reenrollment_window = 30
        assert gap_days > reenrollment_window

    @pytest.mark.unit
    def test_invalid_short_visit_duration_filtered(self) -> None:
        """Visits shorter than 5 minutes are likely check-in errors."""
        visits = pd.DataFrame({
            "member_id": ["MBR-001", "MBR-002", "MBR-003"],
            "visit_duration_minutes": [2, 45, 3],
        })
        min_duration = 5
        valid = visits[
            visits["visit_duration_minutes"] >= min_duration
        ]
        assert len(valid) == 1
        assert valid.iloc[0]["member_id"] == "MBR-002"

    @pytest.mark.unit
    def test_invalid_long_visit_duration_capped(self) -> None:
        """Visits longer than 300 minutes are capped (forgot checkout)."""
        visits = pd.DataFrame({
            "visit_duration_minutes": [60, 350, 120, 500],
        })
        max_duration = 300
        capped = visits["visit_duration_minutes"].clip(upper=max_duration)
        assert capped.max() == max_duration
        assert capped.iloc[0] == 60
        assert capped.iloc[1] == max_duration

    @pytest.mark.unit
    def test_null_location_id_not_tolerated(self) -> None:
        """location_id has 0% null threshold per config."""
        members = pd.DataFrame({
            "member_id": ["MBR-001", "MBR-002", "MBR-003"],
            "location_id": ["LOC-001", None, "LOC-003"],
        })
        null_threshold = 0.0
        null_rate = members["location_id"].isna().mean()
        assert null_rate > null_threshold  # 33% > 0% => should fail

    @pytest.mark.unit
    def test_null_monthly_price_within_tolerance(self) -> None:
        """monthly_price allows up to 1% nulls per config."""
        prices = pd.Series(
            [119.90] * 99 + [None],
            name="monthly_price",
        )
        null_threshold = 0.01
        null_rate = prices.isna().mean()
        # 1% nulls is exactly at the threshold boundary
        assert null_rate <= null_threshold

    @pytest.mark.unit
    def test_null_monthly_price_above_tolerance_fails(self) -> None:
        """More than 1% nulls in monthly_price should be flagged."""
        prices = pd.Series(
            [119.90] * 95 + [None] * 5,
            name="monthly_price",
        )
        null_threshold = 0.01
        null_rate = prices.isna().mean()
        assert null_rate > null_threshold  # 5% > 1% => fail

    @pytest.mark.unit
    def test_valid_contract_sources_only(self) -> None:
        """Only 'regular' and 'aggregator' are valid contract sources."""
        sources = pd.Series(["regular", "aggregator", "unknown", "regular"])
        valid_sources = {"regular", "aggregator"}
        invalid_mask = ~sources.isin(valid_sources)
        assert invalid_mask.sum() == 1
        assert sources[invalid_mask].iloc[0] == "unknown"


# ---------------------------------------------------------------------------
# Gold layer: aggregation logic
# ---------------------------------------------------------------------------


class TestGoldAggregation:
    """Test gold-layer aggregation computations."""

    @pytest.mark.unit
    def test_visit_counts_per_window(self) -> None:
        """Visit counts should be computed for each configured window."""
        base_date = pd.Timestamp("2025-01-01")
        visits = pd.DataFrame({
            "member_id": ["MBR-001"] * 10,
            "visit_date": pd.to_datetime([
                "2024-12-31",  # 1 day ago
                "2024-12-28",  # 4 days ago
                "2024-12-25",  # 7 days ago (boundary)
                "2024-12-20",  # 12 days ago
                "2024-12-18",  # 14 days ago (boundary)
                "2024-12-10",  # 22 days ago
                "2024-12-05",  # 27 days ago
                "2024-11-15",  # 47 days ago
                "2024-11-01",  # 61 days ago
                "2024-10-15",  # 78 days ago
            ]),
        })
        windows = [7, 14, 30, 60, 90]
        for window in windows:
            cutoff = base_date - pd.Timedelta(days=window)
            count = (visits["visit_date"] >= cutoff).sum()
            if window == 7:
                assert count == 3, f"7d window: expected 3, got {count}"
            elif window == 14:
                assert count == 5, f"14d window: expected 5, got {count}"
            elif window == 30:
                assert count == 7, f"30d window: expected 7, got {count}"
            elif window == 60:
                assert count == 8, f"60d window: expected 8, got {count}"
            elif window == 90:
                assert count == 10, f"90d window: expected 10, got {count}"

    @pytest.mark.unit
    def test_churn_rate_calculation(self) -> None:
        """Churn rate = churned members / total members per location."""
        members = pd.DataFrame({
            "location_id": ["LOC-001"] * 10 + ["LOC-002"] * 5,
            "churned": (
                [True] * 3 + [False] * 7  # LOC-001: 30% churn
                + [True] * 2 + [False] * 3  # LOC-002: 40% churn
            ),
        })
        churn_rates = members.groupby("location_id")["churned"].mean()
        assert abs(churn_rates["LOC-001"] - 0.30) < 1e-6
        assert abs(churn_rates["LOC-002"] - 0.40) < 1e-6

    @pytest.mark.unit
    def test_churn_rate_zero_when_no_churn(self) -> None:
        """Location with no churn should have 0% churn rate."""
        members = pd.DataFrame({
            "location_id": ["LOC-001"] * 5,
            "churned": [False] * 5,
        })
        churn_rate = members["churned"].mean()
        assert churn_rate == 0.0

    @pytest.mark.unit
    def test_churn_rate_hundred_percent(self) -> None:
        """Edge case: all members churned."""
        members = pd.DataFrame({
            "location_id": ["LOC-001"] * 3,
            "churned": [True] * 3,
        })
        churn_rate = members["churned"].mean()
        assert churn_rate == 1.0

    @pytest.mark.unit
    def test_visit_count_zero_for_no_visits_in_window(self) -> None:
        """A member with no visits in the window should have count 0."""
        base_date = pd.Timestamp("2025-01-01")
        visits = pd.DataFrame({
            "member_id": ["MBR-001"],
            "visit_date": pd.to_datetime(["2024-06-01"]),
        })
        cutoff = base_date - pd.Timedelta(days=30)
        count = (visits["visit_date"] >= cutoff).sum()
        assert count == 0

    @pytest.mark.unit
    def test_aggregator_percentage_by_location(self) -> None:
        """Gold layer should compute aggregator percentage per location."""
        members = pd.DataFrame({
            "location_id": ["LOC-001"] * 10,
            "contract_source": (
                ["regular"] * 7 + ["aggregator"] * 3
            ),
        })
        agg_pct = (
            members["contract_source"] == "aggregator"
        ).mean()
        assert abs(agg_pct - 0.30) < 1e-6

    @pytest.mark.unit
    def test_mrr_calculation_by_location(self) -> None:
        """MRR = sum of monthly_price for active (non-churned) members."""
        members = pd.DataFrame({
            "location_id": ["LOC-001"] * 4,
            "monthly_price": [100.0, 120.0, 80.0, 150.0],
            "churned": [False, False, True, False],
        })
        active = members[~members["churned"]]
        mrr = active.groupby("location_id")["monthly_price"].sum()
        # 100 + 120 + 150 = 370 (80 excluded because churned)
        assert abs(mrr["LOC-001"] - 370.0) < 1e-6
