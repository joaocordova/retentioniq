"""Tests for the guardrails module.

These tests verify that PII protection works correctly.
PII leakage is a zero-tolerance issue — every edge case matters.
"""

import pandas as pd
import pytest

from src.agents.guardrails import (
    check_confidence,
    mask_pii_in_dataframe,
    mask_pii_in_text,
    validate_agent_output,
    validate_sql_query,
)
from src.exceptions import GuardrailViolationError


# ---------------------------------------------------------------------------
# PII masking in free text
# ---------------------------------------------------------------------------


class TestMaskPIIText:
    """Test PII masking in free text."""

    @pytest.mark.unit
    def test_masks_cpf_with_dots(self) -> None:
        """CPF in standard format (xxx.xxx.xxx-xx) must be masked."""
        text = "Member Joao, CPF 123.456.789-00, is at risk."
        result = mask_pii_in_text(text)
        assert "123.456.789-00" not in result
        assert "[REDACTED]" in result
        assert "Member Joao" in result  # name stays (not regex-detectable)

    @pytest.mark.unit
    def test_masks_cpf_without_formatting(self) -> None:
        """CPF without dots/dash (11 digits) must also be masked."""
        text = "CPF: 12345678900"
        result = mask_pii_in_text(text)
        assert "12345678900" not in result

    @pytest.mark.unit
    def test_masks_email(self) -> None:
        """Email addresses must be masked."""
        text = "Contact: joao.silva@gmail.com for details"
        result = mask_pii_in_text(text)
        assert "joao.silva@gmail.com" not in result
        assert "[REDACTED]" in result

    @pytest.mark.unit
    def test_masks_email_with_subdomain(self) -> None:
        """Emails with subdomains must be masked."""
        text = "Send to user@mail.company.co.br"
        result = mask_pii_in_text(text)
        assert "user@mail.company.co.br" not in result

    @pytest.mark.unit
    def test_masks_phone_with_area_code(self) -> None:
        """Phone in (XX) XXXXX-XXXX format must be masked."""
        text = "Phone: (11) 98765-4321"
        result = mask_pii_in_text(text)
        assert "98765-4321" not in result

    @pytest.mark.unit
    def test_masks_phone_without_formatting(self) -> None:
        """Phone without parentheses/dash must be masked."""
        text = "Call 11987654321"
        result = mask_pii_in_text(text)
        assert "11987654321" not in result

    @pytest.mark.unit
    def test_masks_multiple_pii_types(self) -> None:
        """All PII types in one string should all be masked."""
        text = (
            "CPF 123.456.789-00, email test@test.com, "
            "phone (11) 91234-5678"
        )
        result = mask_pii_in_text(text)
        assert result.count("[REDACTED]") == 3

    @pytest.mark.unit
    def test_no_pii_returns_unchanged(self) -> None:
        """Text without PII should pass through unchanged."""
        text = "Location 42 has 150 active members with 12% churn rate."
        result = mask_pii_in_text(text)
        assert result == text

    @pytest.mark.unit
    def test_custom_mask_token(self) -> None:
        """Custom mask token should replace PII."""
        text = "CPF: 123.456.789-00"
        result = mask_pii_in_text(text, mask_token="***")
        assert "***" in result
        assert "123.456.789-00" not in result

    @pytest.mark.unit
    def test_empty_string(self) -> None:
        """Empty input should return empty output."""
        assert mask_pii_in_text("") == ""

    @pytest.mark.unit
    def test_preserves_surrounding_context(self) -> None:
        """Non-PII context around masked items should be preserved."""
        text = "Before 123.456.789-00 After"
        result = mask_pii_in_text(text)
        assert result.startswith("Before ")
        assert result.endswith(" After")

    @pytest.mark.unit
    def test_multiple_cpfs_all_masked(self) -> None:
        """Multiple CPFs in one text should all be masked."""
        text = "CPF1: 123.456.789-00 CPF2: 987.654.321-11"
        result = mask_pii_in_text(text)
        assert "123.456.789-00" not in result
        assert "987.654.321-11" not in result
        assert result.count("[REDACTED]") == 2


# ---------------------------------------------------------------------------
# SQL injection blocking
# ---------------------------------------------------------------------------


class TestValidateSQL:
    """Test SQL injection prevention."""

    @pytest.mark.unit
    def test_allows_select(self) -> None:
        """Simple SELECT queries should be allowed."""
        query = (
            "SELECT member_id, churn_prob FROM gold.churn_scores "
            "WHERE location_id = '42'"
        )
        result = validate_sql_query(query, ["DROP TABLE", "DELETE FROM"])
        assert result == query

    @pytest.mark.unit
    def test_allows_cte(self) -> None:
        """WITH (CTE) queries should be allowed."""
        query = (
            "WITH recent AS (SELECT * FROM visits "
            "WHERE date > '2026-01-01') SELECT * FROM recent"
        )
        result = validate_sql_query(query, ["DROP TABLE", "DELETE FROM"])
        assert result == query

    @pytest.mark.unit
    def test_blocks_drop_table(self) -> None:
        """DROP TABLE must be blocked."""
        query = "DROP TABLE members; SELECT 1"
        with pytest.raises(GuardrailViolationError, match="sql_safety"):
            validate_sql_query(query, ["DROP TABLE"])

    @pytest.mark.unit
    def test_blocks_delete(self) -> None:
        """DELETE FROM must be blocked."""
        query = "DELETE FROM members WHERE id = 1"
        with pytest.raises(GuardrailViolationError, match="sql_safety"):
            validate_sql_query(query, ["DELETE FROM"])

    @pytest.mark.unit
    def test_blocks_update(self) -> None:
        """UPDATE ... SET must be blocked."""
        query = "UPDATE members SET status = 'active'"
        with pytest.raises(GuardrailViolationError, match="sql_safety"):
            validate_sql_query(query, ["UPDATE.*SET"])

    @pytest.mark.unit
    def test_blocks_alter_table(self) -> None:
        """ALTER TABLE must be blocked."""
        query = "ALTER TABLE members ADD COLUMN hacked bool"
        with pytest.raises(GuardrailViolationError, match="sql_safety"):
            validate_sql_query(query, ["ALTER TABLE"])

    @pytest.mark.unit
    def test_blocks_non_select_start(self) -> None:
        """Queries not starting with SELECT or WITH must be rejected."""
        query = "INSERT INTO members VALUES (1, 'test')"
        with pytest.raises(GuardrailViolationError, match="sql_readonly"):
            validate_sql_query(query, [])

    @pytest.mark.unit
    def test_case_insensitive_blocking(self) -> None:
        """Blocking should be case-insensitive."""
        query = "drop table members"
        with pytest.raises(GuardrailViolationError):
            validate_sql_query(query, ["DROP TABLE"])

    @pytest.mark.unit
    def test_returns_query_when_valid(self) -> None:
        """Valid queries should be returned as-is for downstream use."""
        query = "SELECT COUNT(*) FROM members"
        result = validate_sql_query(query, ["DROP TABLE"])
        assert result == query

    @pytest.mark.unit
    def test_blocks_truncate(self) -> None:
        """TRUNCATE should be caught by the read-only check."""
        query = "TRUNCATE TABLE members"
        with pytest.raises(GuardrailViolationError, match="sql_readonly"):
            validate_sql_query(query, [])


# ---------------------------------------------------------------------------
# Read-only SQL validation
# ---------------------------------------------------------------------------


class TestReadOnlySQL:
    """Ensure only SELECT and WITH queries pass the read-only check."""

    @pytest.mark.unit
    def test_select_with_subquery_allowed(self) -> None:
        """SELECT with a subquery should be fine."""
        query = (
            "SELECT * FROM (SELECT member_id FROM members) sub"
        )
        result = validate_sql_query(query, [])
        assert "member_id" in result

    @pytest.mark.unit
    def test_with_recursive_allowed(self) -> None:
        """WITH RECURSIVE is still a read query."""
        query = (
            "WITH RECURSIVE cte AS (SELECT 1 AS n UNION ALL "
            "SELECT n+1 FROM cte WHERE n < 10) SELECT * FROM cte"
        )
        result = validate_sql_query(query, [])
        assert result == query

    @pytest.mark.unit
    def test_create_table_blocked(self) -> None:
        """CREATE TABLE must be blocked."""
        query = "CREATE TABLE evil (id int)"
        with pytest.raises(GuardrailViolationError, match="sql_readonly"):
            validate_sql_query(query, [])


# ---------------------------------------------------------------------------
# DataFrame PII column masking
# ---------------------------------------------------------------------------


class TestMaskPIIDataFrame:
    """Test PII column masking in DataFrames."""

    @pytest.mark.unit
    def test_masks_specified_columns(self) -> None:
        """PII columns should be entirely replaced with mask token."""
        df = pd.DataFrame({
            "member_id": ["MBR-001", "MBR-002"],
            "name": ["Alice", "Bob"],
            "email": ["alice@test.com", "bob@test.com"],
            "cpf": ["123.456.789-00", "987.654.321-11"],
            "churn_prob": [0.8, 0.3],
        })
        result = mask_pii_in_dataframe(
            df,
            pii_columns=["name", "email", "cpf"],
        )
        # PII columns should all be [REDACTED]
        assert (result["name"] == "[REDACTED]").all()
        assert (result["email"] == "[REDACTED]").all()
        assert (result["cpf"] == "[REDACTED]").all()
        # Non-PII columns should be untouched
        assert list(result["member_id"]) == ["MBR-001", "MBR-002"]
        assert list(result["churn_prob"]) == [0.8, 0.3]

    @pytest.mark.unit
    def test_does_not_modify_original(self) -> None:
        """Masking should return a copy, not modify the original."""
        df = pd.DataFrame({
            "name": ["Alice"],
            "score": [0.9],
        })
        result = mask_pii_in_dataframe(df, pii_columns=["name"])
        assert df["name"].iloc[0] == "Alice"
        assert result["name"].iloc[0] == "[REDACTED]"

    @pytest.mark.unit
    def test_ignores_missing_pii_columns(self) -> None:
        """If a PII column doesn't exist in the DataFrame, skip it."""
        df = pd.DataFrame({
            "member_id": ["MBR-001"],
            "score": [0.5],
        })
        # "name" is not in df, should not raise
        result = mask_pii_in_dataframe(
            df,
            pii_columns=["name", "email"],
        )
        assert list(result.columns) == ["member_id", "score"]

    @pytest.mark.unit
    def test_custom_mask_token_in_dataframe(self) -> None:
        """Custom mask token should work for DataFrame masking."""
        df = pd.DataFrame({
            "cpf": ["123.456.789-00"],
        })
        result = mask_pii_in_dataframe(
            df,
            pii_columns=["cpf"],
            mask_token="***MASKED***",
        )
        assert result["cpf"].iloc[0] == "***MASKED***"

    @pytest.mark.unit
    def test_empty_pii_list_returns_unchanged(self) -> None:
        """No PII columns to mask -> DataFrame returned as-is."""
        df = pd.DataFrame({"score": [0.5, 0.6]})
        result = mask_pii_in_dataframe(df, pii_columns=[])
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# Agent output validation
# ---------------------------------------------------------------------------


class TestValidateAgentOutput:
    """Test output validation — last line of defense for PII."""

    @pytest.mark.unit
    def test_clean_output_passes(self) -> None:
        """Output without PII should pass through unchanged."""
        output = (
            "Location 42 has 15% churn rate. "
            "Focus on members in months 2-4."
        )
        result = validate_agent_output(output)
        assert result == output

    @pytest.mark.unit
    def test_cpf_in_output_gets_masked(self) -> None:
        """CPF in agent output must be caught and masked."""
        output = "Member with CPF 123.456.789-00 is at high risk."
        result = validate_agent_output(output)
        assert "123.456.789-00" not in result
        assert "[REDACTED]" in result

    @pytest.mark.unit
    def test_email_in_output_gets_masked(self) -> None:
        """Email in agent output must be caught and masked."""
        output = "Contact joao@example.com for details."
        result = validate_agent_output(output)
        assert "joao@example.com" not in result

    @pytest.mark.unit
    def test_phone_in_output_gets_masked(self) -> None:
        """Phone number in agent output must be caught and masked."""
        output = "Call the member at (11) 99876-5432 to follow up."
        result = validate_agent_output(output)
        assert "99876-5432" not in result

    @pytest.mark.unit
    def test_mixed_pii_all_masked(self) -> None:
        """Output with multiple PII types should have all masked."""
        output = (
            "Member CPF 111.222.333-44 email me@co.com "
            "phone (21) 91111-2222"
        )
        result = validate_agent_output(output)
        assert "111.222.333-44" not in result
        assert "me@co.com" not in result
        assert "91111-2222" not in result

    @pytest.mark.unit
    def test_output_custom_mask_token(self) -> None:
        """Custom mask token should work for output validation."""
        output = "CPF: 123.456.789-00"
        result = validate_agent_output(output, mask_token="<HIDDEN>")
        assert "<HIDDEN>" in result


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------


class TestCheckConfidence:
    """Test confidence threshold logic."""

    @pytest.mark.unit
    def test_above_threshold_passes(self) -> None:
        """Confidence above threshold should pass."""
        is_confident, fallback = check_confidence(
            0.8, 0.6, "I don't know"
        )
        assert is_confident is True
        assert fallback is None

    @pytest.mark.unit
    def test_below_threshold_returns_fallback(self) -> None:
        """Confidence below threshold should return fallback."""
        is_confident, fallback = check_confidence(
            0.3, 0.6, "I don't know"
        )
        assert is_confident is False
        assert fallback == "I don't know"

    @pytest.mark.unit
    def test_exactly_at_threshold_passes(self) -> None:
        """At exactly the threshold, confidence is sufficient (not <)."""
        is_confident, fallback = check_confidence(0.6, 0.6, "fallback")
        # 0.6 < 0.6 is False, so this passes
        assert is_confident is True

    @pytest.mark.unit
    def test_zero_confidence_fails(self) -> None:
        """Zero confidence should always fail."""
        is_confident, _ = check_confidence(0.0, 0.6, "fallback")
        assert is_confident is False

    @pytest.mark.unit
    def test_perfect_confidence_passes(self) -> None:
        """1.0 confidence should always pass."""
        is_confident, fallback = check_confidence(1.0, 0.6, "fallback")
        assert is_confident is True
        assert fallback is None

    @pytest.mark.unit
    def test_fallback_message_preserved(self) -> None:
        """The exact fallback message should be returned when failing."""
        custom_msg = "Please consult the analytics team."
        _, fallback = check_confidence(0.1, 0.5, custom_msg)
        assert fallback == custom_msg
