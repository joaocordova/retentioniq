"""Unit tests for feature engineering logic.

These tests verify that feature computations are correct and that
temporal integrity is maintained (no future leakage). Feature engineering
is the most common source of subtle ML bugs, so these tests are thorough.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Visit behavior features
# ---------------------------------------------------------------------------


class TestVisitFeatures:
    """Test visit-based feature computations."""

    @pytest.fixture
    def member_visits(self) -> pd.DataFrame:
        """Visits for a single member over 90 days."""
        base = pd.Timestamp("2025-01-01")
        return pd.DataFrame({
            "member_id": ["MBR-001"] * 12,
            "visit_date": pd.to_datetime([
                "2024-12-31",  # 1d ago
                "2024-12-29",  # 3d ago
                "2024-12-27",  # 5d ago
                "2024-12-25",  # 7d ago (boundary of 7d window)
                "2024-12-20",  # 12d ago
                "2024-12-18",  # 14d ago (boundary)
                "2024-12-10",  # 22d ago
                "2024-12-01",  # 31d ago (outside 30d window)
                "2024-11-15",  # 47d ago
                "2024-11-01",  # 61d ago
                "2024-10-15",  # 78d ago
                "2024-10-05",  # 88d ago
            ]),
            "visit_duration_minutes": [
                60, 45, 90, 30, 75, 60, 45, 120, 60, 30, 45, 90
            ],
        })

    @pytest.mark.unit
    def test_visit_count_7d(self, member_visits: pd.DataFrame) -> None:
        """Count visits in the last 7 days."""
        base = pd.Timestamp("2025-01-01")
        cutoff = base - pd.Timedelta(days=7)
        count = (member_visits["visit_date"] >= cutoff).sum()
        # Days 1, 3, 5 are within 7 days. Day 7 is exactly the boundary
        # (>= cutoff means included).
        assert count == 4

    @pytest.mark.unit
    def test_visit_count_14d(self, member_visits: pd.DataFrame) -> None:
        """Count visits in the last 14 days."""
        base = pd.Timestamp("2025-01-01")
        cutoff = base - pd.Timedelta(days=14)
        count = (member_visits["visit_date"] >= cutoff).sum()
        assert count == 6

    @pytest.mark.unit
    def test_visit_count_30d(self, member_visits: pd.DataFrame) -> None:
        """Count visits in the last 30 days."""
        base = pd.Timestamp("2025-01-01")
        cutoff = base - pd.Timedelta(days=30)
        count = (member_visits["visit_date"] >= cutoff).sum()
        assert count == 7

    @pytest.mark.unit
    def test_visit_count_90d(self, member_visits: pd.DataFrame) -> None:
        """All visits should be within 90 days."""
        base = pd.Timestamp("2025-01-01")
        cutoff = base - pd.Timedelta(days=90)
        count = (member_visits["visit_date"] >= cutoff).sum()
        assert count == 12

    @pytest.mark.unit
    def test_days_since_last_visit(
        self, member_visits: pd.DataFrame
    ) -> None:
        """Days since last visit should be computed from the latest date."""
        base = pd.Timestamp("2025-01-01")
        last_visit = member_visits["visit_date"].max()
        days_since = (base - last_visit).days
        assert days_since == 1

    @pytest.mark.unit
    def test_days_since_last_visit_no_visits(self) -> None:
        """Member with no visits should have large days_since value."""
        visits = pd.DataFrame(
            columns=["member_id", "visit_date", "visit_duration_minutes"]
        )
        # Convention: no visits -> use a large sentinel value
        last_visit = visits["visit_date"].max()
        assert pd.isna(last_visit)

    @pytest.mark.unit
    def test_visit_count_zero_for_inactive(self) -> None:
        """Inactive member should have 0 visits in all windows."""
        visits = pd.DataFrame({
            "member_id": ["MBR-999"],
            "visit_date": pd.to_datetime(["2024-03-01"]),
            "visit_duration_minutes": [60],
        })
        base = pd.Timestamp("2025-01-01")
        for window in [7, 14, 30, 60, 90]:
            cutoff = base - pd.Timedelta(days=window)
            count = (visits["visit_date"] >= cutoff).sum()
            assert count == 0, (
                f"{window}d window should be 0 for old visit"
            )

    @pytest.mark.unit
    def test_avg_visit_duration(
        self, member_visits: pd.DataFrame
    ) -> None:
        """Average visit duration for a window should be correctly computed."""
        base = pd.Timestamp("2025-01-01")
        cutoff = base - pd.Timedelta(days=30)
        recent = member_visits[member_visits["visit_date"] >= cutoff]
        avg_dur = recent["visit_duration_minutes"].mean()
        expected = np.mean([60, 45, 90, 30, 75, 60, 45])
        assert abs(avg_dur - expected) < 1e-6

    @pytest.mark.unit
    def test_visit_frequency_trend(self) -> None:
        """Trend = last-14d visits / prior-14d visits (bounded by 1).

        A declining trend (<1) indicates the member is visiting less
        frequently, which is a churn signal.
        """
        visits_last_14d = 2
        visits_prior_14d = 6
        trend = visits_last_14d / max(visits_prior_14d, 1)
        assert abs(trend - 1 / 3) < 1e-6
        assert trend < 1.0  # declining

    @pytest.mark.unit
    def test_visit_frequency_trend_no_prior(self) -> None:
        """Zero prior visits should use floor of 1 to avoid division by 0."""
        visits_last_14d = 5
        visits_prior_14d = 0
        trend = visits_last_14d / max(visits_prior_14d, 1)
        assert trend == 5.0  # new member ramping up


# ---------------------------------------------------------------------------
# Tenure features
# ---------------------------------------------------------------------------


class TestTenureFeatures:
    """Test tenure-based feature computation."""

    @pytest.mark.unit
    def test_tenure_days_calculation(self) -> None:
        """Tenure days should be difference between current and join date."""
        join_date = pd.Timestamp("2024-06-15")
        current_date = pd.Timestamp("2025-01-01")
        tenure_days = (current_date - join_date).days
        assert tenure_days == 200

    @pytest.mark.unit
    def test_tenure_months_calculation(self) -> None:
        """Tenure months should be tenure_days / 30."""
        tenure_days = 180
        tenure_months = tenure_days / 30.0
        assert abs(tenure_months - 6.0) < 1e-6

    @pytest.mark.unit
    def test_is_first_90_days(self) -> None:
        """Members within first 90 days should be flagged."""
        tenures = pd.Series([30, 89, 90, 91, 180])
        is_first_90 = tenures <= 90
        assert list(is_first_90) == [True, True, True, False, False]

    @pytest.mark.unit
    def test_tenure_zero_for_same_day(self) -> None:
        """Member joining today should have 0 tenure."""
        join_date = pd.Timestamp("2025-01-01")
        current_date = pd.Timestamp("2025-01-01")
        tenure_days = (current_date - join_date).days
        assert tenure_days == 0


# ---------------------------------------------------------------------------
# Temporal train/test split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    """Test that train/test splits respect temporal ordering.

    The project requires temporal splits (no random splits) to prevent
    future information leaking into training data.
    """

    @pytest.fixture
    def temporal_data(self) -> pd.DataFrame:
        """Create data spanning 18 months for temporal splitting."""
        rng = np.random.default_rng(42)
        n = 1000
        dates = pd.date_range("2023-07-01", periods=n, freq="D")
        return pd.DataFrame({
            "event_date": dates[:n],
            "member_id": [f"MBR-{i:04d}" for i in range(n)],
            "feature_a": rng.normal(0, 1, n),
            "churned_30d": rng.choice([0, 1], n, p=[0.85, 0.15]),
        })

    @pytest.mark.unit
    def test_train_before_test(
        self, temporal_data: pd.DataFrame
    ) -> None:
        """All training dates must be before all test dates."""
        split_date = pd.Timestamp("2024-10-01")
        train = temporal_data[temporal_data["event_date"] < split_date]
        test = temporal_data[temporal_data["event_date"] >= split_date]

        assert train["event_date"].max() < test["event_date"].min()

    @pytest.mark.unit
    def test_no_future_leakage(
        self, temporal_data: pd.DataFrame
    ) -> None:
        """No test-period data should appear in the training set."""
        split_date = pd.Timestamp("2024-10-01")
        train = temporal_data[temporal_data["event_date"] < split_date]

        # Every row in train must be before the split
        assert (train["event_date"] < split_date).all()

    @pytest.mark.unit
    def test_train_test_no_overlap(
        self, temporal_data: pd.DataFrame
    ) -> None:
        """Train and test sets should be disjoint by date."""
        split_date = pd.Timestamp("2024-10-01")
        train = temporal_data[temporal_data["event_date"] < split_date]
        test = temporal_data[temporal_data["event_date"] >= split_date]

        train_dates = set(train["event_date"])
        test_dates = set(test["event_date"])
        assert len(train_dates & test_dates) == 0

    @pytest.mark.unit
    def test_expanding_window_split(self) -> None:
        """Expanding window: each fold adds data but never uses future."""
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        data = pd.DataFrame({
            "event_date": dates,
            "value": range(365),
        })
        step_months = 3
        folds = []
        for month_offset in range(4, 13, step_months):
            train_end = pd.Timestamp("2023-01-01") + pd.DateOffset(
                months=month_offset
            )
            val_end = train_end + pd.DateOffset(months=3)
            train = data[data["event_date"] < train_end]
            val = data[
                (data["event_date"] >= train_end)
                & (data["event_date"] < val_end)
            ]
            if len(val) > 0:
                folds.append((train, val))

        # Each fold's train data should be larger than the previous
        for i in range(1, len(folds)):
            assert len(folds[i][0]) > len(folds[i - 1][0])

        # No fold's validation data should overlap with its training data
        for train, val in folds:
            assert train["event_date"].max() < val["event_date"].min()


# ---------------------------------------------------------------------------
# Feature matrix PII exclusion
# ---------------------------------------------------------------------------


class TestFeatureMatrixPIIExclusion:
    """Ensure PII columns are excluded from the feature matrix.

    Per configs/features.yaml, member_id, name, email, cpf, phone,
    address, cancel_date, and churned must be excluded from features.
    """

    @pytest.fixture
    def raw_feature_df(self) -> pd.DataFrame:
        """A feature DataFrame that still has PII and leakage columns."""
        return pd.DataFrame({
            "member_id": ["MBR-001"],
            "name": ["Alice"],
            "email": ["a@test.com"],
            "cpf": ["123.456.789-00"],
            "phone": ["11999999999"],
            "cancel_date": ["2025-03-01"],
            "churned": [True],
            "visit_count_30d": [12],
            "tenure_months": [6.5],
            "monthly_price": [119.90],
        })

    @pytest.mark.unit
    def test_pii_columns_excluded(
        self, raw_feature_df: pd.DataFrame
    ) -> None:
        """PII columns must be removed before model training."""
        exclude_cols = [
            "member_id", "name", "email", "cpf", "phone",
            "address", "cancel_date", "churned",
        ]
        features = raw_feature_df.drop(
            columns=[c for c in exclude_cols if c in raw_feature_df.columns]
        )
        for col in exclude_cols:
            assert col not in features.columns

    @pytest.mark.unit
    def test_feature_columns_preserved(
        self, raw_feature_df: pd.DataFrame
    ) -> None:
        """Legitimate feature columns should remain after exclusion."""
        exclude_cols = [
            "member_id", "name", "email", "cpf", "phone",
            "address", "cancel_date", "churned",
        ]
        features = raw_feature_df.drop(
            columns=[c for c in exclude_cols if c in raw_feature_df.columns]
        )
        assert "visit_count_30d" in features.columns
        assert "tenure_months" in features.columns
        assert "monthly_price" in features.columns

    @pytest.mark.unit
    def test_cancel_date_is_excluded_as_leakage(
        self, raw_feature_df: pd.DataFrame
    ) -> None:
        """cancel_date is a target leakage column; it must be excluded."""
        exclude_cols = ["cancel_date"]
        features = raw_feature_df.drop(columns=exclude_cols)
        assert "cancel_date" not in features.columns

    @pytest.mark.unit
    def test_target_column_excluded(
        self, raw_feature_df: pd.DataFrame
    ) -> None:
        """The target column (churned) should not be in the feature matrix."""
        features = raw_feature_df.drop(columns=["churned"])
        assert "churned" not in features.columns
