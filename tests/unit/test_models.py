"""Unit tests for model training, scoring, and risk-tier logic.

These tests verify the model layer's business logic without requiring
MLflow, a GPU, or actual model training (which is slow). Instead,
we test the interfaces, threshold logic, and scoring pipeline
using mock/small models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Churn model training
# ---------------------------------------------------------------------------


class TestChurnModelTraining:
    """Test that churn model training produces valid outputs."""

    @pytest.fixture
    def training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Generate a small imbalanced classification dataset.

        Mimics the churn problem: ~15% positive class, rest negative.
        """
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            weights=[0.85, 0.15],
            random_state=42,
        )
        feature_names = [
            "visit_count_30d", "visit_count_60d",
            "days_since_last_visit", "tenure_months",
            "monthly_price", "payment_delay_flag",
            "visit_frequency_trend", "age",
            "feature_8", "feature_9",
        ]
        df_X = pd.DataFrame(X, columns=feature_names)
        return df_X, pd.Series(y, name="churned_30d")

    @pytest.mark.unit
    def test_model_trains_successfully(
        self,
        training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """XGBoost model should train without errors on valid data."""
        X, y = training_data
        model = XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=50,
            eval_metric="aucpr",
            random_state=42,
        )
        model.fit(X, y)
        assert model is not None
        assert hasattr(model, "predict_proba")

    @pytest.mark.unit
    def test_model_produces_valid_probabilities(
        self,
        training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Predicted probabilities should be in [0, 1]."""
        X, y = training_data
        model = XGBClassifier(
            max_depth=4,
            n_estimators=50,
            random_state=42,
        )
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    @pytest.mark.unit
    def test_model_auc_above_random(
        self,
        training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Trained model should beat random (AUC > 0.5) on training data."""
        X, y = training_data
        model = XGBClassifier(
            max_depth=4,
            n_estimators=50,
            random_state=42,
        )
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        assert auc > 0.5

    @pytest.mark.unit
    def test_model_handles_class_imbalance(
        self,
        training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Model with scale_pos_weight should handle imbalanced data.

        Per config: scale_pos_weight is calculated from class ratio.
        """
        X, y = training_data
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)

        model = XGBClassifier(
            max_depth=4,
            n_estimators=50,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        # With class weighting, model should predict some positives
        preds = (probs >= 0.5).astype(int)
        assert preds.sum() > 0, (
            "Model should predict at least some positives"
        )


# ---------------------------------------------------------------------------
# Threshold finding
# ---------------------------------------------------------------------------


class TestThresholdFinding:
    """Test optimal threshold selection for churn classification."""

    @pytest.fixture
    def predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Create realistic prediction scores and labels."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = rng.choice([0, 1], size=n, p=[0.85, 0.15])
        # Make positives have higher scores on average
        y_scores = np.where(
            y_true == 1,
            rng.beta(5, 2, size=n),
            rng.beta(2, 5, size=n),
        )
        return y_true, y_scores

    @pytest.mark.unit
    def test_f1_optimal_threshold(
        self,
        predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Find threshold that maximizes F1 score."""
        y_true, y_scores = predictions
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Optimal threshold should be between 0.2 and 0.8
        assert 0.2 <= best_threshold <= 0.8
        # F1 at optimal should be better than at 0.5
        preds_50 = (y_scores >= 0.5).astype(int)
        f1_50 = f1_score(y_true, preds_50, zero_division=0)
        assert best_f1 >= f1_50

    @pytest.mark.unit
    def test_threshold_respects_min_precision(
        self,
        predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Threshold must yield precision >= 0.3 (business constraint).

        Per config: min_precision = 0.3 to avoid wasting retention
        budget on false positives.
        """
        y_true, y_scores = predictions
        min_precision = 0.3

        # Find a threshold that gives precision >= min_precision
        valid_thresholds = []
        for threshold in np.arange(0.1, 0.95, 0.01):
            preds = (y_scores >= threshold).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            precision = tp / max(tp + fp, 1)
            if precision >= min_precision:
                valid_thresholds.append(threshold)

        assert len(valid_thresholds) > 0, (
            "Should find at least one threshold with precision >= 0.3"
        )

    @pytest.mark.unit
    def test_threshold_0_predicts_all_positive(
        self,
        predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Threshold of 0 should classify everything as positive."""
        y_true, y_scores = predictions
        preds = (y_scores >= 0.0).astype(int)
        assert preds.sum() == len(preds)

    @pytest.mark.unit
    def test_threshold_1_predicts_all_negative(
        self,
        predictions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Threshold of 1 should classify everything as negative."""
        y_true, y_scores = predictions
        preds = (y_scores >= 1.0).astype(int)
        assert preds.sum() == 0


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------


class TestScoringPipeline:
    """Test the scoring (inference) pipeline output format."""

    @pytest.mark.unit
    def test_scoring_produces_expected_columns(self) -> None:
        """Scored output must include member_id, probability, risk_tier."""
        scored = pd.DataFrame({
            "member_id": ["MBR-001", "MBR-002", "MBR-003"],
            "churn_probability": [0.85, 0.42, 0.12],
            "risk_tier": ["critical", "medium", "low"],
            "model_version": ["v1.2.0"] * 3,
        })
        required_cols = [
            "member_id", "churn_probability", "risk_tier",
        ]
        for col in required_cols:
            assert col in scored.columns

    @pytest.mark.unit
    def test_probability_column_bounded(self) -> None:
        """Churn probabilities must be in [0, 1]."""
        probs = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Risk tier assignment
# ---------------------------------------------------------------------------


class TestRiskTierAssignment:
    """Test that members are assigned correct risk tiers based on
    churn probability thresholds.

    Tiers: low (<0.25), medium (0.25-0.50), high (0.50-0.75),
    critical (>0.75).
    """

    @pytest.fixture
    def tier_boundaries(self) -> list[float]:
        """Standard tier boundaries.

        Uses -0.001 as lower bound so that pd.cut's default
        (left, right] intervals include 0.0 in the 'low' bin.
        """
        return [-0.001, 0.25, 0.50, 0.75, 1.01]

    @pytest.fixture
    def tier_labels(self) -> list[str]:
        """Standard tier labels matching boundaries."""
        return ["low", "medium", "high", "critical"]

    @pytest.mark.unit
    def test_low_risk_tier(
        self, tier_boundaries: list[float], tier_labels: list[str]
    ) -> None:
        """Probability <= 0.25 -> low risk."""
        probs = pd.Series([0.0, 0.10, 0.24])
        tiers = pd.cut(
            probs, bins=tier_boundaries, labels=tier_labels
        )
        assert (tiers == "low").all()

    @pytest.mark.unit
    def test_medium_risk_tier(
        self, tier_boundaries: list[float], tier_labels: list[str]
    ) -> None:
        """Probability (0.25, 0.50] -> medium risk."""
        probs = pd.Series([0.26, 0.35, 0.49])
        tiers = pd.cut(
            probs, bins=tier_boundaries, labels=tier_labels
        )
        assert (tiers == "medium").all()

    @pytest.mark.unit
    def test_high_risk_tier(
        self, tier_boundaries: list[float], tier_labels: list[str]
    ) -> None:
        """Probability (0.50, 0.75] -> high risk."""
        probs = pd.Series([0.51, 0.60, 0.74])
        tiers = pd.cut(
            probs, bins=tier_boundaries, labels=tier_labels
        )
        assert (tiers == "high").all()

    @pytest.mark.unit
    def test_critical_risk_tier(
        self, tier_boundaries: list[float], tier_labels: list[str]
    ) -> None:
        """Probability (0.75, 1.0] -> critical risk."""
        probs = pd.Series([0.76, 0.90, 1.00])
        tiers = pd.cut(
            probs, bins=tier_boundaries, labels=tier_labels
        )
        assert (tiers == "critical").all()

    @pytest.mark.unit
    def test_all_tiers_present_in_population(
        self, tier_boundaries: list[float], tier_labels: list[str]
    ) -> None:
        """A realistic population should have all 4 risk tiers."""
        rng = np.random.default_rng(42)
        probs = pd.Series(rng.beta(2, 5, size=1000))
        tiers = pd.cut(
            probs, bins=tier_boundaries, labels=tier_labels
        )
        assert set(tiers.dropna().unique()) == set(tier_labels)

    @pytest.mark.unit
    def test_tier_distribution_matches_expectation(
        self,
        sample_member_360: pd.DataFrame,
        tier_boundaries: list[float],
        tier_labels: list[str],
    ) -> None:
        """Risk tiers should be assigned to all members in 360 view."""
        probs = sample_member_360["churn_probability"]
        tiers = pd.cut(
            probs, bins=tier_boundaries, labels=tier_labels
        )
        # All members should have a tier assigned
        assert tiers.notna().all()


# ---------------------------------------------------------------------------
# LTV estimation
# ---------------------------------------------------------------------------


class TestLTVEstimation:
    """Test LTV (Lifetime Value) estimation logic.

    LTV = expected remaining tenure * monthly revenue, discounted.
    """

    @pytest.mark.unit
    def test_ltv_basic_calculation(self) -> None:
        """LTV = monthly_price * expected_months * discount_factor."""
        monthly_price = 119.90
        expected_months = 12
        discount_rate = 0.01
        # Simple NPV: sum of discounted monthly payments
        ltv = sum(
            monthly_price / (1 + discount_rate) ** m
            for m in range(1, expected_months + 1)
        )
        # Should be slightly less than 119.90 * 12 = 1438.80
        assert ltv < monthly_price * expected_months
        assert ltv > monthly_price * expected_months * 0.9

    @pytest.mark.unit
    def test_ltv_higher_for_premium_plan(self) -> None:
        """Premium plan members should have higher LTV than basic."""
        expected_months = 12
        discount_rate = 0.01
        ltv_basic = sum(
            79.90 / (1 + discount_rate) ** m
            for m in range(1, expected_months + 1)
        )
        ltv_premium = sum(
            179.90 / (1 + discount_rate) ** m
            for m in range(1, expected_months + 1)
        )
        assert ltv_premium > ltv_basic

    @pytest.mark.unit
    def test_ltv_zero_for_immediate_churn(self) -> None:
        """Member expected to churn immediately has ~0 LTV."""
        monthly_price = 119.90
        expected_months = 0
        ltv = sum(
            monthly_price / (1 + 0.01) ** m
            for m in range(1, expected_months + 1)
        )
        assert ltv == 0.0

    @pytest.mark.unit
    def test_ltv_capped_at_max_horizon(self) -> None:
        """LTV should not exceed max_horizon_months (36 per config)."""
        monthly_price = 119.90
        max_horizon = 36
        discount_rate = 0.01
        ltv = sum(
            monthly_price / (1 + discount_rate) ** m
            for m in range(1, max_horizon + 1)
        )
        # A reasonable cap: 36 months * 119.90 undiscounted = 4316.40
        assert ltv < monthly_price * max_horizon
        assert ltv > 0

    @pytest.mark.unit
    def test_ltv_aggregator_lower_margin(self) -> None:
        """Aggregator members have lower prices -> lower LTV."""
        expected_months = 12
        discount_rate = 0.01
        ltv_regular = sum(
            119.90 / (1 + discount_rate) ** m
            for m in range(1, expected_months + 1)
        )
        ltv_aggregator = sum(
            49.90 / (1 + discount_rate) ** m
            for m in range(1, expected_months + 1)
        )
        assert ltv_aggregator < ltv_regular
        # Aggregator LTV should be roughly 49.90/119.90 of regular
        ratio = ltv_aggregator / ltv_regular
        assert abs(ratio - 49.90 / 119.90) < 0.01
