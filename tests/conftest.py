"""Shared pytest fixtures for RetentionIQ test suite.

All fixtures generate deterministic synthetic data with a fixed seed
so tests are reproducible. The data mimics the real schema defined
in configs/data.yaml but at a small scale suitable for fast unit tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_members() -> pd.DataFrame:
    """Create a small DataFrame of 100 members for testing.

    Mix of regular/aggregator contract sources, some churned.
    Covers edge cases: new members (<30d tenure), long-tenured,
    various plan types, and missing cancel_date for active members.
    """
    rng = np.random.default_rng(42)
    n = 100

    member_ids = [f"MBR-{i:04d}" for i in range(n)]
    location_ids = [f"LOC-{rng.integers(1, 11):03d}" for _ in range(n)]
    contract_sources = rng.choice(
        ["regular", "aggregator"], size=n, p=[0.7, 0.3]
    )
    plan_types = []
    for src in contract_sources:
        if src == "aggregator":
            plan_types.append("aggregator_standard")
        else:
            plan_types.append(
                rng.choice(["basic", "standard", "premium"])
            )

    base_date = pd.Timestamp("2025-01-01")
    join_offsets = rng.integers(30, 540, size=n)
    join_dates = [
        (base_date - pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in join_offsets
    ]

    churned = rng.choice([True, False], size=n, p=[0.25, 0.75])
    cancel_dates = []
    for i in range(n):
        if churned[i]:
            cancel_offset = rng.integers(1, int(join_offsets[i]))
            cancel_date = (
                base_date - pd.Timedelta(days=int(join_offsets[i]))
                + pd.Timedelta(days=int(cancel_offset))
            )
            cancel_dates.append(cancel_date.strftime("%Y-%m-%d"))
        else:
            cancel_dates.append(None)

    prices = []
    for pt in plan_types:
        if pt == "basic":
            prices.append(79.90)
        elif pt == "standard":
            prices.append(119.90)
        elif pt == "premium":
            prices.append(179.90)
        else:
            prices.append(49.90)

    aggregator_platforms = []
    for src in contract_sources:
        if src == "aggregator":
            aggregator_platforms.append(
                rng.choice(["gympass", "totalpass", "wellhub"])
            )
        else:
            aggregator_platforms.append("none")

    return pd.DataFrame({
        "member_id": member_ids,
        "location_id": location_ids,
        "join_date": join_dates,
        "cancel_date": cancel_dates,
        "churned": churned,
        "contract_source": contract_sources,
        "aggregator_platform": aggregator_platforms,
        "plan_type": plan_types,
        "monthly_price": prices,
        "age": rng.integers(18, 65, size=n),
        "gender": rng.choice(["M", "F"], size=n),
        "name": [f"Member {i}" for i in range(n)],
        "email": [f"member{i}@test.com" for i in range(n)],
        "cpf": [
            f"{rng.integers(100,999)}.{rng.integers(100,999)}."
            f"{rng.integers(100,999)}-{rng.integers(10,99)}"
            for _ in range(n)
        ],
    })


@pytest.fixture
def sample_visits(sample_members: pd.DataFrame) -> pd.DataFrame:
    """Create visit history for sample members.

    Generates between 0 and 30 visits per member over the last 90 days.
    Active members visit more frequently; churned members taper off.
    Includes edge cases: very short visits (likely errors) and long ones.
    """
    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("2025-01-01")

    records = []
    for _, member in sample_members.iterrows():
        is_churned = member["churned"]
        n_visits = rng.integers(0, 8 if is_churned else 31)
        for _ in range(n_visits):
            visit_offset = rng.integers(0, 90)
            visit_date = (
                base_date - pd.Timedelta(days=int(visit_offset))
            ).strftime("%Y-%m-%d")
            # Most visits are 30-90 min, but include edge cases
            duration = int(rng.choice(
                [2, 3, 45, 60, 75, 90, 120, 350],
                p=[0.02, 0.03, 0.20, 0.30, 0.20, 0.15, 0.08, 0.02],
            ))
            records.append({
                "member_id": member["member_id"],
                "location_id": member["location_id"],
                "visit_date": visit_date,
                "visit_duration_minutes": duration,
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_retention_actions(
    sample_members: pd.DataFrame,
) -> pd.DataFrame:
    """Create retention action records for sample members.

    About 30% of members receive a retention action. Actions are
    distributed across types with realistic costs.
    """
    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("2025-01-01")

    action_types = [
        ("sms_reengagement", 2.50),
        ("phone_call", 15.00),
        ("discount_offer", 50.00),
        ("personal_trainer_session", 80.00),
    ]

    records = []
    for _, member in sample_members.iterrows():
        if rng.random() < 0.3:
            action_name, cost = action_types[rng.integers(0, len(action_types))]
            action_offset = rng.integers(0, 60)
            action_date = (
                base_date - pd.Timedelta(days=int(action_offset))
            ).strftime("%Y-%m-%d")
            records.append({
                "member_id": member["member_id"],
                "location_id": member["location_id"],
                "action_date": action_date,
                "action_type": action_name,
                "cost": cost,
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_member_360(sample_members: pd.DataFrame) -> pd.DataFrame:
    """Create a pre-built 360-degree member view (gold layer).

    Combines member attributes with computed features that would
    normally come from the feature engineering pipeline.
    """
    rng = np.random.default_rng(42)
    n = len(sample_members)

    base_date = pd.Timestamp("2025-01-01")

    df = sample_members[
        ["member_id", "location_id", "contract_source", "plan_type",
         "monthly_price", "age", "churned"]
    ].copy()

    join_dates = pd.to_datetime(sample_members["join_date"])
    df["tenure_days"] = (base_date - join_dates).dt.days
    df["tenure_months"] = df["tenure_days"] / 30.0

    df["visit_count_7d"] = rng.poisson(lam=2, size=n)
    df["visit_count_14d"] = df["visit_count_7d"] + rng.poisson(lam=2, size=n)
    df["visit_count_30d"] = (
        df["visit_count_14d"] + rng.poisson(lam=4, size=n)
    )
    df["visit_count_60d"] = (
        df["visit_count_30d"] + rng.poisson(lam=6, size=n)
    )
    df["visit_count_90d"] = (
        df["visit_count_60d"] + rng.poisson(lam=8, size=n)
    )

    df["days_since_last_visit"] = rng.integers(0, 45, size=n)
    df["avg_visit_duration_30d"] = rng.normal(60, 15, size=n).clip(15, 150)
    df["visit_frequency_trend"] = rng.uniform(0.3, 2.0, size=n)
    df["payment_delay_flag"] = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    # Churn probability — correlated with actual churn for realism
    noise = rng.normal(0, 0.1, size=n)
    df["churn_probability"] = (
        df["churned"].astype(float) * 0.6 + noise + 0.2
    ).clip(0, 1)

    df["risk_tier"] = pd.cut(
        df["churn_probability"],
        bins=[-0.001, 0.25, 0.5, 0.75, 1.01],
        labels=["low", "medium", "high", "critical"],
    )

    df["ltv"] = (
        df["monthly_price"] * df["tenure_months"] * rng.uniform(0.8, 1.2, size=n)
    )

    return df


@pytest.fixture
def sample_cate_estimates() -> pd.DataFrame:
    """Create mock CATE (conditional average treatment effect) estimates.

    Provides estimated treatment effects for 50 members across 4
    action types, with mean and standard deviation for each.
    """
    rng = np.random.default_rng(42)

    member_ids = [f"MBR-{i:04d}" for i in range(50)]
    actions = [
        "sms_reengagement",
        "phone_call",
        "discount_offer",
        "personal_trainer_session",
    ]

    records = []
    for mid in member_ids:
        for action in actions:
            cate_mean = rng.uniform(0.0, 0.15)
            cate_std = rng.uniform(0.01, 0.05)
            records.append({
                "member_id": mid,
                "action": action,
                "cate_mean": cate_mean,
                "cate_std": cate_std,
            })

    return pd.DataFrame(records)
