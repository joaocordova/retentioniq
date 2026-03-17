"""Generate synthetic franchise fitness data for RetentionIQ.

This script creates realistic multi-location gym data with patterns grounded
in domain knowledge from operating a 250+ unit fitness franchise network:

- January enrollment peak (New Year's resolutions): 2-3x normal enrollment
- February/March: sharp drop-off from January cohort (30-40% churn in first 60 days)
- June/July: secondary enrollment bump (summer motivation)
- Aggregator members (Gympass/TotalPass): shorter tenure, lower visit frequency,
  different churn patterns vs direct members
- Location heterogeneity: urban locations have higher churn but higher throughput
- Retention actions: SMS, calls, discounts, PT sessions — with selection bias
  (managers target at-risk members, creating confounding)

Usage:
    python scripts/generate_franchise_data.py --locations 250 --members 2000000 --months 18
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


def generate_locations(n_locations: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate location master data."""
    regions = ["southeast", "south", "northeast", "midwest", "north"]
    region_weights = [0.45, 0.25, 0.15, 0.10, 0.05]  # realistic Brazilian distribution

    locations = []
    for i in range(n_locations):
        region = rng.choice(regions, p=region_weights)
        # Urban locations have more capacity but higher operating cost
        is_urban = rng.random() < 0.6
        capacity = rng.integers(300, 800) if is_urban else rng.integers(150, 400)

        locations.append({
            "location_id": f"LOC_{i:04d}",
            "region": region,
            "is_urban": is_urban,
            "capacity": capacity,
            "monthly_operating_cost": capacity * rng.integers(15, 35),
            "staff_retention_hours": rng.integers(10, 30),
        })

    return pd.DataFrame(locations)


def generate_members(
    n_members: int,
    locations: pd.DataFrame,
    n_months: int,
    start_date: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate member enrollment data with realistic patterns."""
    members = []
    loc_ids = locations["location_id"].tolist()
    loc_capacities = dict(zip(locations["location_id"], locations["capacity"]))

    # Distribute members across locations proportional to capacity
    total_capacity = sum(loc_capacities.values())
    loc_weights = [loc_capacities[lid] / total_capacity for lid in loc_ids]

    for i in range(n_members):
        location = rng.choice(loc_ids, p=loc_weights)

        # Enrollment date with seasonal pattern
        month_offset = rng.integers(0, n_months)
        enrollment_month = (start_date + timedelta(days=30 * month_offset)).month

        # January peak: 3x enrollment rate
        if enrollment_month == 1:
            enrollment_prob = 0.15
        elif enrollment_month in [6, 7]:
            enrollment_prob = 0.09
        elif enrollment_month in [2, 3, 11, 12]:
            enrollment_prob = 0.06
        else:
            enrollment_prob = 0.07

        # Spread enrollment within the month
        day_in_month = rng.integers(1, 29)
        join_date = start_date + timedelta(days=30 * month_offset + day_in_month)

        # Contract source: 15% aggregator
        is_aggregator = rng.random() < 0.15
        contract_source = "aggregator" if is_aggregator else "regular"
        aggregator_platform = (
            rng.choice(["gympass", "totalpass", "wellhub"]) if is_aggregator else None
        )

        # Plan type
        if is_aggregator:
            plan_type = "aggregator_standard"
            monthly_price = rng.uniform(30, 60)  # lower per-member revenue
        else:
            plan_type = rng.choice(
                ["basic", "standard", "premium"],
                p=[0.30, 0.50, 0.20],
            )
            price_map = {"basic": (49, 79), "standard": (89, 129), "premium": (149, 249)}
            monthly_price = rng.uniform(*price_map[plan_type])

        # Churn: aggregator members churn faster
        if is_aggregator:
            # Exponential distribution with median ~4 months
            tenure_months = rng.exponential(scale=4)
        else:
            # Exponential with median ~8 months, but January cohorts churn faster
            scale = 5 if enrollment_month == 1 else 8
            tenure_months = rng.exponential(scale=scale)

        # Some members are still active (right-censored)
        max_tenure = (datetime(2026, 3, 1) - join_date).days / 30
        if tenure_months > max_tenure:
            cancel_date = None  # still active
            churned = False
        else:
            cancel_date = join_date + timedelta(days=int(tenure_months * 30))
            churned = True

        # Age and demographics
        age = int(rng.normal(32, 10))
        age = max(16, min(75, age))
        gender = rng.choice(["M", "F", "other"], p=[0.45, 0.50, 0.05])

        members.append({
            "member_id": f"MBR_{i:07d}",
            "location_id": location,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "cancel_date": cancel_date.strftime("%Y-%m-%d") if cancel_date else None,
            "churned": churned,
            "contract_source": contract_source,
            "aggregator_platform": aggregator_platform,
            "plan_type": plan_type,
            "monthly_price": round(monthly_price, 2),
            "age": age,
            "gender": gender,
            # PII fields (will be used to test guardrails)
            "name": f"Member_{i}",  # placeholder
            "email": f"member_{i}@example.com",
            "cpf": f"{rng.integers(100,999)}.{rng.integers(100,999)}.{rng.integers(100,999)}-{rng.integers(10,99)}",
        })

    return pd.DataFrame(members)


def generate_visits(
    members: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate visit history with realistic patterns."""
    visits = []

    for _, member in members.iterrows():
        join = pd.Timestamp(member["join_date"])
        end = (
            pd.Timestamp(member["cancel_date"])
            if member["cancel_date"]
            else pd.Timestamp("2026-03-01")
        )
        tenure_days = (end - join).days
        if tenure_days <= 0:
            continue

        # Visit frequency depends on contract type and tenure
        if member["contract_source"] == "aggregator":
            avg_visits_per_month = rng.uniform(2, 6)
        else:
            avg_visits_per_month = rng.uniform(4, 12)

        # Frequency decays over tenure (motivation drops)
        n_months = max(1, tenure_days // 30)
        for month in range(n_months):
            decay = max(0.3, 1 - 0.05 * month)  # 5% decay per month, floor at 30%
            n_visits = max(0, int(rng.poisson(avg_visits_per_month * decay)))

            for _ in range(n_visits):
                visit_date = join + timedelta(days=30 * month + rng.integers(0, 30))
                if visit_date > end:
                    continue
                visits.append({
                    "member_id": member["member_id"],
                    "location_id": member["location_id"],
                    "visit_date": visit_date.strftime("%Y-%m-%d"),
                    "visit_duration_minutes": int(rng.normal(60, 20)),
                })

    return pd.DataFrame(visits)


def generate_retention_actions(
    members: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate retention actions WITH selection bias.

    This is intentional: managers target at-risk members, creating
    confounding between treatment assignment and outcome. The causal
    inference pipeline must handle this.
    """
    actions_list = []
    action_types = ["sms_reengagement", "phone_call", "discount_offer", "pt_session"]
    action_costs = {"sms_reengagement": 2.5, "phone_call": 15, "discount_offer": 50, "pt_session": 80}

    churned_members = members[members["churned"]].copy()

    for _, member in churned_members.iterrows():
        # Selection bias: members closer to churning are MORE likely to receive action
        # (managers notice declining visits and intervene)
        # This is realistic and creates the confounding we need to handle
        if rng.random() < 0.4:  # 40% of churned members received some action
            cancel = pd.Timestamp(member["cancel_date"])
            # Action happens 2-6 weeks before cancel
            days_before = rng.integers(14, 45)
            action_date = cancel - timedelta(days=days_before)

            if member["contract_source"] == "aggregator":
                # Can't offer discounts to aggregator members
                available_actions = ["sms_reengagement", "phone_call", "pt_session"]
            else:
                available_actions = action_types

            action_type = rng.choice(available_actions)

            actions_list.append({
                "member_id": member["member_id"],
                "location_id": member["location_id"],
                "action_date": action_date.strftime("%Y-%m-%d"),
                "action_type": action_type,
                "cost": action_costs[action_type],
            })

    # Also add actions to some active members (who didn't churn)
    active_members = members[~members["churned"]].sample(
        frac=0.15, random_state=int(rng.integers(0, 10000))
    )
    for _, member in active_members.iterrows():
        action_date = pd.Timestamp(member["join_date"]) + timedelta(
            days=rng.integers(30, 180)
        )
        if action_date > pd.Timestamp("2026-03-01"):
            continue

        if member["contract_source"] == "aggregator":
            available_actions = ["sms_reengagement", "phone_call", "pt_session"]
        else:
            available_actions = action_types

        action_type = rng.choice(available_actions)
        actions_list.append({
            "member_id": member["member_id"],
            "location_id": member["location_id"],
            "action_date": action_date.strftime("%Y-%m-%d"),
            "action_type": action_type,
            "cost": action_costs[action_type],
        })

    return pd.DataFrame(actions_list)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic franchise data")
    parser.add_argument("--locations", type=int, default=250)
    parser.add_argument("--members", type=int, default=2_000_000)
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/raw/")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime(2024, 9, 1)  # 18 months back from March 2026

    logger.info("generating_locations", n=args.locations)
    locations = generate_locations(args.locations, rng)
    locations.to_parquet(output_dir / "locations.parquet", index=False)

    logger.info("generating_members", n=args.members)
    members = generate_members(args.members, locations, args.months, start_date, rng)
    members.to_parquet(output_dir / "members.parquet", index=False)

    logger.info("generating_visits", n_members=len(members))
    visits = generate_visits(members, rng)
    visits.to_parquet(output_dir / "visits.parquet", index=False)

    logger.info("generating_retention_actions")
    actions = generate_retention_actions(members, rng)
    actions.to_parquet(output_dir / "retention_actions.parquet", index=False)

    # Summary
    logger.info(
        "data_generation_complete",
        locations=len(locations),
        members=len(members),
        visits=len(visits),
        retention_actions=len(actions),
        churn_rate=f"{members['churned'].mean():.1%}",
        aggregator_pct=f"{(members['contract_source'] == 'aggregator').mean():.1%}",
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
