"""Optimization layer for RetentionIQ.

Provides deterministic and stochastic budget allocation for
retention actions across locations and members.
"""

from src.optimization.allocator import (
    AllocationResult,
    build_deterministic_model,
    load_optimization_config,
    sample_cate_scenarios,
    solve_allocation,
)
from src.optimization.stochastic import (
    build_stochastic_model,
    compare_deterministic_vs_stochastic,
    solve_stochastic,
)

__all__ = [
    "AllocationResult",
    "build_deterministic_model",
    "build_stochastic_model",
    "compare_deterministic_vs_stochastic",
    "load_optimization_config",
    "sample_cate_scenarios",
    "solve_allocation",
    "solve_stochastic",
]
