"""Custom exception hierarchy for RetentionIQ.

Structured exceptions make error handling explicit and debuggable.
Every exception carries enough context to diagnose the issue without
reading stack traces.
"""


class RetentionIQError(Exception):
    """Base exception for all RetentionIQ errors."""


# --- Data Layer ---


class DataError(RetentionIQError):
    """Base for data-related errors."""


class SchemaValidationError(DataError):
    """Raised when data fails Great Expectations validation."""

    def __init__(self, layer: str, suite: str, failures: list[str]) -> None:
        self.layer = layer
        self.suite = suite
        self.failures = failures
        msg = (
            f"Schema validation failed at {layer} layer "
            f"(suite: {suite}): {len(failures)} expectation(s) failed. "
            f"First failure: {failures[0] if failures else 'unknown'}"
        )
        super().__init__(msg)


class DataFreshnessError(DataError):
    """Raised when data is stale beyond acceptable SLA."""

    def __init__(self, table: str, last_updated: str, sla_hours: int) -> None:
        self.table = table
        self.last_updated = last_updated
        self.sla_hours = sla_hours
        super().__init__(
            f"Table '{table}' last updated at {last_updated}, "
            f"exceeding {sla_hours}h freshness SLA"
        )


# --- Model Layer ---


class ModelError(RetentionIQError):
    """Base for model-related errors."""


class ModelNotFoundError(ModelError):
    """Raised when a model is not found in MLflow registry."""

    def __init__(self, model_name: str, stage: str = "Production") -> None:
        self.model_name = model_name
        self.stage = stage
        super().__init__(
            f"Model '{model_name}' not found in MLflow registry at stage '{stage}'"
        )


class ModelPerformanceDegradationError(ModelError):
    """Raised when model performance drops below threshold."""

    def __init__(
        self, model_name: str, metric: str, current: float, threshold: float
    ) -> None:
        self.model_name = model_name
        self.metric = metric
        self.current_value = current
        self.threshold = threshold
        super().__init__(
            f"Model '{model_name}' {metric} dropped to {current:.4f}, "
            f"below threshold {threshold:.4f}"
        )


# --- Causal Layer ---


class CausalError(RetentionIQError):
    """Base for causal inference errors."""


class CausalConfigError(CausalError):
    """Raised when causal DAG configuration is invalid."""


class RefutationFailedError(CausalError):
    """Raised when causal estimate fails refutation tests."""

    def __init__(self, test_name: str, details: str) -> None:
        self.test_name = test_name
        self.details = details
        super().__init__(
            f"Causal estimate failed refutation test '{test_name}': {details}. "
            f"This means our causal assumptions may be wrong — "
            f"do NOT use this estimate for decision-making."
        )


# --- Optimization Layer ---


class OptimizationError(RetentionIQError):
    """Base for optimization errors."""


class InfeasibleProblemError(OptimizationError):
    """Raised when the optimization problem has no feasible solution."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"Optimization problem is infeasible: {reason}. "
            f"Check budget constraints and location capacity."
        )


# --- Agent Layer ---


class AgentError(RetentionIQError):
    """Base for agent-related errors."""


class PIILeakageError(AgentError):
    """Raised when PII is detected in agent output.

    This is a critical error — PII must NEVER reach the LLM or the user
    without masking. If this fires, the guardrail caught it, but we need
    to investigate why it appeared in the first place.
    """

    def __init__(self, field_type: str) -> None:
        self.field_type = field_type
        super().__init__(
            f"PII leakage detected: {field_type} found in agent output. "
            f"Output was blocked. Investigate the tool that produced this data."
        )


class GuardrailViolationError(AgentError):
    """Raised when agent output violates a guardrail rule."""

    def __init__(self, rule: str, details: str) -> None:
        self.rule = rule
        self.details = details
        super().__init__(f"Guardrail violation [{rule}]: {details}")
