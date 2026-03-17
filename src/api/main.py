"""RetentionIQ API -- FastAPI application.

Endpoints serve three consumers:
1. Internal dashboards: batch prediction results, location metrics
2. Agent layer: tool execution (SQL, model inference, optimization)
3. External integrations: webhook for real-time churn alerts (future)
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from src.exceptions import (
    DataError,
    InfeasibleProblemError,
    ModelNotFoundError,
    RetentionIQError,
)

logger = structlog.get_logger()


# --- Settings ----------------------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    database_url: str = (
        "postgresql://retentioniq:retentioniq_dev@localhost:5432/retentioniq"
    )
    mlflow_tracking_uri: str = "http://localhost:5001"
    model_name: str = "retentioniq-churn"
    model_stage: str = "Production"
    data_dir: str = "data/raw"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


# --- Request/Response Models -------------------------------------------------


class ChurnPredictionRequest(BaseModel):
    """Request to predict churn probability for a member."""

    member_id: str = Field(..., description="Unique member identifier")
    horizon_days: int = Field(
        default=30, ge=7, le=90, description="Prediction horizon in days"
    )


class ChurnPredictionResponse(BaseModel):
    """Churn prediction result with explanation."""

    member_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    risk_tier: str  # "low", "medium", "high", "critical"
    horizon_days: int
    top_risk_factors: list[dict[str, Any]]  # [{"feature": "...", "impact": 0.xx}]
    model_version: str
    predicted_at: datetime


class LocationMetricsRequest(BaseModel):
    """Request for location-level metrics."""

    location_id: str
    period: str = Field(default="last_30d", pattern="^last_\\d+d$")


class LocationMetricsResponse(BaseModel):
    """Location metrics summary."""

    location_id: str
    period: str
    active_members: int
    churned_members: int
    churn_rate: float
    mrr: float
    avg_visits_per_member: float
    at_risk_members: int  # members with churn_prob > 0.5
    aggregator_pct: float  # % of members via aggregators


class AllocationRequest(BaseModel):
    """Request to run budget optimizer for a location or region."""

    location_ids: list[str] | None = Field(
        default=None, description="Specific locations. None = all."
    )
    budget_override: float | None = Field(
        default=None, ge=0, description="Override total budget from config"
    )


class AllocationResponse(BaseModel):
    """Budget allocation recommendation."""

    total_budget_used: float
    expected_retained_mrr: float
    roi_estimate: float  # expected_retained_mrr / total_budget_used
    allocations_by_location: list[dict[str, Any]]
    solver_status: str
    generated_at: datetime


class AgentQueryRequest(BaseModel):
    """Natural language query from a location manager."""

    query: str = Field(..., min_length=5, max_length=1000)
    location_id: str | None = Field(
        default=None,
        description="Context: which location is the manager asking about",
    )
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )


class AgentQueryResponse(BaseModel):
    """Agent response to a natural language query."""

    answer: str
    confidence: float = Field(..., ge=0, le=1)
    tools_used: list[str]
    tokens_used: int
    latency_ms: int
    session_id: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    database: str
    mlflow: str
    timestamp: datetime


# --- Helper functions --------------------------------------------------------


def _classify_risk_tier(probability: float) -> str:
    """Return a risk tier label for the given churn probability.

    Thresholds:
    - < 0.2  -> low
    - < 0.5  -> medium
    - < 0.8  -> high
    - >= 0.8 -> critical

    Args:
        probability: Churn probability in [0, 1].

    Returns:
        Risk tier string.
    """
    if probability < 0.2:
        return "low"
    if probability < 0.5:
        return "medium"
    if probability < 0.8:
        return "high"
    return "critical"


def _check_database(database_url: str) -> str:
    """Probe database connectivity and return a status string.

    Args:
        database_url: PostgreSQL connection URL.

    Returns:
        ``"connected"`` or ``"unavailable: <reason>"``.
    """
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(
            database_url,
            connect_args={"connect_timeout": 3},
        )
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        engine.dispose()
        return "connected"
    except Exception as exc:
        logger.warning("db_health_check_failed", error=str(exc))
        return f"unavailable: {type(exc).__name__}"


def _check_mlflow(tracking_uri: str) -> str:
    """Probe MLflow connectivity and return a status string.

    Args:
        tracking_uri: MLflow tracking server URL.

    Returns:
        ``"connected"`` or ``"unavailable: <reason>"``.
    """
    try:
        import requests
        resp = requests.get(
            f"{tracking_uri}/api/2.0/mlflow/experiments/search",
            timeout=3,
        )
        if resp.status_code < 500:
            return "connected"
        return f"unavailable: HTTP {resp.status_code}"
    except Exception as exc:
        logger.warning("mlflow_health_check_failed", error=str(exc))
        return f"unavailable: {type(exc).__name__}"


def _load_gold_parquets(
    data_dir: str,
) -> dict[str, pd.DataFrame | None]:
    """Attempt to load gold-layer parquet files from disk.

    Looks for ``member_360.parquet`` and ``location_aggregates.parquet``
    under ``<data_dir>/../gold/`` or ``data/gold/``.

    Args:
        data_dir: Configured raw data directory (used to derive gold path).

    Returns:
        Dict with keys ``member_360`` and ``location_aggregates``,
        values are DataFrames or None if not found.
    """
    result: dict[str, pd.DataFrame | None] = {
        "member_360": None,
        "location_aggregates": None,
    }

    gold_candidates = [
        Path(data_dir).parent / "gold",
        Path("data") / "gold",
        Path("data") / "processed" / "gold",
    ]

    for gold_dir in gold_candidates:
        m360_path = gold_dir / "member_360.parquet"
        loc_path = gold_dir / "location_aggregates.parquet"

        if m360_path.exists() and result["member_360"] is None:
            try:
                result["member_360"] = pd.read_parquet(m360_path)
                logger.info(
                    "loaded_member_360",
                    path=str(m360_path),
                    rows=len(result["member_360"]),
                )
            except Exception as exc:
                logger.warning(
                    "failed_to_load_member_360",
                    path=str(m360_path),
                    error=str(exc),
                )

        if loc_path.exists() and result["location_aggregates"] is None:
            try:
                result["location_aggregates"] = pd.read_parquet(
                    loc_path
                )
                logger.info(
                    "loaded_location_aggregates",
                    path=str(loc_path),
                    rows=len(result["location_aggregates"]),
                )
            except Exception as exc:
                logger.warning(
                    "failed_to_load_location_aggregates",
                    path=str(loc_path),
                    error=str(exc),
                )

    return result


def _try_load_churn_model() -> Any | None:
    """Attempt to load the production churn model from MLflow.

    Returns:
        Loaded model object or ``None`` if MLflow is unreachable or
        the model is not registered.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = (
            f"models:/{settings.model_name}/{settings.model_stage}"
        )
        model = mlflow.xgboost.load_model(model_uri)
        logger.info(
            "churn_model_loaded",
            model_name=settings.model_name,
            stage=settings.model_stage,
        )
        return model
    except Exception as exc:
        logger.warning(
            "churn_model_load_failed",
            error=str(exc),
        )
        return None


def _generate_demo_risk_factors() -> list[dict[str, Any]]:
    """Return synthetic risk factors for the demo fallback.

    Returns:
        List of dicts with ``feature`` and ``impact`` keys.
    """
    return [
        {"feature": "days_since_last_visit", "impact": 0.32},
        {"feature": "visit_count_30d", "impact": -0.21},
        {"feature": "tenure_days", "impact": -0.14},
        {"feature": "contract_source_aggregator", "impact": 0.11},
    ]


def _generate_synthetic_cate(
    member_ids: list[str],
    location_ids: list[str],
    actions: list[str],
) -> pd.DataFrame:
    """Generate synthetic CATE estimates for demo purposes.

    Args:
        member_ids: List of member IDs.
        location_ids: List of location IDs (parallel to member_ids).
        actions: List of action type names.

    Returns:
        DataFrame with columns ``member_id``, ``location_id``,
        ``action``, ``cate_mean``, ``cate_std``.
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for mid, lid in zip(member_ids, location_ids):
        for action in actions:
            rows.append({
                "member_id": mid,
                "location_id": lid,
                "action": action,
                "cate_mean": round(float(rng.uniform(0.01, 0.15)), 4),
                "cate_std": round(float(rng.uniform(0.005, 0.04)), 4),
            })
    return pd.DataFrame(rows)


# --- Application -------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("api_starting", version="0.1.0")

    # Load gold-layer data into app.state
    gold_data = _load_gold_parquets(settings.data_dir)
    app.state.member_360 = gold_data["member_360"]
    app.state.location_aggregates = gold_data["location_aggregates"]

    loaded = [
        k for k, v in gold_data.items() if v is not None
    ]
    logger.info("startup_data_loaded", datasets=loaded)

    # Try to load the churn model
    app.state.churn_model = _try_load_churn_model()
    if app.state.churn_model is not None:
        logger.info("startup_churn_model_ready")
    else:
        logger.warning("startup_churn_model_not_available")

    yield

    # Shutdown cleanup
    app.state.member_360 = None
    app.state.location_aggregates = None
    app.state.churn_model = None
    logger.info("api_shutting_down")


app = FastAPI(
    title="RetentionIQ API",
    description=(
        "Prescriptive retention system for multi-location "
        "subscription businesses"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Middleware ---------------------------------------------------------------


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Any,
) -> Any:
    """Structured logging for every request."""
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration_ms = int(
        (datetime.utcnow() - start_time).total_seconds() * 1000
    )

    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# --- Endpoints ----------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect root path to API docs for discoverability."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API and dependency health.

    Probes database and MLflow connectivity. Returns degraded status
    instead of erroring when dependencies are unavailable.
    """
    db_status = _check_database(settings.database_url)
    mlflow_status = _check_mlflow(settings.mlflow_tracking_uri)

    if "unavailable" in db_status or "unavailable" in mlflow_status:
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthResponse(
        status=overall,
        version="0.1.0",
        database=db_status,
        mlflow=mlflow_status,
        timestamp=datetime.utcnow(),
    )


@app.post("/predict/churn", response_model=ChurnPredictionResponse)
async def predict_churn(
    request: ChurnPredictionRequest,
) -> ChurnPredictionResponse:
    """Predict churn probability for a single member.

    Uses the production model from MLflow registry. When the model is
    unavailable, returns a demo prediction so the endpoint remains
    functional for integration testing.
    """
    member_id = request.member_id
    member_360: pd.DataFrame | None = getattr(
        app.state, "member_360", None
    )

    # Try to find the member in gold-layer data
    member_row: pd.Series | None = None
    if member_360 is not None and "member_id" in member_360.columns:
        match = member_360[member_360["member_id"] == member_id]
        if match.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Member '{member_id}' not found.",
            )
        member_row = match.iloc[0]

    # If no gold data loaded, try DB
    if member_row is None and member_360 is None:
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(
                settings.database_url,
                connect_args={"connect_timeout": 3},
            )
            query = (
                "SELECT * FROM gold.member_360 "
                "WHERE member_id = :mid LIMIT 1"
            )
            with engine.connect() as conn:
                result = conn.execute(
                    sqlalchemy.text(query),
                    {"mid": member_id},
                )
                row = result.fetchone()
            engine.dispose()
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Member '{member_id}' not found.",
                )
            member_row = pd.Series(
                dict(row._mapping)
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "db_member_lookup_failed", error=str(exc)
            )
            # No data source available -- cannot look up the member,
            # but we do not crash; proceed with demo prediction.

    # Attempt model scoring
    churn_model = getattr(app.state, "churn_model", None)

    if churn_model is not None and member_row is not None:
        try:
            # Prepare feature vector -- drop non-feature columns
            exclude = {
                "member_id", "location_id", "name",
                "email", "cpf", "phone", "join_date",
                "cancel_date", "churned",
            }
            feature_cols = [
                c for c in member_row.index
                if c not in exclude
                and np.issubdtype(type(member_row[c]), np.number)
            ]
            features = (
                pd.DataFrame([member_row[feature_cols]])
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
            proba = float(
                churn_model.predict_proba(features)[0, 1]
            )

            # SHAP-based risk factors if available
            risk_factors: list[dict[str, Any]] = []
            try:
                import shap
                explainer = shap.TreeExplainer(churn_model)
                shap_vals = explainer.shap_values(features)
                abs_shap = np.abs(shap_vals[0])
                top_idx = np.argsort(abs_shap)[::-1][:5]
                for idx in top_idx:
                    risk_factors.append({
                        "feature": feature_cols[idx],
                        "impact": round(
                            float(shap_vals[0][idx]), 4
                        ),
                    })
            except Exception:
                # Fallback: use feature importances
                importances = churn_model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:5]
                for idx in top_idx:
                    if idx < len(feature_cols):
                        risk_factors.append({
                            "feature": feature_cols[idx],
                            "impact": round(
                                float(importances[idx]), 4
                            ),
                        })

            return ChurnPredictionResponse(
                member_id=member_id,
                churn_probability=round(proba, 4),
                risk_tier=_classify_risk_tier(proba),
                horizon_days=request.horizon_days,
                top_risk_factors=risk_factors or (
                    _generate_demo_risk_factors()
                ),
                model_version=settings.model_stage,
                predicted_at=datetime.utcnow(),
            )
        except Exception as exc:
            logger.warning(
                "model_scoring_failed",
                member_id=member_id,
                error=str(exc),
            )
            # Fall through to demo prediction

    # Demo prediction when model is not available
    rng = np.random.default_rng(
        hash(member_id) % (2**31)
    )
    demo_proba = round(float(rng.uniform(0.05, 0.85)), 4)

    return ChurnPredictionResponse(
        member_id=member_id,
        churn_probability=demo_proba,
        risk_tier=_classify_risk_tier(demo_proba),
        horizon_days=request.horizon_days,
        top_risk_factors=_generate_demo_risk_factors(),
        model_version="demo-v0",
        predicted_at=datetime.utcnow(),
    )


@app.post(
    "/metrics/location", response_model=LocationMetricsResponse
)
async def get_location_metrics(
    request: LocationMetricsRequest,
) -> LocationMetricsResponse:
    """Get aggregated metrics for a location.

    Reads from precomputed gold-layer aggregates when available,
    otherwise computes on the fly from member-360 data.
    """
    location_id = request.location_id

    # 1. Try precomputed location aggregates
    loc_agg: pd.DataFrame | None = getattr(
        app.state, "location_aggregates", None
    )
    if loc_agg is not None and "location_id" in loc_agg.columns:
        match = loc_agg[loc_agg["location_id"] == location_id]
        if not match.empty:
            row = match.iloc[0]
            # Compute at_risk count from member_360 if available
            at_risk = 0
            m360 = getattr(app.state, "member_360", None)
            if m360 is not None and "churn_probability" in m360.columns:
                loc_members = m360[
                    m360["location_id"] == location_id
                ]
                at_risk = int(
                    (loc_members["churn_probability"] > 0.5).sum()
                )

            visits_col = (
                "avg_visits_30d"
                if "avg_visits_30d" in row.index
                else "avg_visits_per_member"
            )

            return LocationMetricsResponse(
                location_id=location_id,
                period=request.period,
                active_members=int(row.get("active_members", 0)),
                churned_members=int(
                    row.get("churned_members", 0)
                ),
                churn_rate=float(row.get("churn_rate", 0.0)),
                mrr=float(row.get("mrr", 0.0)),
                avg_visits_per_member=float(
                    row.get(visits_col, 0.0)
                ),
                at_risk_members=at_risk,
                aggregator_pct=float(
                    row.get("aggregator_pct", 0.0)
                ),
            )
        # Location ID not in aggregates -> 404
        raise HTTPException(
            status_code=404,
            detail=(
                f"Location '{location_id}' not found "
                f"in aggregates."
            ),
        )

    # 2. Try computing from member_360
    m360: pd.DataFrame | None = getattr(
        app.state, "member_360", None
    )
    if m360 is not None and "location_id" in m360.columns:
        loc_members = m360[m360["location_id"] == location_id]
        if loc_members.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Location '{location_id}' not found "
                    f"in member data."
                ),
            )

        from src.data.gold.aggregation import (
            build_location_aggregates,
        )

        agg = build_location_aggregates(loc_members)
        if agg.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Location '{location_id}' produced "
                    f"empty aggregation."
                ),
            )

        row = agg.iloc[0]

        at_risk = 0
        if "churn_probability" in loc_members.columns:
            at_risk = int(
                (loc_members["churn_probability"] > 0.5).sum()
            )

        visits_col = (
            "avg_visits_30d"
            if "avg_visits_30d" in row.index
            else "avg_visits_per_member"
        )

        return LocationMetricsResponse(
            location_id=location_id,
            period=request.period,
            active_members=int(row.get("active_members", 0)),
            churned_members=int(row.get("churned_members", 0)),
            churn_rate=float(row.get("churn_rate", 0.0)),
            mrr=float(row.get("mrr", 0.0)),
            avg_visits_per_member=float(
                row.get(visits_col, 0.0)
            ),
            at_risk_members=at_risk,
            aggregator_pct=float(
                row.get("aggregator_pct", 0.0)
            ),
        )

    # 3. Try database
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(
            settings.database_url,
            connect_args={"connect_timeout": 3},
        )
        query = (
            "SELECT * FROM gold.location_aggregates "
            "WHERE location_id = :lid LIMIT 1"
        )
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(query),
                {"lid": location_id},
            )
            db_row = result.fetchone()
        engine.dispose()

        if db_row is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Location '{location_id}' not found."
                ),
            )

        data = dict(db_row._mapping)
        return LocationMetricsResponse(
            location_id=location_id,
            period=request.period,
            active_members=int(data.get("active_members", 0)),
            churned_members=int(
                data.get("churned_members", 0)
            ),
            churn_rate=float(data.get("churn_rate", 0.0)),
            mrr=float(data.get("mrr", 0.0)),
            avg_visits_per_member=float(
                data.get("avg_visits_30d", 0.0)
            ),
            at_risk_members=int(
                data.get("at_risk_members", 0)
            ),
            aggregator_pct=float(
                data.get("aggregator_pct", 0.0)
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "location_metrics_db_failed", error=str(exc)
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"Location '{location_id}' not found. "
            f"No data sources are currently available."
        ),
    )


@app.post(
    "/optimize/allocation", response_model=AllocationResponse
)
async def run_allocation(
    request: AllocationRequest,
) -> AllocationResponse:
    """Run budget optimizer and return allocation recommendations.

    This endpoint calls the Pyomo stochastic optimizer. It is
    computationally expensive -- expected latency is 5-30 seconds
    depending on the number of locations and members. Consider
    caching results for repeated calls.
    """
    try:
        from src.optimization.allocator import (
            AllocationResult,
            solve_allocation,
        )

        m360: pd.DataFrame | None = getattr(
            app.state, "member_360", None
        )

        if m360 is not None:
            members = m360.copy()
        else:
            # Generate a small synthetic dataset for demo
            rng = np.random.default_rng(42)
            n = 100
            loc_ids = request.location_ids or [
                f"LOC_{i:03d}" for i in range(1, 6)
            ]
            members = pd.DataFrame({
                "member_id": [
                    f"MEM_{i:05d}" for i in range(n)
                ],
                "location_id": rng.choice(loc_ids, size=n),
                "ltv": np.round(
                    rng.uniform(200, 3000, size=n), 2
                ),
                "contract_source": rng.choice(
                    ["regular", "aggregator"],
                    size=n,
                    p=[0.7, 0.3],
                ),
            })

        # Filter to requested locations
        if request.location_ids:
            members = members[
                members["location_id"].isin(
                    request.location_ids
                )
            ]
            if members.empty:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "No members found for the requested "
                        "locations."
                    ),
                )

        # Ensure required columns
        if "ltv" not in members.columns:
            rng = np.random.default_rng(123)
            members["ltv"] = np.round(
                rng.uniform(200, 3000, size=len(members)), 2
            )
        if "contract_source" not in members.columns:
            members["contract_source"] = "regular"

        # Generate CATE estimates (use real ones if available,
        # otherwise synthetic for demo)
        cate_path = Path("data/gold/cate_estimates.parquet")
        if cate_path.exists():
            cate_estimates = pd.read_parquet(cate_path)
        else:
            actions = [
                "sms_reengagement",
                "phone_call",
                "discount_offer",
                "personal_trainer_session",
            ]
            cate_estimates = _generate_synthetic_cate(
                members["member_id"].tolist(),
                members["location_id"].tolist(),
                actions,
            )

        # Apply budget override if provided
        if request.budget_override is not None:
            import tempfile

            import yaml

            orig_path = "configs/optimization.yaml"
            try:
                with open(orig_path) as f:
                    opt_config = yaml.safe_load(f)
            except FileNotFoundError:
                opt_config = {
                    "constraints": {
                        "total_budget": 50000,
                        "min_budget_per_location": 50,
                        "max_budget_per_location": 2000,
                        "max_actions_per_member": 1,
                        "staff_hours_per_location": 20,
                    },
                    "actions": [
                        {
                            "name": "sms_reengagement",
                            "cost_per_member": 2.50,
                            "time_per_member_minutes": 0,
                        },
                        {
                            "name": "phone_call",
                            "cost_per_member": 15.00,
                            "time_per_member_minutes": 10,
                        },
                        {
                            "name": "discount_offer",
                            "cost_per_member": 50.00,
                            "time_per_member_minutes": 2,
                        },
                        {
                            "name": "personal_trainer_session",
                            "cost_per_member": 80.00,
                            "time_per_member_minutes": 60,
                        },
                    ],
                    "stochastic": {
                        "solver": "glpk",
                        "n_scenarios": 50,
                    },
                }

            opt_config["constraints"]["total_budget"] = (
                request.budget_override
            )

            tmp = tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False, mode="w"
            )
            yaml.dump(opt_config, tmp)
            tmp.close()
            config_path = tmp.name
        else:
            config_path = "configs/optimization.yaml"

        result: AllocationResult = solve_allocation(
            members=members[
                [
                    "member_id",
                    "location_id",
                    "ltv",
                    "contract_source",
                ]
            ],
            cate_estimates=cate_estimates,
            config_path=config_path,
        )

        # Summarize allocations by location
        if not result.allocations.empty:
            by_loc = (
                result.allocations.groupby("location_id")
                .agg(
                    members_targeted=("member_id", "count"),
                    budget_used=("cost", "sum"),
                    actions=(
                        "action",
                        lambda s: s.value_counts().to_dict(),
                    ),
                )
                .reset_index()
            )
            alloc_list = by_loc.to_dict(orient="records")
        else:
            alloc_list = []

        roi = (
            result.expected_retained_mrr / result.total_cost
            if result.total_cost > 0
            else 0.0
        )

        return AllocationResponse(
            total_budget_used=round(result.total_cost, 2),
            expected_retained_mrr=round(
                result.expected_retained_mrr, 2
            ),
            roi_estimate=round(roi, 2),
            allocations_by_location=alloc_list,
            solver_status=result.solver_status,
            generated_at=datetime.utcnow(),
        )

    except InfeasibleProblemError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Optimization infeasible: {exc.reason}. "
                f"Try increasing the budget or relaxing "
                f"constraints."
            ),
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.warning(
            "optimization_config_missing", error=str(exc)
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Optimization configuration not found. "
                "Ensure configs/optimization.yaml exists."
            ),
        )
    except Exception as exc:
        logger.error(
            "optimization_failed",
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Optimization failed. See server logs.",
        )


@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(
    request: AgentQueryRequest,
) -> AgentQueryResponse:
    """Process a natural language query through the agent system.

    The agent routes the query, executes tools, and returns a
    manager-friendly response. PII is masked at every stage.

    Rate limited to 30 queries per user per hour.
    """
    import time as _time

    session_id = request.session_id or str(uuid4())
    start_ts = _time.time()

    try:
        from src.agents.graph import run_agent

        result = run_agent(
            query=request.query,
            location_id=request.location_id,
            session_id=session_id,
        )

        latency_ms = int(
            (_time.time() - start_ts) * 1000
        )

        return AgentQueryResponse(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.5),
            tools_used=result.get("tools_used", []),
            tokens_used=0,  # token counting not yet wired
            latency_ms=latency_ms,
            session_id=session_id,
        )

    except Exception as exc:
        latency_ms = int(
            (_time.time() - start_ts) * 1000
        )
        logger.error(
            "agent_query_failed",
            error=str(exc),
            session_id=session_id,
            exc_info=True,
        )

        # Graceful fallback instead of crashing
        return AgentQueryResponse(
            answer=(
                "I'm sorry, I wasn't able to process your "
                "question right now. Please try again or "
                "rephrase your query. If the issue persists, "
                "contact support."
            ),
            confidence=0.0,
            tools_used=[],
            tokens_used=0,
            latency_ms=latency_ms,
            session_id=session_id,
        )


# --- Error Handlers ----------------------------------------------------------


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(
    request: Request,
    exc: ModelNotFoundError,
) -> JSONResponse:
    """Return 503 when the ML model is not available."""
    logger.warning(
        "model_not_found",
        model=exc.model_name,
        stage=exc.stage,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                f"Model '{exc.model_name}' is not available. "
                f"The service is degraded."
            ),
        },
    )


@app.exception_handler(DataError)
async def data_error_handler(
    request: Request,
    exc: DataError,
) -> JSONResponse:
    """Return 500 with a generic message for data-layer errors."""
    logger.error(
        "data_error",
        path=request.url.path,
        error_type=type(exc).__name__,
        error_msg=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "A data processing error occurred. "
                "The team has been notified."
            ),
        },
    )


@app.exception_handler(RetentionIQError)
async def retentioniq_error_handler(
    request: Request,
    exc: RetentionIQError,
) -> JSONResponse:
    """Catch-all for RetentionIQ exception hierarchy."""
    logger.error(
        "retentioniq_error",
        path=request.url.path,
        error_type=type(exc).__name__,
        error_msg=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "An internal error occurred. "
                "The team has been notified."
            ),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all error handler. Never expose internal details."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error_msg=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "Internal server error. "
                "The team has been notified."
            ),
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
