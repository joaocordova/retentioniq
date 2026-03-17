# Architecture — RetentionIQ

## System Overview

RetentionIQ is structured as four layers, each with clear responsibilities and interfaces. Data flows bottom-up; decisions flow top-down.

```
                    ┌─────────────────────┐
                    │   Agent Layer       │  Natural language interface
                    │   (LangGraph)       │  for location managers
                    └─────────┬───────────┘
                              │ tool calls
                    ┌─────────▼───────────┐
                    │   Decision Engine   │  Causal effects + optimization
                    │   (DoWhy + Pyomo)   │  "what action, for whom, where"
                    └─────────┬───────────┘
                              │ features + scores
                    ┌─────────▼───────────┐
                    │   Model Layer       │  Survival, churn, LTV models
                    │   (MLflow tracked)  │  registered + versioned
                    └─────────┬───────────┘
                              │ features
                    ┌─────────▼───────────┐
                    │   Data Platform     │  Medallion architecture
                    │   (Dagster + Feast) │  bronze → silver → gold → features
                    └─────────────────────┘
```

## Data Platform

### Medallion Architecture

```
Raw Sources          Bronze              Silver              Gold
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ CSV/API  │───▶│ Schema       │───▶│ Deduped      │───▶│ Member       │
│ exports  │    │ validated    │    │ Cleaned      │    │ 360° view    │
│          │    │ Typed        │    │ Standardized │    │              │
│ Postgres │───▶│ Append-only  │───▶│ Business     │───▶│ Location     │
│ replica  │    │ Immutable    │    │ rules applied│    │ aggregates   │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                              │
                                                    ┌─────────▼──────────┐
                                                    │  Feast Feature     │
                                                    │  Store             │
                                                    │                    │
                                                    │  Online: Redis/PG  │
                                                    │  Offline: Parquet  │
                                                    └────────────────────┘
```

**Bronze:** Raw data as-is. Schema validation only (Great Expectations). No transformations. Append-only — we never modify raw data. This is our audit trail.

**Silver:** Cleaned and standardized. Deduplication (member records that appear in multiple exports). Timezone normalization. Null handling with explicit rules (documented in `src/data/silver/rules.py`). Business rule application: a member who cancels and re-enrolls within 30 days is treated as a single tenure, not two.

**Gold:** Business-ready analytics tables. Member 360° view (joins across visits, transactions, actions). Location daily/weekly/monthly aggregates. Cohort tables for survival analysis. Aggregator segment tables (separated by design, not as an afterthought).

### Data Quality

Great Expectations suites run at each layer transition:
- **Bronze → Silver:** Schema conformance, column types, null thresholds, referential integrity (every visit has a valid member_id)
- **Silver → Gold:** Business rule validation (no negative tenure, no future cancel dates, MRR calculations balance)
- **Gold → Features:** Feature distribution checks (no extreme outliers from bad joins), freshness SLA (features must be < 24h old)

Failed expectations block the pipeline and trigger alerts. No silent data corruption.

### Orchestration (Dagster)

```python
# Simplified asset graph
@asset
def bronze_members(raw_csv: str) -> pd.DataFrame: ...

@asset
def silver_members(bronze_members: pd.DataFrame) -> pd.DataFrame: ...

@asset
def gold_member_360(silver_members, silver_visits, silver_transactions) -> pd.DataFrame: ...

@asset
def feature_store_sync(gold_member_360) -> None:
    """Push materialized features to Feast online store."""

@asset
def churn_model(feature_store_sync) -> mlflow.pyfunc.PyFuncModel:
    """Train, evaluate, register model in MLflow."""

@asset
def causal_effects(gold_member_360, feature_store_sync) -> pd.DataFrame:
    """Estimate CATE using CausalForestDML."""

@asset
def budget_allocation(causal_effects) -> pd.DataFrame:
    """Run stochastic optimizer."""
```

Assets > tasks. Dagster's asset-based model means we define **what** we want (a fresh `gold_member_360` table) not **how** to get it. Dagster handles dependency resolution, caching, and incremental materialization.

## Model Layer

### Survival Analysis
- **Kaplan-Meier:** Non-parametric survival curves, stratified by contract type (Regular vs Aggregator) and enrollment cohort
- **Cox Proportional Hazards:** Semi-parametric model identifying hazard ratios for features (visit frequency, payment delays, plan type)
- **Accelerated Failure Time (AFT):** Parametric alternative when proportional hazards assumption is violated (common with aggregator members)
- Library: `lifelines`

### Churn Scoring
- **XGBoost** classifier for 30/60/90-day churn probability
- Features from Feast feature store (point-in-time correct)
- Class imbalance handled via scale_pos_weight (not SMOTE — synthetic minority oversampling creates data leakage in temporal splits)
- Evaluation: AUC-ROC, Precision-Recall curve, calibration plot (Brier score)
- Temporal cross-validation: train on months 1-12, validate on 13-15, test on 16-18. No random splits — this is time series.

### LTV Estimation
- Expected remaining tenure (from survival model) × expected monthly revenue
- Separate models for Regular (higher LTV, longer tenure) and Aggregator (lower margin, higher volatility)
- Used as weight in the optimization objective: retaining a high-LTV member is worth more.

### Model Registry (MLflow)
Every model is:
- Tracked: hyperparameters, metrics, artifacts (confusion matrix, SHAP plots, calibration curves)
- Versioned: model registry with staging → production promotion
- Tagged: data version (DVC hash), training date, feature set version
- Reproducible: `mlflow.log_artifact("dvc.lock")` — can recreate exact training data

## Decision Engine

### Causal Inference Pipeline

```
Step 1: Define DAG          Step 2: Identify           Step 3: Estimate
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ DoWhy Model      │       │ Backdoor         │       │ EconML           │
│                  │       │ criterion        │       │ CausalForestDML  │
│ Treatment:       │──────▶│                  │──────▶│                  │
│  retention_action│       │ Adjustment set:  │       │ CATE per member  │
│                  │       │  visit_freq,     │       │ (heterogeneous   │
│ Outcome:         │       │  tenure,         │       │  treatment       │
│  churned_30d     │       │  plan_type,      │       │  effects)        │
│                  │       │  season          │       │                  │
│ Confounders:     │       │                  │       │ Confidence       │
│  (from DAG)      │       │                  │       │ intervals        │
└──────────────────┘       └──────────────────┘       └──────────────────┘
                                                              │
                                                     Step 4: Refute
                                                     ┌──────────────────┐
                                                     │ Placebo treatment │
                                                     │ Random common    │
                                                     │   cause          │
                                                     │ Data subset      │
                                                     │ Bootstrap        │
                                                     └──────────────────┘
```

**Why this matters:** A predictive churn model tells you "this member has 80% probability of churning." A causal model tells you "if we send this member a discount offer, their churn probability drops by 12 percentage points — but only if they're a direct member in months 2-5 of tenure. For aggregator members, the same action has zero effect."

### Budget Optimizer

Two-stage stochastic program (Pyomo):

**Stage 1 (here-and-now):** Decide budget allocation per location per action type
**Stage 2 (wait-and-see):** Observe actual treatment effects (sampled from CATE confidence intervals) and compute retained revenue

```
Maximize: E[Σ_locations Σ_members (retained_revenue × P(retained | action, CATE_scenario))]

Subject to:
  Σ_locations budget_allocated ≤ total_budget
  budget_allocated[location] ≤ location_capacity[location]
  budget_allocated[location] ≥ min_budget[location]  (every location gets something)
  action_count[location] ≤ staff_hours[location]  (operational constraint)
```

The optimizer doesn't just find the best allocation — it finds the allocation that's **robust** to uncertainty in our causal estimates.

## Agent Layer

### Architecture (LangGraph)

```
                    ┌──────────┐
         ┌────────▶│ Analyst  │────────┐
         │         │ Agent    │        │
         │         └──────────┘        │
         │              │              │
    ┌────┴────┐         │         ┌────▼────┐
    │ Router  │         │         │ Strate- │
    │         │         │         │ gist    │
    │ Classif.│         │         │ Agent   │
    │ query   │         │         │         │
    └────┬────┘         │         └────┬────┘
         │              │              │
         │         ┌────▼────┐         │
         └────────▶│ Writer  │◀────────┘
                   │ Agent   │
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │ Guard-  │
                   │ rails   │
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │ Response│
                   └─────────┘
```

**Analyst Agent:** Has tools for SQL queries, survival analysis, and cohort comparison. Answers "what happened" questions.

**Strategist Agent:** Has tools for causal model inference and optimizer queries. Answers "what should we do" questions.

**Writer Agent:** Takes outputs from Analyst or Strategist, formats into manager-friendly language. No jargon, no p-values — just "focus on these 15 members this week, here's why."

**Router:** Classifies incoming query type and routes to appropriate agent(s). Simple queries (factual) go to Analyst only. Complex queries (diagnostic + prescriptive) chain Analyst → Strategist → Writer.

### Memory (pgvector)

- **Conversation memory:** Current session context (recent messages)
- **Episodic memory:** Past analyses per location stored as embeddings. "Last time we analyzed Location X (2 weeks ago), churn was 10%. Now it's 15% — escalating trend."
- **Semantic memory:** Domain knowledge (what is an aggregator member, what seasonal patterns look like) — retrieved when relevant

### Guardrails

1. **Input:** PII detection before sending to LLM. Member names, CPFs, emails → masked with tokens
2. **Output:** Validate response contains no PII, no harmful recommendations, no hallucinated numbers
3. **Confidence:** If agent confidence < threshold → "I'm not sure about this. I recommend consulting with the analytics team."
4. **Audit:** Every interaction logged: timestamp, user, input (masked), output, tools called, tokens used, latency

### Evaluation

Custom eval framework (not just "does the answer look good"):
- **Task success rate:** For 50+ canonical queries, does the agent produce a correct answer?
- **Tool selection accuracy:** Does the agent call the right tools in the right order?
- **Factual grounding:** Every number in the response must trace to a tool call result
- **Harmful output rate:** Frequency of PII leaks, hallucinated data, dangerous recommendations
- **Cost efficiency:** Tokens used per query (tracked over time)

---

## Architecture Decision Records

### ADR-001: Dagster over Airflow
**Context:** Need workflow orchestration for ML pipelines.
**Decision:** Dagster.
**Rationale:** Asset-based model aligns with how ML engineers think (I want fresh features, not "run step 3"). Better local development experience. Native Python typing. Built-in data quality integration. Airflow's task-based model requires more boilerplate and is harder to test locally.
**Trade-off:** Smaller community, fewer production case studies at scale.

### ADR-002: Batch over streaming for v1
**Context:** Member data updates daily. Could build real-time pipeline.
**Decision:** Batch (daily).
**Rationale:** Retention decisions are not real-time. A manager doesn't need to know a member is at risk within seconds — daily scoring is sufficient. Streaming adds complexity (Kafka, Flink, exactly-once semantics) with no business value for this use case. If real-time churn alerts become a requirement, we can add a streaming layer for scoring only, while keeping the causal/optimization pipeline batch.
**Trade-off:** 24h delay in risk scores.

### ADR-003: Separate models for Regular vs Aggregator
**Context:** Aggregator members (Gympass/TotalPass) behave differently — shorter tenure, different visit patterns, no direct billing relationship.
**Decision:** Separate survival models, separate churn models, separate causal DAGs.
**Rationale:** Pooling them introduces confounding. The causal structure is different: for aggregator members, the gym has no control over pricing or plan changes. Retention levers are limited to experience quality. A single model would learn an average effect that's wrong for both segments.
**Trade-off:** Smaller training set per model. Mitigated by having ~15% aggregator members × 2M = ~300K records — sufficient.

### ADR-004: pgvector over dedicated vector DB
**Context:** Need vector storage for agent memory and RAG.
**Decision:** pgvector extension on existing PostgreSQL.
**Rationale:** Already running PostgreSQL for operational data. pgvector adds vector similarity search without new infrastructure. At our scale (< 1M vectors), performance is adequate. Pinecone/Weaviate/Qdrant are better at billion-scale, which we don't need.
**Trade-off:** No advanced vector DB features (hybrid search, auto-indexing). Acceptable.

### ADR-005: Temporal cross-validation
**Context:** Evaluating churn models requires a validation strategy.
**Decision:** Expanding window temporal CV. Never random splits.
**Rationale:** Churn data is temporal. A model trained on January data predicting December churn would leak future information in a random split. We use expanding window: train on months 1-N, predict month N+1, advance. This mimics production exactly.
**Trade-off:** Fewer validation folds than k-fold CV. Compensated by using multiple cutoff dates.

### ADR-006: No hardcoded secrets
**Context:** Project will be open-sourced on GitHub. Credentials in git history are permanent.
**Decision:** All secrets via environment variables. Docker Compose uses `${VAR:-default}` syntax for safe local dev defaults. `.env` is in `.gitignore`. `.env.example` serves as template.
**Rationale:** Even "dev" passwords in docker-compose.yml end up in git history and can mislead users into deploying with default credentials. Environment variable substitution costs nothing and prevents credential leakage.
**Trade-off:** Slightly more setup friction (must create `.env` file). Mitigated by `.env.example` template and `make setup` automation.
