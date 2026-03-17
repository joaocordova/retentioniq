# RetentionIQ

**Prescriptive retention system for multi-location subscription businesses.**

Not another churn prediction model. This system answers a harder question: *given limited budget, which retention action should we take, for which customer segment, at which location, to maximize retained revenue?*

Built with real operational constraints: 250+ independent locations, mixed contract types (direct + aggregator), LGPD/GDPR compliance, and managers who need answers in plain language — not dashboards.

---

## The Problem

Most subscription businesses approach retention backwards. They build a churn model, score customers, and hand a ranked list to operations. Then operations does... whatever they were already doing, just slightly more targeted.

The actual questions that matter are causal, not predictive:
- **Does** sending a win-back SMS reduce churn, or do we just think it does because we send it to people who were already likely to stay?
- If we have R$50K/month for retention across 250 locations, **how** should we allocate it?
- For members who joined through aggregators (Gympass/TotalPass), are the same retention levers effective, or is it a fundamentally different problem?

This project builds the full pipeline from raw transactional data to prescriptive recommendations, with an AI agent layer that lets non-technical managers interact with the system naturally.

## What This Is (and Isn't)

**This is** a production-oriented ML system that demonstrates:
- End-to-end data engineering with a medallion architecture
- Survival analysis and causal inference applied to a real retention problem
- Budget optimization under uncertainty using stochastic programming
- LLM agents with tool use, memory, guardrails, and evaluation
- MLOps: experiment tracking, data versioning, CI/CD, monitoring, drift detection

**This is not:**
- A notebook with `model.fit()` and a confusion matrix
- A tutorial wrapped in a GitHub repo
- A Kaggle competition solution optimizing for leaderboard position

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RetentionIQ                          │
│                                                             │
│  ┌────────────┐   ┌────────────┐   ┌────────────────────┐  │
│  │   Data      │   │   Models   │   │   Decision         │  │
│  │   Platform  │──▶│   Layer    │──▶│   Engine           │  │
│  │            │   │            │   │                    │  │
│  │ Bronze/    │   │ Survival   │   │ Causal Forests     │  │
│  │ Silver/    │   │ Analysis   │   │ (heterogeneous     │  │
│  │ Gold       │   │            │   │  treatment effects) │  │
│  │            │   │ Churn      │   │                    │  │
│  │ Feast      │   │ Scoring    │   │ Stochastic         │  │
│  │ (features) │   │            │   │ Optimizer          │  │
│  │            │   │ LTV        │   │ (budget allocation) │  │
│  └────────────┘   └────────────┘   └────────────────────┘  │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    Agent Layer                         │ │
│  │  LangGraph orchestration · pgvector memory            │ │
│  │  Tool use (SQL, models, optimizer) · Guardrails       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Infra: Docker · FastAPI · MLflow · Dagster · CI/CD   │ │
│  │  Evidently (monitoring) · Great Expectations (DQ)     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

## Dataset

This project uses the [Fitness Club Member Dataset](https://www.kaggle.com/) combined with synthetic multi-location and contract data generated to simulate a franchise network. The synthetic generation is fully documented and reproducible — see `scripts/generate_franchise_data.py`.

Why synthetic augmentation? Real franchise data is proprietary. But the patterns modeled here — seasonal enrollment peaks in January, aggregator vs. direct member behavior differences, location-level heterogeneity — are grounded in domain knowledge from operating a 250+ unit fitness franchise network.

**Data scale:** ~2M member records across 250 locations, 18 months of transactional history.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Orchestration | Dagster over Airflow | Asset-based (not task-based), better type safety, native Python, simpler local dev. Airflow is battle-tested but overengineered for this scale. |
| Feature Store | Feast | Open-source, works with PostgreSQL, handles point-in-time joins correctly (critical for survival analysis features). |
| Causal Inference | DoWhy + EconML | DoWhy forces you to declare assumptions as a DAG before estimating — this is a feature, not a bug. EconML's CausalForestDML gives heterogeneous treatment effects out of the box. |
| Optimization | Pyomo | Handles stochastic programming natively. PuLP is simpler but can't do two-stage stochastic programs. |
| Agent Framework | LangGraph | State machines > chains for production agents. Explicit state, conditional routing, checkpoints, human-in-the-loop. |
| Vector Store | pgvector | Already running PostgreSQL for operational data. Zero additional infra cost. Good enough for agent memory at this scale. |
| Serving | FastAPI | Industry standard for ML APIs. Async, Pydantic validation, OpenAPI docs for free. |
| Monitoring | Evidently | Open-source, handles data drift + model performance + target drift. Integrates with Dagster for automated checks. |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full ADR (Architecture Decision Records).

## Project Structure

```
retentioniq/
├── src/
│   ├── data/              # Medallion layers: bronze → silver → gold
│   │   ├── bronze/        # Raw ingestion, schema validation
│   │   ├── silver/        # Cleaning, deduplication, standardization
│   │   ├── gold/          # Business-ready tables, aggregations
│   │   └── quality/       # Great Expectations suites
│   ├── features/          # Feature engineering + Feast store
│   ├── models/
│   │   ├── survival/      # Kaplan-Meier, Cox PH, AFT models
│   │   ├── churn/         # Classification models (XGBoost, LightGBM)
│   │   └── ltv/           # Customer lifetime value estimation
│   ├── causal/
│   │   ├── dag.py         # Causal DAG definition (DoWhy)
│   │   ├── effects.py     # ATE/CATE estimation (EconML)
│   │   └── forests.py     # CausalForestDML for personalized effects
│   ├── optimization/
│   │   ├── allocator.py   # Budget allocation (deterministic)
│   │   └── stochastic.py  # Two-stage stochastic program (Pyomo)
│   ├── agents/
│   │   ├── graph.py       # LangGraph state machine
│   │   ├── tools.py       # SQL, model inference, optimizer tools
│   │   ├── memory.py      # pgvector long-term memory
│   │   ├── guardrails.py  # PII masking, output validation
│   │   └── eval/          # Agent evaluation framework
│   ├── api/               # FastAPI endpoints
│   └── monitoring/        # Evidently reports, drift detection
├── tests/
│   ├── unit/              # Function-level tests
│   └── integration/       # Pipeline-level tests
├── notebooks/             # Exploration only — no production code here
│   ├── 01_eda.ipynb
│   ├── 02_survival_analysis.ipynb
│   ├── 03_causal_dag.ipynb
│   ├── 04_treatment_effects.ipynb
│   ├── 05_optimization.ipynb
│   └── 06_agent_prototype.ipynb
├── configs/               # YAML configs (no hardcoded params)
├── scripts/               # One-off scripts (data generation, migration)
├── docs/
│   ├── ARCHITECTURE.md    # System design + ADRs
│   ├── PRD.md             # Product requirements
│   ├── SETUP.md           # Development setup
│   └── RUNBOOK.md         # Operational runbook
├── .claude/               # Claude Code skill + context
├── .github/workflows/     # CI/CD
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── dagster_workspace.yaml
```

## Getting Started

```bash
# Clone and setup
git clone https://github.com/joaocordova/retentioniq.git
cd retentioniq

# Create environment and install dependencies
make setup

# Generate synthetic franchise data
make data

# Run the full pipeline (Dagster)
make pipeline

# Start the API
make serve

# Run tests
make test

# Start monitoring dashboard
make monitor
```

See [docs/SETUP.md](docs/SETUP.md) for detailed instructions.

## Notebooks — The Story

The notebooks tell the analytical story. They're exploration, not production code.

| # | Notebook | Question Answered |
|---|---------|-------------------|
| 01 | EDA | What does member behavior look like across 250 locations? Where are the patterns? |
| 02 | Survival Analysis | How long do members stay? Is the survival curve different for aggregator vs. direct? |
| 03 | Causal DAG | What causes churn? (Not: what correlates with churn.) |
| 04 | Treatment Effects | Which retention actions actually work? For whom? |
| 05 | Optimization | Given budget constraints, what's the optimal allocation across locations? |
| 06 | Agent Prototype | Can a manager ask "why is churn up at location X?" and get a causal answer? |

## What I Learned Building This

**Causal inference changes how you think about ML.** Most ML projects optimize prediction accuracy. But a model that perfectly predicts who will churn is useless if you can't identify which intervention prevents it. The shift from "predict churn" to "estimate the causal effect of action X on segment Y" is fundamental — and surprisingly underrepresented in most ML engineering portfolios.

**Agents in production are 80% engineering, 20% prompting.** The hard part isn't getting an LLM to generate a good answer. It's building the guardrails, evaluation framework, memory management, and fallback strategies that make it reliable enough for a non-technical user to trust. Every agent demo looks impressive; very few survive contact with real users.

**Optimization under uncertainty is the bridge between analytics and operations.** A point estimate of treatment effect is useful. A budget allocation that accounts for the uncertainty in those estimates and remains feasible across scenarios — that's what actually drives business decisions.

## Tech Stack

**Data:** PostgreSQL (pgvector), DVC, Great Expectations, Feast
**ML:** scikit-learn, XGBoost, lifelines (survival), DoWhy, EconML, Pyomo
**LLM/Agents:** LangChain, LangGraph, pgvector (memory)
**Serving:** FastAPI, Docker, uvicorn
**MLOps:** MLflow, Dagster, Evidently, GitHub Actions
**Testing:** pytest, Great Expectations, custom agent eval framework

## License

MIT

## Author

**João Cordova** — Senior Data Analyst → AI Systems Architect
Economics background. Specialization in Data Science & Operations Research (USP).
Building data systems for a 250+ unit fitness franchise network.

[LinkedIn](https://linkedin.com/in/joaocordova) · [GitHub](https://github.com/joaocordova)
