# Product Requirements Document — RetentionIQ

**Author:** João Cordova
**Status:** In Development
**Last Updated:** March 2026

---

## 1. Problem Statement

A fitness franchise network with 250+ independently operated locations loses approximately 8-12% of its member base monthly. Current retention efforts are reactive (contact members after missed visits) and uniform (same intervention for all segments). There is no mechanism to:

1. Distinguish whether a retention action **causes** reduced churn or merely **correlates** with it (selection bias: we contact "at-risk" members who may have stayed anyway)
2. Personalize actions by member segment (a month-2 member who joined via Gympass responds differently than a month-10 direct member)
3. Allocate retention budget optimally across locations with heterogeneous capacity and member profiles
4. Enable location managers — who are fitness professionals, not analysts — to understand and act on retention intelligence

## 2. Target Users

| Persona | Role | Need |
|---------|------|------|
| **Location Manager** | Runs day-to-day operations at a single gym | "Tell me who to call this week and what to offer them" |
| **Regional Director** | Oversees 15-30 locations | "Which locations need more retention budget? Which actions are working?" |
| **C-Level / Franchisor** | Strategic decisions for the network | "What's the ROI of our retention program? How should we allocate next quarter's budget?" |

## 3. Core Capabilities

### 3.1 Churn Risk Scoring (Predictive)
- Score each active member on probability of churning in next 30/60/90 days
- Separate models for direct and aggregator members (fundamentally different behaviors)
- Features: visit frequency trends, contract age, payment history, seasonal signals
- Refresh: daily batch scoring, stored in feature store for real-time retrieval

### 3.2 Causal Treatment Effect Estimation (Causal)
- Estimate the causal effect of each retention action on churn probability
- Actions modeled: SMS re-engagement, discount offer, personal trainer session, plan upgrade offer
- Heterogeneous effects: estimate CATE (Conditional Average Treatment Effect) by member segment
- Key distinction: Regular vs Aggregator members have different causal graphs
- Robustness: sensitivity analysis for unmeasured confounding

### 3.3 Budget Optimization (Prescriptive)
- Given: total retention budget, per-location capacity constraints, estimated treatment effects
- Find: optimal allocation of budget across locations and actions to maximize retained MRR
- Handle uncertainty: two-stage stochastic program with scenarios for treatment effect estimates
- Output: actionable allocation table per location per action

### 3.4 AI Assistant (Interface)
- Natural language interface for location managers
- Answers questions like: "Why did churn increase at my location last month?"
- Uses tools: SQL queries, model inference, optimization engine
- Memory: remembers past analyses per location for trend detection
- Guardrails: PII masking, output validation, fallback on low confidence
- LGPD/GDPR compliant: full audit logging, no personal data sent to LLM

## 4. Data Requirements

### Input Data (simulated from domain knowledge)
- **Members:** id, join_date, cancel_date (null if active), plan_type, contract_source (direct/aggregator), location_id, demographic segment
- **Visits:** member_id, visit_date, visit_duration, visit_type
- **Transactions:** member_id, date, amount, type (enrollment, monthly, upgrade, discount)
- **Retention Actions:** member_id, action_date, action_type, cost
- **Locations:** id, region, capacity, monthly_operating_cost, manager_id

### Data Characteristics
- ~2M member records across 250 locations
- 18 months of history
- Seasonal patterns: January enrollment peak, February/March drop, June/July secondary peak
- ~15% of members via aggregators (Gympass, TotalPass, Wellhub)
- Data quality issues intentionally included: missing visit records, duplicated transactions, timezone inconsistencies

## 5. Success Metrics

| Metric | Baseline (heuristic) | Target (RetentionIQ) |
|--------|---------------------|---------------------|
| Churn prediction AUC-ROC | 0.72 (logistic regression on visit frequency) | > 0.85 |
| Budget allocation efficiency | Uniform (equal per location) | > 20% improvement in retained MRR per $ spent |
| Causal effect estimation | None (assumed all actions work equally) | CATE estimates with refutation tests passing |
| Manager adoption | N/A | Agent resolves > 70% of manager queries without escalation |
| Time to insight | 2-3 days (analyst manually queries) | < 30 seconds (agent response) |

## 6. Non-Functional Requirements

- **Latency:** API prediction endpoint < 200ms p95; Agent response < 15s p95
- **Throughput:** Support batch scoring of 2M members in < 30 minutes
- **Data freshness:** Features updated daily; models retrained weekly
- **Privacy:** LGPD-compliant; PII never sent to external LLM APIs; full audit trail
- **Monitoring:** Data drift detection with automated alerts; model performance tracked over time
- **Reliability:** Pipeline retries on failure; graceful degradation if model service is down

## 7. Out of Scope (v1)

- Real-time streaming pipeline (batch is sufficient for daily scoring)
- Mobile app interface (API-first; frontend is a separate concern)
- A/B testing infrastructure (planned for v2 — needed to close the causal loop)
- Multi-language agent support (English only for v1)
- Integration with CRM/ERP systems

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Causal estimates biased by unmeasured confounders | Wrong treatment recommendations | Sensitivity analysis (Rosenbaum bounds); document assumptions explicitly in DAG |
| Agent hallucination | Manager acts on false information | Guardrails with source attribution; confidence threshold; fallback to "I don't know" |
| Aggregator members have insufficient data for causal analysis | Can't estimate treatment effects for ~15% of base | Separate analysis track; acknowledge limitations transparently |
| Model drift due to seasonal patterns | Degraded predictions in peak/trough months | Seasonal features; monthly retrain; drift monitoring with auto-alerts |
