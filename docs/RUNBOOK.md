# Operational Runbook — RetentionIQ

## Daily Operations

### Pipeline Schedule
The Dagster pipeline runs daily at 06:00 UTC (03:00 BRT):
1. **Bronze ingestion** — Pull new data from source PostgreSQL
2. **Silver cleaning** — Apply business rules, dedup, standardize
3. **Gold aggregation** — Build member 360° view, location aggregates
4. **Feature materialization** — Push to Feast online store
5. **Churn scoring** — Batch score all active members
6. **Monitoring** — Run drift detection, generate Evidently reports

### Health Checks
- **API:** `GET /health` — returns 200 if all dependencies are reachable
- **Dagster:** Check run history at `http://dagster:3000/runs`
- **MLflow:** Verify model registry at `http://mlflow:5001/#/models`

## Alert Playbook

### Data Drift Detected
**Alert:** Evidently detects feature distribution shift (KS p < 0.05)
**Severity:** Warning
**Action:**
1. Check which features drifted in Evidently dashboard
2. Investigate root cause: new location onboarded? Seasonal change? Data quality issue?
3. If seasonal: expected, no action needed (document in runbook notes)
4. If data quality: fix upstream, re-run pipeline
5. If structural: trigger model retrain

### Model Performance Degradation
**Alert:** Churn model AUC-ROC drops > 5pp from baseline
**Severity:** High
**Action:**
1. Check monitoring dashboard for correlated data drift
2. Review recent data quality reports
3. If drift confirmed: trigger retrain with latest data
4. If no drift: investigate concept drift — member behavior may have changed
5. After retrain: compare new model vs old in MLflow, promote if better

### Pipeline Failure
**Alert:** Dagster run fails
**Severity:** Critical
**Action:**
1. Check Dagster UI for error details
2. Common causes: database connection timeout, schema change upstream, disk space
3. For transient errors: retry run
4. For schema changes: update Great Expectations suite, fix transformation code
5. Notify team if pipeline is down > 2 hours

### Agent Error Rate Spike
**Alert:** > 10% of agent queries result in fallback response
**Severity:** Medium
**Action:**
1. Check agent evaluation logs
2. Common causes: LLM API rate limit, tool execution timeout, new query pattern
3. Review recent queries that triggered fallback
4. If new query pattern: add to eval scenarios, update tools/prompts
5. If API issue: check provider status page

## Weekly Tasks

- [ ] Review model performance trends in MLflow
- [ ] Check Evidently weekly report for gradual drift
- [ ] Review agent evaluation metrics (task success rate, cost per query)
- [ ] Verify DVC data versions are consistent with pipeline runs
- [ ] Audit agent interaction logs for PII leakage (LGPD compliance)

## Monthly Tasks

- [ ] Full model retrain with expanded training window
- [ ] Re-estimate causal effects with latest data
- [ ] Re-run budget optimizer with updated treatment effects
- [ ] Review and update Great Expectations suites
- [ ] Update agent eval scenarios with new query patterns from production
- [ ] Generate monthly retention report for C-Level

## Disaster Recovery

### Database Corruption
1. Stop all services: `docker-compose down`
2. Restore from latest backup (Azure Backup or pg_dump)
3. Re-run pipeline from bronze layer: `dagster asset materialize --select "bronze_*"`

### Model Registry Lost
1. MLflow artifacts stored in mounted volume
2. If volume lost: retrain all models (configs are in Git, data is in DVC)
3. Re-register models in MLflow with correct tags

### Complete Environment Rebuild
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d    # Recreate
make setup              # Reinstall deps
python scripts/init_db.py  # Recreate schemas + agent_memory table
dvc pull                # Restore data (or: make data)
make pipeline           # Rebuild all assets
```

## Security Notes

- All credentials are managed via environment variables (see `.env.example`)
- Never commit `.env` files — they are in `.gitignore`
- Docker Compose uses `${VAR:-default}` substitution for local dev
- Production deployments must override `POSTGRES_PASSWORD` and API keys
- PII masking is enforced at the code level in `src/agents/guardrails.py`
- Agent memory connection strings read from `RETENTIONIQ_DB_URL` env var
