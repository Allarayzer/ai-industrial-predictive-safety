# Docker deployment

Reference container setup for the AI-CTA pipeline, following the
deployment guidance in **Chapter 12.5** of the monograph.

## What's included

The stack ships all nine services described in § 12.5:

| Service    | Port | Purpose                                                        |
|------------|------|----------------------------------------------------------------|
| api        | 8000 | REST API (FastAPI; `/predict`, `/score`, `/score-batch`, `/health`) |
| postgres   | —    | Shared relational store (n8n state + MLflow backend)           |
| redis      | —    | Queue backend for n8n in queue mode                            |
| n8n        | 5678 | Low-code workflow orchestrator                                 |
| mlflow     | 5000 | Experiment tracking + model registry                           |
| prometheus | 9090 | Metrics scraping + alerting                                    |
| grafana    | 3000 | Observability dashboards (backed by Prometheus + InfluxDB)     |
| influxdb   | 8086 | Time-series store for telemetry                                |
| simulator  | —    | Industrial telemetry generator (feeds `/predict` for smoke tests) |

## Prerequisites

- Docker Engine 24+
- Docker Compose v2 (`docker compose ...`)
- **~8 GB RAM and ~5 GB disk** for the full stack. For lightweight
  development use the four-service subset.

## Starting the stack

From the repository root, bring up everything:

```bash
docker compose -f docker/docker-compose.yml up --build
```

First-time build takes several minutes while the API image installs
the scientific Python stack.

### Minimal four-service subset

If you only need the core API + orchestration for local hacking:

```bash
docker compose -f docker/docker-compose.yml up api postgres redis n8n
```

## Verifying it works

```bash
# API healthcheck
curl http://localhost:8000/health

# Canonical single-sample predict (book § 10.8)
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"temperature": 50.0, "vibration": 0.25, "pressure": 1.0}'

# n8n editor
open http://localhost:5678   # macOS; or any browser

# MLflow UI
open http://localhost:5000

# Grafana (admin / admin_change_me)
open http://localhost:3000

# Prometheus
open http://localhost:9090
```

## Stopping

```bash
docker compose -f docker/docker-compose.yml down
```

To also remove persisted volumes (Postgres, n8n, MLflow, InfluxDB, Grafana, Prometheus):

```bash
docker compose -f docker/docker-compose.yml down -v
```

## Production hardening

Before any non-development use:

- Replace all default passwords (`aicta_change_me`, `admin_change_me`,
  `aicta-dev-token-change-me`) with managed secrets (Vault, AWS SSM,
  Docker swarm secrets).
- Enable n8n authentication (`N8N_BASIC_AUTH_ACTIVE=true`,
  `N8N_BASIC_AUTH_USER`, `N8N_BASIC_AUTH_PASSWORD`).
- Put the API behind a TLS-terminating reverse proxy (Caddy, Traefik,
  nginx).
- Restrict service-to-service traffic to a private Docker network and
  expose only the public ingress.
- Replace the synthetic `simulator` container with a real SCADA/IIoT
  ingest process.
- Rotate the InfluxDB admin token and scope it via buckets.
- Back up Grafana dashboards and MLflow experiment databases.

See **Chapter 19** of the monograph (cybersecurity, ethics, regulation)
for the full hardening checklist.
