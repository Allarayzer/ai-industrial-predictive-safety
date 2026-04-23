# Docker deployment
Reference container setup for the AI-CTA pipeline, following the
deployment guidance in **Chapter 12.5** of the monograph.
## What's included
| Service   | Port  | Purpose                                  |
|-----------|-------|------------------------------------------|
| api       | 8000  | REST API (FastAPI; `/score`, `/health`)  |
| n8n       | 5678  | Workflow orchestrator                    |
| postgres  | —     | Persistence for n8n state                |
| redis     | —     | Queue backend for n8n in queue mode      |
This is the minimal stack. Production deployments described in the
monograph also include MLflow (model lifecycle), Grafana (dashboards),
Prometheus (metrics), and InfluxDB (time-series data).
## Prerequisites
- Docker Engine 24+
- Docker Compose v2 (`docker compose ...`)
## Starting the stack
From the repository root:
```bash
docker compose -f docker/docker-compose.yml up --build
```
First-time build will take several minutes as the API image installs
the scientific Python stack.
## Verifying it works
```bash
# API healthcheck
curl http://localhost:8000/health
# n8n editor
open http://localhost:5678   # macOS; or visit in any browser
```
## Stopping
```bash
docker compose -f docker/docker-compose.yml down
```
To also remove persisted volumes (Postgres data, n8n workflows):
```bash
docker compose -f docker/docker-compose.yml down -v
```
## Production hardening
Before any non-development use:
- Replace the default Postgres password (`n8n_change_me`) with a
  managed secret.
- Enable n8n authentication (`N8N_BASIC_AUTH_ACTIVE=true`,
  `N8N_BASIC_AUTH_USER`, `N8N_BASIC_AUTH_PASSWORD`).
- Put the API behind a TLS-terminating reverse proxy (Caddy, Traefik,
  nginx).
- Restrict service-to-service traffic to a private Docker network and
  expose only the public ingress.
- Add observability: Prometheus exporters and a Grafana dashboard.
- Move model artifacts to an external store (MLflow + object storage)
  rather than rebuilding the API image to update them.
See **Chapter 19** of the monograph (cybersecurity, ethics, regulation)
for the full hardening checklist.
