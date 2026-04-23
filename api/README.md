# REST API
FastAPI service exposing the AI-CTA pipeline as REST endpoints,
following the design described in **Chapter 10.8** of the monograph.
## Endpoints
| Method | Path           | Purpose                              |
|--------|----------------|--------------------------------------|
| GET    | `/health`      | Liveness probe                       |
| GET    | `/version`     | Package version                      |
| POST   | `/score`       | Score a single sensor reading        |
| POST   | `/score-batch` | Score a batch (window) of readings   |
Interactive Swagger UI is available at `/docs` when the service is
running.
## Install and run
```bash
# Install with API extras
pip install -e ".[api]"
# Start the dev server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
## Example request
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"temperature": 52.3, "vibration": 0.34, "pressure": 1.02}'
```
Expected response shape:
```json
{
  "timestamp": "2026-04-22T18:42:11.394283",
  "anomaly_score": 0.42,
  "risk_score": 0.31,
  "risk_level": "Warning"
}
```
## Production notes
This service is a reference implementation suitable for prototyping
and evaluation. Production deployments should add:
- Authentication and authorization (e.g., API keys, OAuth2)
- Request rate limiting
- Structured logging and tracing
- Persistence of model artifacts (replace the synthetic startup
  pipeline with one loaded from MLflow or an artifact store)
- Horizontal scaling behind a reverse proxy
