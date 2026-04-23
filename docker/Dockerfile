# syntax=docker/dockerfile:1
# Reference Dockerfile for the AI-CTA REST API (monograph § 12.5).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Install build deps for scientific stack, then drop them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*
# Install Python dependencies first (layer cache friendly).
COPY pyproject.toml requirements.txt /app/
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2.0"
# Copy the source.
COPY src /app/src
COPY api /app/api
COPY README.md /app/
# Make the package importable.
ENV PYTHONPATH=/app/src
EXPOSE 8000
# Healthcheck via the service's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        urllib.request.urlopen('http://localhost:8000/health', timeout=3); sys.exit(0)" \
    || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
