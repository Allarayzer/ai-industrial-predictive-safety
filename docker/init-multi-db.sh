#!/bin/bash
# Create both the n8n and mlflow databases inside the shared Postgres
# container. Invoked by the postgres entrypoint on first run.
set -euo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"

for db in n8n mlflow; do
    echo "Creating database '$db' for user '$POSTGRES_USER'..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
        SELECT 'CREATE DATABASE $db'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
        GRANT ALL PRIVILEGES ON DATABASE $db TO $POSTGRES_USER;
EOSQL
done
