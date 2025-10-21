#!/bin/sh
set -e

# If the credentials JSON is provided via env var, write it to a temp file
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ]; then
  printf "%s" "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/gcp-key.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
fi

# Run the FastAPI app
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}"
