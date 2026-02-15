#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p .pids .logs

# Load .env if present (without overwriting existing exported env vars).
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env"
  set +a
fi

REDIS_URL="${REDIS_URL:-redis://localhost:6380/0}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5500}"
if [ -n "${JOB_DISCOVERY_PROVIDER:-}" ]; then
  :
elif [ -n "${SERPAPI_API_KEY:-}" ]; then
  JOB_DISCOVERY_PROVIDER="serpapi"
else
  JOB_DISCOVERY_PROVIDER="seed"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set. /generate and live discovery will fail until it is set in .env or exported."
fi

REDIS_PORT="$(REDIS_URL="$REDIS_URL" python3 - <<'PY'
import os
from urllib.parse import urlparse
u = urlparse(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
print(u.port or 6379)
PY
)"

if python3 - <<PY
import socket, sys
port=int(${REDIS_PORT})
s=socket.socket(); s.settimeout(0.2)
try:
  s.connect(("127.0.0.1", port))
  sys.exit(0)
except Exception:
  sys.exit(1)
finally:
  s.close()
PY
then
  :
else
  echo "Starting redis-server on port ${REDIS_PORT}..."
  redis-server --save "" --appendonly no --port "${REDIS_PORT}" --daemonize yes --pidfile ".pids/redis.pid" >".logs/redis.log" 2>&1 || true
fi

echo "Starting RQ worker..."
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
REDIS_URL="$REDIS_URL" \
nohup python3 worker.py >".logs/worker.log" 2>&1 & echo $! > .pids/worker.pid

echo "Starting API..."
REDIS_URL="$REDIS_URL" JOB_DISCOVERY_PROVIDER="$JOB_DISCOVERY_PROVIDER" \
nohup uvicorn api:app --host 0.0.0.0 --port "$API_PORT" >".logs/api.log" 2>&1 & echo $! > .pids/api.pid

echo "Starting UI server..."
nohup python3 -m http.server "$UI_PORT" >".logs/ui.log" 2>&1 & echo $! > .pids/ui.pid

echo ""
echo "UI:  http://localhost:${UI_PORT}/index.html"
echo "API: http://localhost:${API_PORT}"
echo "Redis: ${REDIS_URL}"
