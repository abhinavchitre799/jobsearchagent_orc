#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

kill_pidfile() {
  local pidfile="$1"
  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile" || true)"
    if [ -n "$pid" ]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$pidfile" || true
  fi
}

kill_pidfile ".pids/ui.pid"
kill_pidfile ".pids/api.pid"
kill_pidfile ".pids/worker.pid"
kill_pidfile ".pids/redis.pid"

echo "Stopped dev processes (best effort)."

