#!/usr/bin/env bash
# One-stop launcher: sets up venv, installs deps, and runs server+GUI.
# Usage:
#   bash scripts/run.sh [all|server|gui]
# Env overrides:
#   TORCH_INDEX_URL   Index URL for CUDA torch wheels (optional)
#   HOST              Default 127.0.0.1
#   PORT              Default 8000
#   MODEL             Default nvidia/Nemotron-Nano-VL-12B-V2-FP8
#   EXTRA_ARGS        Extra flags for vLLM server
#   OPENAI_API_BASE   GUI base; defaults to http://$HOST:$PORT/v1 when running server
#   OPENAI_API_KEY    GUI key; default EMPTY
#   MODEL_ID          GUI model id; default same as MODEL
#   SKIP_INSTALL      If set to 1, skip dependency installation

set -euo pipefail

# Resolve repository root so the script can be run from anywhere
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

ACTION=${1:-all}

OS_NAME="$(uname -s || echo unknown)"
if [ "$OS_NAME" = "Darwin" ] && [ "${NO_MAC_GUI_ONLY:-0}" != "1" ]; then
  # vLLM/CUDA server doesn't run on macOS; default to GUI mode for convenience
  if [ "$ACTION" = "all" ]; then
    echo "[run] macOS detected; switching action 'all' -> 'gui' (server requires NVIDIA/CUDA)."
    ACTION="gui"
  elif [ "$ACTION" = "server" ];
  then
    echo "[run][error] 'server' mode is not supported on macOS (requires NVIDIA/CUDA). Use 'gui' and point to a remote server."
    exit 2
  fi
fi

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MODEL=${MODEL:-nvidia/Nemotron-Nano-VL-12B-V2-FP8}
EXTRA_ARGS=${EXTRA_ARGS:-}

PY_BIN="python3"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python"
fi

log() { printf "[run] %s\n" "$*"; }
err() { printf "[run][error] %s\n" "$*" >&2; }

activate_venv() {
  if [ ! -d .venv ]; then
    log "Creating virtual environment .venv"
    "$PY_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

install_torch() {
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    log "Installing torch from TORCH_INDEX_URL=$TORCH_INDEX_URL"
    pip install --index-url "$TORCH_INDEX_URL" torch -U
  else
    log "Installing torch (default index)"
    pip install torch -U
  fi
}

install_deps() {
  if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    log "Skipping dependency installation (SKIP_INSTALL=1)"
    return
  fi
  log "Upgrading pip/setuptools/wheel"
  python -m pip install --upgrade pip wheel setuptools
  if [ "$ACTION" = "all" ] || [ "$ACTION" = "server" ]; then
    log "Installing server deps"
    install_torch
    pip install -r "$REPO_ROOT/requirements-server.txt"
  fi
  if [ "$ACTION" = "all" ] || [ "$ACTION" = "gui" ]; then
    log "Installing GUI deps"
    pip install -r "$REPO_ROOT/requirements.txt"
  fi
}

SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && ps -p "$SERVER_PID" >/dev/null 2>&1; then
    log "Stopping server (pid $SERVER_PID)"
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_server_bg() {
  log "Launching vLLM OpenAI server on $HOST:$PORT"
  mkdir -p .logs
  # Run in background, redirect logs
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --trust-remote-code \
    --quantization modelopt \
    --host "$HOST" \
    --port "$PORT" \
    $EXTRA_ARGS \
    > .logs/server.log 2>&1 &
  SERVER_PID=$!
  log "Server PID: $SERVER_PID (logs: .logs/server.log)"
}

have_curl() { command -v curl >/dev/null 2>&1; }

wait_for_server() {
  local base="http://$HOST:$PORT/v1"
  local deadline=$((SECONDS + 240))
  log "Waiting for server to become ready at $base ..."
  while [ $SECONDS -lt $deadline ]; do
    if have_curl; then
      code=$(curl -s -o /dev/null -w "%{http_code}" "$base/models" || true)
      if [ "$code" = "200" ]; then
        log "Server is ready."
        return 0
      fi
    else
      # Python fallback
      "$PY_BIN" - "$base" <<'PY'
import sys, json, time
import urllib.request
url = sys.argv[1] + "/models"
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        print(r.status)
        sys.exit(0 if r.status==200 else 1)
except Exception:
    sys.exit(1)
PY
      if [ $? -eq 0 ]; then
        log "Server is ready."
        return 0
      fi
    fi
    sleep 2
  done
  err "Server did not become ready in time. See .logs/server.log"
  return 1
}

run_gui() {
  # If we started the server here, prefer that base; otherwise let env or default apply.
  if [ -n "$SERVER_PID" ]; then
    export OPENAI_API_BASE=${OPENAI_API_BASE:-http://$HOST:$PORT/v1}
    export MODEL_ID=${MODEL_ID:-$MODEL}
  else
    export OPENAI_API_BASE=${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}
    export MODEL_ID=${MODEL_ID:-$MODEL}
  fi
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

  log "Starting GUI (Streamlit)"
  log "  OPENAI_API_BASE=$OPENAI_API_BASE"
  log "  MODEL_ID=$MODEL_ID"
  streamlit run "$REPO_ROOT/app.py"
}

main() {
  case "$ACTION" in
    all)
      activate_venv
      install_deps
      start_server_bg
      wait_for_server
      run_gui
      ;;
    server)
      activate_venv
      install_deps
      log "Streaming server logs below (Ctrl+C to stop)"
      # Foreground server (no background), inherit stdout/stderr
      python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --trust-remote-code \
        --quantization modelopt \
        --host "$HOST" \
        --port "$PORT" \
        $EXTRA_ARGS
      ;;
    gui)
      activate_venv
      install_deps
      run_gui
      ;;
    *)
      err "Unknown action: $ACTION"
      err "Usage: bash scripts/run.sh [all|server|gui]"
      exit 1
      ;;
  esac
}

main "$@"
