#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-ghcr.io/jcartu/glm52-exl3-lora@sha256:3014c71c1d216b8c9fb53326f3c6ffaa993a8145567c4a3513dc6c645ec60e5b}"
export MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
export ADAPTER_DIR="${ADAPTER_DIR:-$SCRIPT_DIR/adapter}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.cache/glm52-exl3-lora-v31}"
export PORT="${PORT:-8000}"
export BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,1,2,0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"
export DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-64}"
export ENABLE_MTP="${ENABLE_MTP:-1}"
export MTP_TOKENS="${MTP_TOKENS:-3}"
export MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-greedy}"
export GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
export LORA_NAME="${LORA_NAME:-adapter}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-glm52-exl3-lora}"

COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.yml}"
COMPOSE=(docker compose -f "$COMPOSE_FILE")
API_BASE="http://127.0.0.1:$PORT"

usage() {
  cat <<'EOF'
Usage: ./server.sh [start|stop|restart|logs|status|pull|load|unload]

Required for start/restart:
  MODEL_DIR      GLM-5.2 EXL3 model directory
  ADAPTER_DIR    BF16 rank-16 LoRA directory

Common overrides:
  IMAGE, CACHE_DIR, PORT, BIND_ADDRESS, CUDA_VISIBLE_DEVICES,
  GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, MAX_NUM_SEQS,
  MAX_NUM_BATCHED_TOKENS, DCP_KV_CACHE_INTERLEAVE_SIZE,
  ENABLE_MTP, MTP_TOKENS, MTP_DRAFT_SAMPLE_METHOD,
  GLM52_INDEX_TOPK_PATTERN, LORA_NAME, COMPOSE_PROJECT_NAME, COMPOSE_FILE

The adapter is mounted at /adapter but loaded dynamically. After the API is ready,
run `./server.sh load`; use `./server.sh unload` before replacing its files.
EOF
}

require_runtime() {
  command -v docker >/dev/null 2>&1 || {
    echo "docker is required" >&2
    exit 1
  }
  docker compose version >/dev/null
  [[ -f "$COMPOSE_FILE" ]] || {
    echo "Compose file not found: $COMPOSE_FILE" >&2
    exit 1
  }
}

require_inputs() {
  [[ -f "$MODEL_DIR/config.json" ]] || {
    echo "Model config not found: $MODEL_DIR/config.json" >&2
    exit 1
  }
  [[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
    echo "Model index not found: $MODEL_DIR/model.safetensors.index.json" >&2
    exit 1
  }
  [[ -f "$ADAPTER_DIR/adapter_config.json" ]] || {
    echo "Adapter config not found: $ADAPTER_DIR/adapter_config.json" >&2
    exit 1
  }
  compgen -G "$ADAPTER_DIR/*.safetensors" >/dev/null || {
    echo "No adapter safetensors found in: $ADAPTER_DIR" >&2
    exit 1
  }
  mkdir -p "$CACHE_DIR"
}

require_lora_name() {
  [[ "$LORA_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "LORA_NAME may contain only letters, numbers, dot, underscore, and hyphen" >&2
    exit 1
  }
}

action="${1:-start}"
case "$action" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

require_runtime

case "$action" in
  start)
    require_inputs
    docker pull "$IMAGE"
    "${COMPOSE[@]}" config --quiet
    "${COMPOSE[@]}" up -d --force-recreate
    echo "Starting on $API_BASE"
    echo "After /health is ready, load the mounted adapter with: $0 load"
    ;;
  stop)
    "${COMPOSE[@]}" down
    ;;
  restart)
    require_inputs
    docker pull "$IMAGE"
    "${COMPOSE[@]}" config --quiet
    "${COMPOSE[@]}" up -d --force-recreate
    echo "Restarting on $API_BASE"
    ;;
  logs)
    "${COMPOSE[@]}" logs --tail 100 -f glm52
    ;;
  status)
    "${COMPOSE[@]}" ps
    curl -fsS "$API_BASE/v1/models" || true
    printf '\n'
    ;;
  pull)
    docker pull "$IMAGE"
    ;;
  load)
    require_lora_name
    printf -v payload '{"lora_name":"%s","lora_path":"/adapter","load_inplace":false,"is_3d_lora_weight":false}' "$LORA_NAME"
    curl -fsS -X POST "$API_BASE/v1/load_lora_adapter" \
      -H 'Content-Type: application/json' \
      --data-binary "$payload"
    printf '\n'
    ;;
  unload)
    require_lora_name
    printf -v payload '{"lora_name":"%s"}' "$LORA_NAME"
    curl -fsS -X POST "$API_BASE/v1/unload_lora_adapter" \
      -H 'Content-Type: application/json' \
      --data-binary "$payload"
    printf '\n'
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
