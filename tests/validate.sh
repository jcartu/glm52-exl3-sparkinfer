#!/usr/bin/env bash
# One-shot smoke validation of a running endpoint: health -> greedy inference -> tool calls.
set -uo pipefail
export PORT="${PORT:-8000}"; export MODEL="${MODEL:-GLM-5.2-EXL3-TR3-3.0bpw}"; H="http://127.0.0.1:$PORT"
fail=0
step(){ printf '%-42s' "$1"; }

step "health /v1/models"
curl -fsS --max-time 10 "$H/v1/models" >/dev/null && echo PASS || { echo FAIL; fail=1; }

step "greedy inference"
r=$(curl -fsS --max-time 180 "$H/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",\"temperature\":0,\"max_tokens\":2048,
  \"messages\":[{\"role\":\"user\",\"content\":\"Reply exactly: VALIDATE-OK\"}]}" \
  | python3 -c 'import sys,json;print((json.load(sys.stdin)["choices"][0]["message"].get("content") or "").strip())' 2>/dev/null)
[ "$r" = "VALIDATE-OK" ] && echo PASS || { echo "FAIL ($r)"; fail=1; }

step "tool calls (4 scenarios, non-streaming)"
python3 "$(dirname "$0")/tool_call_test.py" >/dev/null 2>&1 && echo PASS || { echo FAIL; fail=1; }

step "tool calls (streaming deltas)"
python3 "$(dirname "$0")/tool_call_stream_test.py" 2>/dev/null | grep -q 'PASS' && echo PASS || { echo FAIL; fail=1; }

[ "$fail" = 0 ] && echo "ALL PASS" || echo "FAILURES PRESENT"
exit $fail
