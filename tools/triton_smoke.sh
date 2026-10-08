#!/usr/bin/env bash
# End-to-end test of the Triton deployable, on a GPU host.
#
# Covers what the unit tests structurally cannot: that the image starts, that
# `tritonserver` loads the python backend and the model, that the fla-on-CUDA assertion
# passes, that batching coalesces, and that the answers match the vLLM deployable.
#
# The latency assertion is the cold-start check. The first forward compiles the Gated
# DeltaNet kernels -- measured 54.2 s on a cold cache against ~40 ms steady state -- so a
# slow *second* request means warm-up did not cover that shape and a real caller will pay
# for it. initialize() sweeps three state lengths across both readout paths for this
# reason, and this is what proves the sweep worked.
#
# Usage: triton_smoke.sh [IMAGE] [PORT]
#
# With no IMAGE, the ECR registry is resolved from the caller's own account rather than
# hard-coded, so this script carries no account id. Override REPO/TAG/REGION, or pass a
# full image URI as the first argument.

set -uo pipefail

REGION="${REGION:-us-west-2}"
REPO="${REPO:-strands-decider-serving}"
TAG="${TAG:-v23-triton-onepass}"

if [ -n "${1:-}" ]; then
    IMAGE="$1"
else
    ACCOUNT="$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null)"
    if [ -z "$ACCOUNT" ] || [ "$ACCOUNT" = "None" ]; then
        echo "could not resolve the AWS account id, and no IMAGE argument was given." >&2
        echo "Either configure credentials or run: $0 <account>.dkr.ecr.<region>.amazonaws.com/$REPO:$TAG" >&2
        exit 2
    fi
    IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"
fi

PORT="${2:-8100}"
NAME="sd-triton-smoke"
CACHE="${TRITON_KERNEL_CACHE:-/opt/triton-cache}"
BASELINE="${BASELINE:-/opt/tctx/baseline_vllm.json}"
PARITY="${PARITY:-/opt/tctx/tools/parity_check.py}"

PASS=0; FAIL=0
ok()   { echo "  PASS: $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $*"; FAIL=$((FAIL+1)); }
info() { echo "  ..    $*"; }

cleanup() {
    echo "==> container log (last 40)"
    docker logs "$NAME" 2>&1 | tail -40
    docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> image: $IMAGE"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not present"; exit 2; }
docker rm -f "$NAME" >/dev/null 2>&1 || true

mkdir -p "$CACHE"

# SageMaker starts the image as `docker run <image> serve`, so the smoke test does too --
# testing the path that production uses rather than a convenient one.
echo "==> starting (SageMaker mode: argv 'serve', port 8080 inside)"
docker run -d --name "$NAME" --gpus all \
  -p "${PORT}:8080" \
  -v "${CACHE}:/opt/strands-decider/triton-cache" \
  -e SAGEMAKER_TRITON_DEFAULT_MODEL_NAME=decider \
  -e SD_ENGINE=merged \
  "$IMAGE" serve >/dev/null

BASE="http://localhost:${PORT}"

# ---- 1. it comes up at all
echo "==> waiting for readiness (up to 600s: pull is done, but kernels may compile)"
READY=0
START=$(date +%s)
for _ in $(seq 1 120); do
    if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
        bad "container exited during startup"; break
    fi
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "${BASE}/ping" 2>/dev/null || echo 000)
    if [ "$CODE" = "200" ]; then READY=1; break; fi
    sleep 5
done
ELAPSED=$(( $(date +%s) - START ))
if [ "$READY" = "1" ]; then ok "/ping 200 after ${ELAPSED}s"; else bad "never became ready (${ELAPSED}s)"; exit 1; fi

# ---- 2. the fla-on-CUDA assertion passed, i.e. no silent CPU fallback
if docker logs "$NAME" 2>&1 | grep -qi "device_platform.*not 'cuda'\|reference path"; then
    bad "fla fell back off CUDA -- the Gated DeltaNet layers are on the slow path"
else
    ok "no fla CPU-fallback warning (the silent-degradation trap)"
fi
if docker logs "$NAME" 2>&1 | grep -q "warm-up covered"; then
    ok "$(docker logs "$NAME" 2>&1 | grep -o 'warm-up covered.*' | tail -1)"
else
    info "no warm-up line in the log (check SD_WARMUP)"
fi

# ---- 3. Triton's own health, and that the model actually loaded
for probe in "v2/health/live" "v2/health/ready" "v2/models/decider/ready"; do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "${BASE}/${probe}" || echo 000)
    # SageMaker mode exposes 8080 only; Triton's native routes may not be mapped there.
    [ "$CODE" = "200" ] && ok "${probe} 200" || info "${probe} -> ${CODE} (not mapped in SageMaker mode)"
done

# ---- 4. a real inference through /invocations, all three primitives
# The System One body, and the KServe v2 envelope it has to travel in.
#
# SageMaker's /invocations on the Triton DLC is a thin proxy to Triton's v2 `infer`, so a
# bare System One body gets HTTP 500 "Unable to parse 'inputs'". shape is [1,1] because
# the model sets max_batch_size > 0 and Triton prepends the batch dimension.
SD_REQ='{"state":"Help! My payouts have been failing for 3 days!","questions":{
  "urgent":{"type":"noul","instructions":"Does this convey urgency?"},
  "team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"money","technical":"bugs","sales":"pricing"}},
  "frustration":{"type":"score","instructions":"How frustrated?","criteria":["calm","frustrated","depressed"]}}}'
REQ=$(python3 -c '
import json, sys
print(json.dumps({"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                              "datatype": "BYTES", "data": [sys.argv[1]]}]}))
' "$SD_REQ")

ENV_BODY=$(curl -s -m 120 -X POST "${BASE}/invocations" -H 'Content-Type: application/json' -d "$REQ")
echo "  envelope: $(echo "$ENV_BODY" | head -c 300)"
BODY=$(python3 -c '
import json, sys
env = json.loads(sys.argv[1])
if "error" in env and "outputs" not in env:
    print(json.dumps({"error": env["error"]})); raise SystemExit(0)
for o in env.get("outputs") or []:
    if o.get("name") == "RESPONSE_JSON":
        print(o["data"][0]); raise SystemExit(0)
print(json.dumps({"error": "no RESPONSE_JSON"}))
' "$ENV_BODY")
echo "  response: $(echo "$BODY" | head -c 400)"
if python3 - "$BODY" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
assert "error" not in d, d["error"]
a = d["answers"]
assert set(a) == {"urgent", "team", "frustration"}, sorted(a)
assert a["urgent"]["type"] == "noul" and 0.0 <= a["urgent"]["noul"] <= 1.0
assert a["team"]["choice"] in {"billing", "technical", "sales"}
assert abs(sum(a["team"]["probabilities"].values()) - 1.0) < 1e-3
assert 0.0 <= a["frustration"]["score"] <= 2.0
assert len(a["frustration"]["legend"]) == 3
assert 0.0 <= a["team"]["confidence"] <= 1.0
for k in ("model", "usage", "latency_ms"):
    assert k in d, k
print("  shape ok:", a["team"]["choice"], a["urgent"]["noul"], a["frustration"]["score"])
PY
then ok "structured probability output, all three primitives"; else bad "response shape"; fi

# ---- 5. second-request latency: the cold-start / warm-up check
T0=$(python3 -c 'import time;print(time.time())')
curl -s -m 120 -o /dev/null -X POST "${BASE}/invocations" -H 'Content-Type: application/json' -d "$REQ"
SECOND=$(python3 -c "import time;print(f'{time.time()-$T0:.2f}')")
if python3 -c "import sys;sys.exit(0 if $SECOND < 5.0 else 1)"; then
    ok "second request ${SECOND}s (<5s: kernels were compiled during warm-up, not here)"
else
    bad "second request ${SECOND}s -- warm-up missed this shape; a real caller pays for it"
fi

# ---- 6. a malformed payload is a caller error, not a server error
BAD_ENV=$(python3 -c '
import json
print(json.dumps({"inputs":[{"name":"REQUEST_JSON","shape":[1,1],"datatype":"BYTES",
                             "data":[json.dumps({"state":"x","questions":{}})]}]}))')
CODE=$(curl -s -o /tmp/badenv -w '%{http_code}' -m 30 -X POST "${BASE}/invocations" \
         -H 'Content-Type: application/json' -d "$BAD_ENV")
python3 -c '
import json,sys
env=json.load(open("/tmp/badenv"))
out=[o for o in (env.get("outputs") or []) if o.get("name")=="RESPONSE_JSON"]
open("/tmp/badbody","w").write(out[0]["data"][0] if out else json.dumps(env))
' 2>/dev/null || cp /tmp/badenv /tmp/badbody
if grep -q '"error"' /tmp/badbody 2>/dev/null; then
    ok "empty questions -> error body (HTTP $CODE), not a server fault"
else
    bad "empty questions: HTTP $CODE body=$(head -c 200 /tmp/badbody)"
fi
MALFORMED=$(python3 -c '
import json
print(json.dumps({"inputs":[{"name":"REQUEST_JSON","shape":[1,1],"datatype":"BYTES",
                             "data":["{not json"]}]}))')
# The envelope has to be unwrapped before looking for the error, exactly as above. Grepping
# the raw response does NOT work and reported a false failure for two builds: the backend's
# error travels as JSON *inside* a BYTES string, so the bytes on the wire read
#   "data":["{\"error\": {\"type\": \"invalid_request\", ...}}"]
# and `grep '"error"'` cannot match the escaped \"error\". The server was right the whole
# time; the assertion was reading the wrong layer.
MAL_ENV=$(curl -s -m 30 -X POST "${BASE}/invocations" \
            -H 'Content-Type: application/json' -d "$MALFORMED")
MAL_BODY=$(python3 -c '
import json, sys
env = json.loads(sys.argv[1])
for o in env.get("outputs") or []:
    if o.get("name") == "RESPONSE_JSON":
        print(o["data"][0]); raise SystemExit(0)
print(json.dumps(env))
' "$MAL_ENV" 2>/dev/null || printf '%s' "$MAL_ENV")
if python3 -c '
import json, sys
d = json.loads(sys.argv[1])
err = d.get("error")
assert err, f"no error key in {sorted(d)}"
# A caller error, specifically -- not a generic server fault.
msg = err if isinstance(err, str) else err.get("message", "")
assert "json" in msg.lower(), f"error does not name the JSON problem: {msg!r}"
' "$MAL_BODY" 2>/dev/null; then
    ok "malformed JSON -> error body naming the parse failure"
else
    bad "malformed JSON: body=$(printf '%s' "$MAL_BODY" | head -c 200)"
fi

# ---- 7. concurrency: does the dynamic batcher actually coalesce?
echo "==> 16 concurrent requests (dynamic batching)"
CONC_START=$(date +%s.%N)
for i in $(seq 1 16); do
    curl -s -m 180 -o "/tmp/conc_$i" -X POST "${BASE}/invocations" \
      -H 'Content-Type: application/json' -d "$REQ" &
done
wait
CONC_WALL=$(python3 -c "print(f'{$(date +%s.%N) - $CONC_START:.2f}')")
GOOD=$(grep -l 'RESPONSE_JSON' /tmp/conc_* 2>/dev/null | xargs -r grep -l 'answers' 2>/dev/null | wc -l | tr -d ' ')
rm -f /tmp/conc_*
if [ "$GOOD" = "16" ]; then
    ok "16/16 concurrent requests answered in ${CONC_WALL}s ($(python3 -c "print(f'{16/$CONC_WALL:.1f}')") req/s)"
else
    bad "only ${GOOD}/16 concurrent requests answered (wall ${CONC_WALL}s)"
fi

# ---- 8. parity against the vLLM deployable: ZERO DECISION FLIPS is the gate
if [ -f "$BASELINE" ] && [ -f "$PARITY" ]; then
    echo "==> parity vs $BASELINE (gate: zero decision flips)"
    if python3 "$PARITY" --base "$BASE" --path /invocations --triton --against "$BASELINE" \
         --repeats 2 2>&1 | tee /tmp/parity.out | tail -25; then
        ok "zero decision flips against the vLLM deployable"
    else
        if grep -q "DECISION FLIPS     : 0" /tmp/parity.out; then
            bad "no flips, but cases were missing or errored -- see above"
        else
            bad "DECISION FLIPS against the vLLM deployable"
        fi
    fi
else
    info "no baseline at $BASELINE; skipping parity"
fi

printf '\n==> %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
