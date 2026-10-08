#!/usr/bin/env bash
# Build the Triton image on the in-region amd64 GPU box and push it to ECR.
#
# Not from the Mac. The SageMaker Triton DLC base alone is 27.7 GB and this repo's Mac is
# arm64, so a cross-build would emulate every RUN step and then push tens of gigabytes over
# a home connection. The build-box README in ../strands-decider/LambdaGpuLaunchDemo/
# measured 106 s in-region against about 60 minutes from the laptop, and that was for a
# 4.7 GB base.
#
# This follows that pattern (S3 context sync -> SSM -> docker build --platform linux/amd64)
# but targets the existing GPU host rather than standing up a separate builder, because one
# already exists and the instance role can already push to ECR
# (AmazonEC2ContainerRegistryPowerUser).
#
# Usage:
#   deploy/build_on_box.sh [TAG]
#
# Env:
#   BOX_ID    instance to build on          (REQUIRED, no default)
#   REGION    AWS region                    (default us-west-2)
#   BUCKET    build-context bucket          (default hobson-v17-<acct>-<region>)
#   REPO      ECR repository                (default strands-decider-serving)
#   HF_TOKEN  passed as a BuildKit secret if set (never lands in an image layer -- but it
#             DOES land in SSM command history, see the warning below)
#   TRITON_IMAGE  base image override, passed through as a --build-arg
#
# TRITON_IMAGE matters more than a normal override: the base image's CUDA major version
# decides which SageMaker instance families the result can be hosted on. The 26.05 and
# 25.09 DLC tags are CUDA 13, and a CUDA 13 runtime will not start on SageMaker's ml.g5
# fleet -- it fails with CUDA error 803, "unsupported display driver / cuda driver
# combination", because those hosts' drivers predate CUDA 13. 25.04-py3 is CUDA 12 and
# runs on both g5 (A10G) and g6 (L4), which is the difference between an endpoint that
# can be placed and one that cannot. See the build section of README.md.

set -euo pipefail

TAG="${1:-v21-triton}"
REGION="${REGION:-us-west-2}"
# Required rather than defaulted: a hard-coded instance id is one account's infrastructure
# baked into a script, and a stale one sends the build at whatever now holds that id.
BOX_ID="${BOX_ID:-}"
if [ -z "$BOX_ID" ]; then
    cat >&2 <<USAGE
BOX_ID is required: the in-region amd64 GPU instance to build on.

    BOX_ID=i-0123456789abcdef0 $0 ${TAG}

It needs docker, the SSM agent, ~200 GB free on / and an instance role that can push to
ECR (AmazonEC2ContainerRegistryPowerUser). To list candidates:

    aws ec2 describe-instances --region ${REGION} \\
      --filters Name=instance-state-name,Values=running \\
      --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output text
USAGE
    exit 2
fi
REPO="${REPO:-strands-decider-serving}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TRITON_IMAGE="${TRITON_IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:25.04-py3}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"
BUCKET="${BUCKET:-hobson-v17-${ACCOUNT}-${REGION}}"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
DLC_REGISTRY="763104351884.dkr.ecr.${REGION}.amazonaws.com"
PREFIX="build/${REPO}-${TAG}"

echo "==> context  : $HERE"
echo "==> bucket   : s3://${BUCKET}/${PREFIX}/"
echo "==> image    : ${REGISTRY}/${REPO}:${TAG}"
echo "==> base     : ${TRITON_IMAGE}"
echo "==> builder  : ${BOX_ID} (${REGION})"

# Only what the Dockerfile actually reads. Keeping this explicit rather than syncing the
# whole tree avoids shipping the venv, caches and any local scratch, and makes it obvious
# what the image is built from.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/deploy" "$STAGE/serving" "$STAGE/src" "$STAGE/model_repository"
cp "$HERE/deploy/Dockerfile.triton"        "$STAGE/deploy/"
cp "$HERE/serving/merge_lora.py"           "$STAGE/serving/"
cp -R "$HERE/src/strands_decider"          "$STAGE/src/"
cp -R "$HERE/src/decider_triton"           "$STAGE/src/"
cp -R "$HERE/model_repository/."           "$STAGE/model_repository/"
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
# AppleDouble files. A stray `._decider` next to `decider/` makes Triton try to load it as
# a second model, which fails the whole server start. `aws s3 sync` copies them happily,
# so they are removed here AND the Dockerfile deletes any that still arrive.
find "$STAGE" -name "._*" -delete 2>/dev/null || true
find "$STAGE" -name ".DS_Store" -delete 2>/dev/null || true

echo "==> syncing context ($(du -sh "$STAGE" | cut -f1))"
aws s3 sync "$STAGE" "s3://${BUCKET}/${PREFIX}/" --delete --only-show-errors --region "$REGION"

aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null

# WARNING, and it is not what the BuildKit-secret wording suggests. The token is a
# BuildKit secret inside the build, so it never lands in an image layer -- but to get it
# to the box it is interpolated into the remote script below, which is base64'd and sent
# as an SSM `send-command` parameter. SSM RETAINS command parameters in command history
# (and shows them in the console), so anyone with ssm:GetCommandInvocation or
# ssm:ListCommands in this account can read the token back out. Base64 is not encryption.
#
# So: only pass HF_TOKEN when the checkpoint repo actually requires one, use a short-lived
# read-only token, and rotate it after the build. A proper fix is to put the token in
# Secrets Manager or an SSM SecureString and have the remote script fetch it by name, so
# only the name travels through command history.
SECRET_ARG=""
if [ -n "${HF_TOKEN:-}" ]; then
    echo "==> HF_TOKEN present; passing it as a BuildKit secret"
    echo "    NOTE: it also enters SSM command history on this account. Rotate it after."
    SECRET_ARG='--secret id=hf_token,src=/tmp/ctx/hf_token'
fi

# Heredoc is UNQUOTED, so it expands here and the interpolated values above land in the
# remote script. Two consequences, both of which have bitten:
#   * `$` that must survive to the box is escaped (\$(date +%s) below).
#   * Backticks in the comment text would run as command substitution ON THIS MACHINE.
#     They did: a comment mentioning causal-conv1d printed "causal-conv1d: command not
#     found" and one mentioning docker pull ran a real `docker pull` with no argument.
#     Harmless noise, but it masks real errors, so the comments here use plain quotes.
REMOTE_SCRIPT=$(cat <<EOS
set -euxo pipefail
export AWS_DEFAULT_REGION=${REGION}
# Deliberately NOT pruning the build cache: causal-conv1d compiles its CUDA extension
# for ~19 minutes and caching it is the difference between a 2-minute rebuild and a
# 25-minute one. On a 200 GB root the cache and the images are in direct competition, so
# prune by hand when it gets tight rather than throwing 19 minutes away every build.
df -h / | tail -1

rm -rf /tmp/ctx && mkdir -p /tmp/ctx
aws s3 sync s3://${BUCKET}/${PREFIX}/ /tmp/ctx/ --only-show-errors
$( [ -n "${HF_TOKEN:-}" ] && echo "printf '%s' '${HF_TOKEN}' > /tmp/ctx/hf_token" )

aws ecr get-login-password | docker login --username AWS --password-stdin ${DLC_REGISTRY} >/dev/null
aws ecr get-login-password | docker login --username AWS --password-stdin ${REGISTRY} >/dev/null

# Pushes directly (--push) rather than build-then-push. MEASURED: build-then-push
# exhausted this 194 GB root three times, always at the same place: BuildKit
# finishing the image and then *unpacking* it into the local containerd snapshotter:
#     failed to extract layer ...libcupti_static.a: no space left on device
# The arithmetic is against it: ~45 GB OS + 27.7 GB DLC base + ~25 GB unique layers +
# ~40 GB build cache on a 194 GB disk. Pushing direct exports to the registry and skips
# the local unpack entirely, and the image has to be in ECR for SageMaker anyway -- so this
# is the right order, not a workaround. To run it locally afterwards: prune the build
# cache, then "docker pull".
#
# The output line is load-bearing and took three attempts to get right. SageMaker accepts
# ONLY application/vnd.docker.distribution.manifest.v2+json, and BuildKit's push produces
# OCI media types by default. Measured, in order:
#
#   plain --push                        -> application/vnd.oci.image.index.v1+json
#                                          ValidationException: Unsupported manifest media type
#   --push --provenance=false --sbom=false
#                                       -> application/vnd.oci.image.manifest.v1+json
#                                          ValidationException: Unsupported manifest media type
#   --output ...,oci-mediatypes=false   -> application/vnd.docker.distribution.manifest.v2+json
#
# So all three parts are needed: the attestation flags stop BuildKit emitting an INDEX
# (attestations force one), and oci-mediatypes=false switches the manifest itself to the
# Docker schema. Re-tagging and re-pushing does NOT fix a bad manifest -- docker pushes the
# same manifest it already has, with the same digest.
START=\$(date +%s)
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  -f /tmp/ctx/deploy/Dockerfile.triton \
  --build-arg TRITON_IMAGE=${TRITON_IMAGE} \
  ${SECRET_ARG} \
  --output type=image,name=${REGISTRY}/${REPO}:${TAG},push=true,oci-mediatypes=false \
  --provenance=false \
  --sbom=false \
  /tmp/ctx
DONE=\$(date +%s)

rm -f /tmp/ctx/hf_token
echo "build+push: \$((DONE-START))s"
# No docker image inspect here: a direct push never materialises the image locally,
# so ask ECR for the size instead.
aws ecr describe-images --repository-name ${REPO} --image-ids imageTag=${TAG} \
  --query 'imageDetails[0].imageSizeInBytes' --output text
df -h / | tail -1
EOS
)

echo "==> building on ${BOX_ID} (this takes a while; the base is 27.7 GB)"

# The script is base64'd and run with an explicit `bash`, not sent as shell text.
# AWS-RunShellScript executes its commands with /bin/sh, which on this Ubuntu host is
# dash, and dash rejects the very first line:
#     set: Illegal option -o pipefail
# Base64 also means nothing in the script body -- quotes, heredocs, newlines -- has to
# survive a second round of shell and JSON quoting.
PAYLOAD=$(printf '%s' "$REMOTE_SCRIPT" | base64 | tr -d '\n')
RUNNER="set -e; printf '%s' '$PAYLOAD' | base64 -d > /tmp/sd-build.sh; bash /tmp/sd-build.sh"

CMD_ID=$(aws ssm send-command \
  --instance-ids "$BOX_ID" --document-name AWS-RunShellScript --region "$REGION" \
  --timeout-seconds 120 \
  --parameters "$(python3 -c '
import json,sys
print(json.dumps({"commands":[sys.stdin.read()],"executionTimeout":["7200"]}))' <<< "$RUNNER")" \
  --query Command.CommandId --output text)

echo "==> ssm command: $CMD_ID"
while true; do
    STATUS=$(aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$BOX_ID" \
      --region "$REGION" --query Status --output text 2>/dev/null || echo Pending)
    case "$STATUS" in Pending|InProgress|Delayed) printf '.'; sleep 15 ;; *) echo; break ;; esac
done

aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$BOX_ID" \
  --region "$REGION" --query StandardOutputContent --output text | tail -30
ERR=$(aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$BOX_ID" \
  --region "$REGION" --query StandardErrorContent --output text)
[ -n "$ERR" ] && [ "$ERR" != "None" ] && { echo "--- stderr (last 40) ---" >&2; echo "$ERR" | tail -40 >&2; }

echo "[status: $STATUS]"
[ "$STATUS" = "Success" ] || exit 1
echo "==> pushed ${REGISTRY}/${REPO}:${TAG}"
