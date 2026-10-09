#!/usr/bin/env python3
"""Create the SageMaker real-time endpoint for the Decider, with autoscaling.

boto3, not the SageMaker Python SDK. The SDK is mid-churn: v3 is a split meta-package
(`sagemaker-core` / `-train` / `-serve` / `-mlops`) and the only deployment snippet in
`serving/README.md` is written against v1's `sagemaker.model.Model`. boto3's
`CreateModel` / `CreateEndpointConfig` / `CreateEndpoint` have been stable for years and
are what the SDK calls anyway, so the scaffolding does not rot.

Everything here is idempotent-ish: existing models and endpoint configs are reused unless
`--replace` is given, and an existing endpoint is updated rather than recreated.

    deploy/create_endpoint.py --image <acct>.dkr.ecr.us-west-2.amazonaws.com/strands-decider-serving:v23-triton-onepass
    deploy/create_endpoint.py --delete

This HAS been run against a live account and does work: it built the `strands-decider-g6`
endpoint on `ml.g6.xlarge`, which reached `InService` and served traffic. Two things learned
by doing it, both already fixed below: the endpoint config name must hash its contents or a
new image silently never deploys (see `cfg_fingerprint`), and `ensure_endpoint` has to tell
the three endpoint states apart because SageMaker rejects two of them with the same error
code. It creates billable resources -- ~$27/day for one `ml.g6.xlarge`, charged whether the
endpoint is used or not -- so pair every run with `--delete` when you are finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

# ---- defaults, each with a reason ------------------------------------------------------

# ml.g6.xlarge (L4, 24 GB, sm89). This is the only family this image has been observed
# serving on, and every measured number in the repo comes from an L4.
#
# ml.g5.xlarge (A10G, sm86) looks like the obvious fallback -- it has quota 4 here and had
# capacity on both days we tried -- and it does not work. MEASURED, twice:
#
#   * the CUDA 13 DLC (26.05) on a g5 host with kernel driver 535.309.01 entered CUDA
#     forward-compatibility mode and died with error 803.
#   * the CUDA 12 DLC (25.04) on a g5 host with driver 470.256.02 refused outright --
#     "built for NVIDIA Driver Release 575.51 or later ... compatibility mode is
#     UNAVAILABLE" -- after which Triton started with no GPU visible, left the model
#     "UNAVAILABLE: instance group decider_0 ... has kind KIND_GPU but no GPUs are
#     available", never passed /ping, and the endpoint failed 30 minutes later with
#     "did not pass the ping health check".
#
# So the g5 fleet carries drivers (470.x, 535.x) too old for any current Triton DLC, and
# which host you get is a lottery. Rebuilding on CUDA 12 does NOT unlock it.
#
# When Ada capacity is short -- six consecutive InsufficientInstanceCapacity failures across
# ml.g6.xlarge (x2), ml.g6e.xlarge, ml.g6.2xlarge and ml.g6.4xlarge, ~31 minutes each -- use
# `--instance-pools`, which asks SageMaker to fall back across up to five types within ONE
# CreateEndpoint. That supersedes the old advice here (launch several endpoints under
# different --name values and keep whichever lands), which worked but needed a human
# watching three deploys and left two to tear down. Do not fall back to g5: the drivers on
# that fleet are too old for any current Triton DLC, measured twice, see above.
DEFAULT_INSTANCE = "ml.g6.xlarge"

# ---- going bigger: ml.g6e.xlarge, measured ---------------------------------------------
#
#   instance        GPU    GPU mem   bandwidth   vCPU   $/hr (us-west-2 HOSTING)
#   ml.g6.xlarge    L4      22.9 GB   300 GB/s     4     1.1267  <- default; the README's
#                                                                   numbers are all this
#   ml.g6.2xlarge   L4      22.9 GB   300 GB/s     8     1.2220
#   ml.g6e.xlarge   L40S    45.8 GB   864 GB/s     4     2.6054
#   ml.g6e.2xlarge  L40S    45.8 GB   864 GB/s     8     2.8026
#
# ml.g6e.xlarge runs this image with NO rebuild -- the L40S is sm89 and Dockerfile.triton
# already builds for "8.0;8.6;8.9" -- and it is a large win under load. Measured through
# two live endpoints, same harness, same in-region load generator, 7 questions, distinct
# tickets (`tools/bench_tickets.py --tickets distinct`):
#
#            concurrency 1          concurrency 32 (saturation)
#            dec/s   server p50     dec/s   server p50   tickets/s
#   g6  (L4)  63.4     102.5 ms      99.0     620 ms       14.15
#   g6e (L40S) 66.1      95.7 ms     269.8     209 ms       38.55
#            +4%       -7%          +173%     -66%         +172%
#
# READ THOSE TWO COLUMNS TOGETHER, because they look contradictory and are not. At
# concurrency 1 the bigger card buys ~4%, because a single small pass is bound by the CPU
# issuing ~5,676 kernel launches -- the ~45 ms floor, which the GPU cannot help with. Under
# load the dynamic batcher fills each pass with up to 8 requests x 7 questions = 56 rows, so
# the pass is carrying thousands of tokens and has crossed out of the floor regime into the
# marginal one (tokens x cost-per-token).
#
# CORRECTION, because the obvious reading of that 2.7x is wrong and it mattered. It is
# tempting to call a filled pass "memory-bandwidth-bound" because the L40S/L4 bandwidth
# ratio is 2.88x and the measurement is ~2.9x. That inference does not hold:
#
#  * The card-swap CANNOT discriminate. L40S/L4 is 2.88x on bandwidth (864/300 GB/s) and
#    2.99x on dense BF16 tensor-core throughput (362/121 TFLOPS). Both hypotheses predict
#    the same ~2.9x, so the experiment distinguishes nothing.
#  * Weight streaming is NOT the cost. 2B parameters at 2 bytes is 4.0 GB per forward,
#    which at 864 GB/s is 4.6 ms -- 2.3% of the MEASURED 203.6 ms 56-row pass. A pass that
#    is 97.7% something-else cannot be bound by weight bandwidth, and that is also why
#    batching is not nearly free here (see config.pbtxt's max_batch_size note: pass time is
#    linear in rows at ~3.6 ms/row with no measurable fixed term).
#  * What it actually is: a 56-row pass pushes ~4,808 input tokens in 203.6 ms = ~23,600
#    tok/s, which for a 2B model is ~94 TFLOPS, or ~26% of the L40S's dense BF16 peak. So
#    the pass is arithmetic-shaped but running at about a quarter of the card's peak -- the
#    gap is kernel efficiency in the eager Gated DeltaNet path, not a bandwidth wall.
#
# The practical consequence: the remaining headroom on this card is ~4x in principle and it
# is reachable only through better kernels (SD_FUSE_LAYERS, CUDA graphs, a compiled torso),
# NOT through batch geometry, which is already finished.
#
# So "this model is dispatch-bound" is true of ONE request and false of a saturated server,
# and which one you are measuring decides the instance you should buy. The same fact explains
# why `instance_group count: 2` lost on the L4 (see config.pbtxt): under load that card is
# genuinely saturated, not merely busy, so a second process found no idle GPU to overlap into
# and only took CPU from the first.
#
# Cost per unit of work therefore FAVOURS g6e at full load, despite 2.31x the hourly rate:
#   g6  : 14.15 tickets/s -> ~50,900 tickets/hr  -> $0.0221 per 1,000 tickets
#   g6e : 38.55 tickets/s -> ~138,800 tickets/hr -> $0.0188 per 1,000 tickets  (-15%)
# It is worse value only if your traffic is thin enough that you never leave concurrency 1,
# where you would be paying 2.31x for 4%.
#
# TWO CAVEATS, both honest limits of the measurement above:
#  * The g6e row is a LOWER BOUND. Its server-side p50 is only 209 ms, i.e. the server was
#    not deeply queued, while the implied end-to-end latency was far higher -- so the 4-vCPU
#    load generator, not the endpoint, is what 269.8 decisions/s measures. The g6 row is a
#    real ceiling (p50 620 ms is the server queueing). A fatter load generator would raise
#    the g6e number and not the g6 one.
#  * 1-2 requests out of ~770 errored in the g6e c=16 and c=32 cells. Not enough to move the
#    throughput figure, and not diagnosed.
#
# Also note the prices above are the SageMaker HOSTING rates. The EC2 on-demand rate for
# g6e.xlarge is ~$1.86/hr; quoting that one under-budgets an endpoint by about 40%.
#
# CHECK YOUR QUOTA FIRST, because it is per instance type and is the thing most likely to
# bite. In the account this was built in: ml.g6.xlarge for endpoint usage = 4, but
# ml.g6e.xlarge = 1 -- so on g6e.xlarge autoscaling has nowhere to go and the 2.7x has to be
# enough on its own. `resolve_max_capacity` below reads the real quota rather than trusting
# DEFAULT_MAX_CAPACITY.

# A warm floor of 1, deliberately. Inference Components can scale a model to zero, but GPU
# cold start here is the image pull plus weight load plus the Gated DeltaNet kernel compile
# -- measured 78.8 s for a fresh 23.5 GB pull, and the Triton image is larger. The brief
# called this out and it is why scale-to-zero is not the default.
DEFAULT_MIN_CAPACITY = 1
# 4 is the ml.g6.xlarge endpoint-usage quota in the account this was built in. It is a
# DEFAULT, not a fact about your account or about any other instance type -- see
# `resolve_max_capacity`, which checks the real quota before autoscaling is configured.
DEFAULT_MAX_CAPACITY = 4

# `/ping` returns 503 for the whole pull + load + warm-up window, and SageMaker kills the
# container if it does not pass before this expires. Generous on purpose.
STARTUP_HEALTH_CHECK_TIMEOUT = 1800
MODEL_DATA_DOWNLOAD_TIMEOUT = 1800

# How long SageMaker may spend trying to place instances across ALL instance pools before
# giving up. Only valid alongside `InstancePools`.
#
# 900 s rather than the 3600 s maximum, because the thing being bought here is a FAST
# failure. Six consecutive `InsufficientInstanceCapacity` deploys cost ~31 minutes each
# before surfacing -- over three hours of wall clock for zero information. Capacity that
# has not appeared in 15 minutes across five instance types is not about to, and a deploy
# that fails in 15 minutes can be retried or re-planned inside the same working session.
# Valid range is 300-3600.
PROVISION_TIMEOUT = 900

# Instance pools, the fix for `InsufficientInstanceCapacity` costing half-hours.
#
# SageMaker accepts an ordered list of up to 5 instance types per production variant and
# falls back automatically when the higher-priority one has no capacity, so one
# CreateEndpoint attempt covers every type instead of one. MEASURED: one pooled attempt
# landed an ml.g6e.4xlarge immediately after three serial single-type attempts had each
# failed after ~31 minutes.
#
# Priority 1 is the HIGHEST and is tried first; scale-IN releases the lowest-priority
# (fallback) instances first, so the preferred type is kept as long as possible.
MAX_INSTANCE_POOLS = 5


def _tags(extra: dict | None = None) -> list[dict]:
    tags = {"Project": "strands-decider", "ManagedBy": "decider-server/deploy"}
    tags.update(extra or {})
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def parse_instance_pools(spec: str) -> list[dict]:
    """`"ml.g6e.4xlarge,ml.g6e.2xlarge"` -> ordered `InstancePools`, priority 1 first.

    Priority is taken from POSITION rather than asked for separately: the only thing
    priority expresses is "try these in this order", and a hand-written priority column is
    a second place for the order to disagree with itself.

    Duplicates are rejected rather than de-duplicated. A repeated type does not widen the
    search at all -- it burns one of the five pool slots on a type that already failed --
    and the most likely way to write one is a typo in a longer list, where silently
    accepting it would leave the operator believing they had five fallbacks instead of four.
    """
    types = [t.strip() for t in spec.split(",") if t.strip()]
    if not types:
        raise SystemExit("--instance-pools was given but lists no instance types")
    if len(types) > MAX_INSTANCE_POOLS:
        raise SystemExit(
            f"--instance-pools takes at most {MAX_INSTANCE_POOLS} instance types "
            f"(SageMaker's limit); got {len(types)}: {', '.join(types)}")
    duplicates = sorted({t for t in types if types.count(t) > 1})
    if duplicates:
        raise SystemExit(
            f"--instance-pools repeats {', '.join(duplicates)}. A duplicate pool adds no "
            "fallback and wastes one of five slots; list each type once, in priority order.")
    return [{"InstanceType": t, "Priority": i} for i, t in enumerate(types, start=1)]


def instance_pools_supported() -> bool:
    """Whether the INSTALLED botocore knows about `InstancePools`.

    Checked against the service model rather than a version string, because the version
    that introduced it is a fact about botocore's release history and this is a fact about
    the bytes on this machine.

    Worth checking at all because of the shape of the failure. botocore validates
    parameters against its bundled service model, so an older botocore rejects the request
    client-side with a `ParamValidationError` naming an "unknown parameter" -- which reads
    like the feature does not exist in SageMaker rather than like the SDK is stale, and is
    the wrong thing to conclude. Measured here: botocore 1.42.97 does not have it, 1.43.x
    does.
    """
    try:
        import botocore.session

        variant = (botocore.session.get_session()
                   .get_service_model("sagemaker")
                   .shape_for("ProductionVariant"))
        return "InstancePools" in variant.members
    except Exception:
        # Never block a deploy on a failed introspection -- fall through to letting the
        # real API call be the judge.
        return True


def describe_placement(sm, name: str) -> str:
    """Which instance type an endpoint using pools actually landed on.

    The point of pools is that you do NOT know in advance, so the deploy has to report it;
    otherwise a benchmark number gets attributed to the type that was asked for rather than
    the one that served it. `DescribeEndpoint` returns per-pool counts once provisioned.
    """
    try:
        variants = sm.describe_endpoint(EndpointName=name).get("ProductionVariants") or []
    except ClientError as exc:
        return f"(could not read placement: {exc.response['Error']['Code']})"
    parts = []
    for v in variants:
        pools = v.get("InstancePools") or []
        if pools:
            landed = ", ".join(
                f"{p.get('InstanceType')} x{p.get('CurrentInstanceCount', '?')}"
                for p in pools if p.get("CurrentInstanceCount"))
            parts.append(f"{v['VariantName']}: {landed or 'nothing placed yet'}")
        else:
            parts.append(f"{v['VariantName']}: {v.get('CurrentInstanceType') or '?'} "
                         f"x{v.get('CurrentInstanceCount', '?')}")
    return "; ".join(parts) or "(no variants reported)"


def ensure_model(sm, name: str, image: str, role: str, env: dict, replace: bool,
                 model_data_url: str = "") -> str:
    try:
        sm.describe_model(ModelName=name)
        if not replace:
            print(f"[model] {name} exists; reusing (--replace to recreate)")
            return name
        print(f"[model] deleting {name}")
        sm.delete_model(ModelName=name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("ValidationException", "ResourceNotFound"):
            raise

    print(f"[model] creating {name}")
    container = {"Image": image, "Environment": env}
    if model_data_url:
        # A MODEL REPOSITORY OVERLAY, not weights.
        #
        # Normally there is no ModelDataUrl at all: the weights are baked into the image, so
        # there is no 4.6 GB S3 download on every scale-out.
        #
        # What this is for is tuning `config.pbtxt` without a ~25-minute image rebuild.
        # SageMaker untars the archive over /opt/ml/model, which is exactly where
        # Dockerfile.triton puts the model repository (the DLC hard-codes
        # SAGEMAKER_SINGLE_MODEL_REPO=/opt/ml/model/), so the archive's `decider/` replaces
        # the image's.
        #
        # THE TRAP, because it cost a deploy: it replaces the model repository, which means
        # it replaces `decider/1/model.py` too -- so the archive must contain a model.py,
        # and that model.py then runs against whatever `strands_decider` the IMAGE was built
        # with. A newer model.py passing a newer kwarg dies with
        #   TypeError: load_merged_engine() got an unexpected keyword argument
        # and the endpoint fails on the ping health check ~31 minutes later. This is why
        # `model.py::_engine_kwargs` filters optional kwargs against the real signature.
        # Build the archive with `tools/make_overlay.py`, which checks both of these.
        container["ModelDataUrl"] = model_data_url
        print(f"[model] model repository overlay: {model_data_url}")
    sm.create_model(
        ModelName=name,
        ExecutionRoleArn=role,
        PrimaryContainer=container,
        Tags=_tags(),
    )
    return name


def production_variant(model_name: str, instance: str, variant: str,
                       pools: list[dict] | None) -> dict:
    """The variant body, with `InstancePools` REPLACING `InstanceType` when pools are used.

    Replacing, not accompanying: the API treats them as alternatives ("you replace the
    InstanceType parameter in your production variant with an InstancePools list"), and
    sending both is a validation error rather than a helpful hint about which to prefer.
    Built as a separate pure function so a test can assert that, without an account.
    """
    body = {
        "VariantName": variant,
        "ModelName": model_name,
        "InitialInstanceCount": DEFAULT_MIN_CAPACITY,
        # Weight matters only once there is a second variant for an A/B.
        "InitialVariantWeight": 1.0,
        "ContainerStartupHealthCheckTimeoutInSeconds": STARTUP_HEALTH_CHECK_TIMEOUT,
        "ModelDataDownloadTimeoutInSeconds": MODEL_DATA_DOWNLOAD_TIMEOUT,
    }
    if pools:
        body["InstancePools"] = pools
        body["VariantInstanceProvisionTimeoutInSeconds"] = PROVISION_TIMEOUT
    else:
        body["InstanceType"] = instance
    return body


def ensure_endpoint_config(sm, name: str, model_name: str, instance: str,
                           variant: str, replace: bool,
                           pools: list[dict] | None = None) -> str:
    try:
        sm.describe_endpoint_config(EndpointConfigName=name)
        if not replace:
            print(f"[config] {name} exists; reusing")
            return name
        print(f"[config] deleting {name}")
        sm.delete_endpoint_config(EndpointConfigName=name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("ValidationException", "ResourceNotFound"):
            raise

    if pools:
        shown = " -> ".join(f"{p['InstanceType']}(p{p['Priority']})" for p in pools)
        print(f"[config] creating {name} (pools: {shown}, "
              f"provision timeout {PROVISION_TIMEOUT}s)")
    else:
        print(f"[config] creating {name} ({instance})")
    sm.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[production_variant(model_name, instance, variant, pools)],
        Tags=_tags(),
    )
    return name


def ensure_endpoint(sm, name: str, config: str) -> None:
    """Create the endpoint, or leave it alone if it is already serving this config.

    The three cases have to be told apart before acting, because SageMaker rejects two of
    the four possible API calls with the SAME error code (ValidationException), so a
    try/except on the code alone cannot route them:

      * endpoint absent                -> CreateEndpoint
      * present, already on `config`   -> do nothing. UpdateEndpoint raises
                                          "Cannot update endpoint with the currently in use
                                          endpoint configuration", and falling through to
                                          CreateEndpoint then raises "Cannot create already
                                          existing endpoint". That is what an earlier version
                                          of this function did: re-running it to add
                                          autoscaling to a healthy endpoint crashed instead.
      * present, on a different config -> UpdateEndpoint
      * present but Creating/Updating  -> refuse. SageMaker returns "Cannot update
                                          in-progress endpoint"; say so plainly rather than
                                          turning it into a confusing create attempt.

    CAVEAT ON THE `UpdateEndpoint` ARM, which is live here and does not work on `ml.g6e`.
    `UpdateEndpoint` is a blue/green move: SageMaker brings up the new fleet BEFORE taking
    the old one down, so it needs `2 x InitialInstanceCount` of the instance type at once.
    The endpoint-usage quota for every `ml.g6e` size in this account is **1**, so the
    update is refused for want of capacity and the endpoint keeps serving the old config --
    i.e. it looks like the deploy did nothing. Delete-and-recreate is the only path on g6e
    today, which is why a `config.pbtxt` sweep costs a full teardown per cell.

    A quota bump to 2 would unblock it, and is the single cheapest unblock available here:
    it converts every configuration change from (delete, wait, create, wait) into one
    in-place update, and it is also what autoscaling needs to be more than decorative
    (see `resolve_max_capacity`).
    """
    try:
        current = sm.describe_endpoint(EndpointName=name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("ValidationException", "ResourceNotFound"):
            raise
        print(f"[endpoint] creating {name}")
        sm.create_endpoint(EndpointName=name, EndpointConfigName=config, Tags=_tags())
        return

    status = current["EndpointStatus"]
    in_use = current.get("EndpointConfigName")
    if status not in ("InService", "Failed"):
        print(f"[endpoint] {name} is {status}; not touching it. Wait for it to settle, "
              f"or delete it once it leaves {status}.")
        return
    if in_use == config:
        print(f"[endpoint] {name} already serves {config} ({status}); nothing to do")
        return
    print(f"[endpoint] {name} exists ({status}) on {in_use}; updating to {config}")
    sm.update_endpoint(EndpointName=name, EndpointConfigName=config)


def wait_in_service(sm, name: str, timeout: int = 2400) -> str:
    """Poll rather than use the waiter, so progress is visible: this takes many minutes."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        desc = sm.describe_endpoint(EndpointName=name)
        status = desc["EndpointStatus"]
        if status != last:
            print(f"[endpoint] {status}")
            last = status
        if status in ("InService", "Failed"):
            if status == "Failed":
                print(f"[endpoint] FailureReason: {desc.get('FailureReason')}",
                      file=sys.stderr)
            return status
        time.sleep(20)
    return "Timeout"


def endpoint_usage_quota(region: str, instance_type: str) -> float | None:
    """The account's `<instance type> for endpoint usage` quota, or None if it cannot be read.

    Looked up rather than hard-coded, because the quota is **per instance type and they are
    not the same number**. Measured in the account this was built in:

        ml.g6.xlarge   for endpoint usage -> 4
        ml.g6e.xlarge  for endpoint usage -> 1

    So the `DEFAULT_MAX_CAPACITY = 4` that is right for g6.xlarge is four times the real
    ceiling on g6e.xlarge. That mismatch is worth an API call because of HOW it fails:
    `application-autoscaling` happily accepts `MaxCapacity=4` -- it does not validate
    against the SageMaker quota -- and the endpoint then scales out to 2, gets
    `ResourceLimitExceeded` from SageMaker, and records it in the *scaling activity* log
    rather than anywhere a deploy script or an operator would look. The endpoint stays at 1
    instance, under load, silently, which is precisely the situation autoscaling was added
    to prevent.

    Service Quotas offers no name filter, so this pages the sagemaker service (~25 pages in
    us-west-2) and stops at the first match. A missing `servicequotas:ListServiceQuotas`
    permission returns None rather than raising: not being able to *check* the ceiling is
    not a reason to refuse to deploy.
    """
    want = f"{instance_type} for endpoint usage"
    try:
        sq = boto3.client("service-quotas", region_name=region)
        paginator = sq.get_paginator("list_service_quotas")
        for page in paginator.paginate(ServiceCode="sagemaker"):
            for quota in page.get("Quotas", []):
                if quota["QuotaName"] == want:
                    return float(quota["Value"])
    except ClientError as exc:
        print(f"[quota] could not read quotas ({exc.response['Error']['Code']}); "
              "not checking the autoscaling ceiling")
        return None
    except Exception as exc:  # botocore model/endpoint differences across partitions
        print(f"[quota] quota lookup failed ({type(exc).__name__}); not checking")
        return None
    print(f"[quota] no quota named {want!r} found")
    return None


def resolve_pooled_max_capacity(region: str, pools: list[dict], requested: int) -> int:
    """The autoscaling ceiling for a variant that may land on any of several types.

    Takes the BEST (largest) quota across the pools rather than the worst or the sum.

      * Not the sum: the quotas are per type and a single variant's instances are not
        guaranteed to spread across types, so the sum is a ceiling that may not exist.
      * Not the minimum: that would clamp to the worst fallback and refuse to use headroom
        the preferred type actually has.

    The best is still only an upper bound -- the real ceiling depends on which pool the
    fleet landed in, which is knowable only after the fact (`describe_placement`). Every
    pool's quota is printed so the spread is visible rather than summarised away.
    """
    best, quotas = 0, []
    for pool in pools:
        itype = pool["InstanceType"]
        quota = endpoint_usage_quota(region, itype)
        quotas.append(f"{itype}={'?' if quota is None else int(quota)}")
        if quota is not None:
            best = max(best, int(quota))
    print(f"[quota] endpoint-usage quota per pool: {', '.join(quotas)}")
    if best <= 0:
        # Unreadable everywhere -> do not block the deploy; the pooled create will be the
        # judge. Unlike the single-type path there is no one type to refuse on behalf of.
        return requested
    if requested > best:
        print(f"[quota] best pooled quota is {best}; clamping --max-capacity from "
              f"{requested} to {best}")
        if best == 1:
            print("[quota] NOTE: a maximum of 1 means autoscaling cannot add an instance, "
                  "and it also means UpdateEndpoint cannot move this endpoint to a new "
                  "config (blue/green needs 2x). Delete-and-recreate is the only path.")
        return best
    return requested


def resolve_max_capacity(region: str, instance_type: str, requested: int) -> int:
    """Clamp the autoscaling maximum to what the account can actually place.

    Clamped rather than refused: a max of 1 is a legitimate (if un-scalable) deployment,
    and the operator gets told plainly which it is. The alternative -- trusting `requested`
    -- is the silent scale-out failure described in `endpoint_usage_quota`.
    """
    quota = endpoint_usage_quota(region, instance_type)
    if quota is None:
        return requested
    allowed = int(quota)
    if allowed <= 0:
        raise SystemExit(
            f"the {instance_type} endpoint-usage quota in this account is 0, so this "
            "endpoint cannot be placed at all. Request an increase for "
            f"'{instance_type} for endpoint usage' in Service Quotas, or pick another "
            "instance type.")
    if requested > allowed:
        print(f"[quota] {instance_type} endpoint-usage quota is {allowed}; clamping "
              f"--max-capacity from {requested} to {allowed}")
        if allowed == 1:
            print("[quota] NOTE: a maximum of 1 means autoscaling cannot add an instance. "
                  "The target-tracking policy is still applied (it is harmless and becomes "
                  "useful the moment the quota is raised), but this endpoint will absorb a "
                  "spike as queueing, not as capacity.")
        return allowed
    print(f"[quota] {instance_type} endpoint-usage quota is {allowed}; "
          f"--max-capacity {requested} fits")
    return requested


def configure_autoscaling(region: str, endpoint: str, variant: str,
                          min_cap: int, max_cap: int, target: float) -> None:
    """Target tracking on invocations per instance.

    `SageMakerVariantInvocationsPerInstance` rather than GPUUtilization: the model's
    latency is dominated by prefill, so utilisation sits high even when the queue is short,
    which makes it a poor scaling signal here. Invocations-per-instance maps directly onto
    the thing with a known ceiling.

    `target` is per instance per MINUTE, and the measured ceiling is ~14 tickets/s, i.e.
    ~840/minute. ~250 is about 30% of that and is the recommended operating point: a new
    instance needs ~12 minutes to serve traffic, so scale-out has to be requested long
    before the current instance is in trouble. 600 (~70%) was deployed first and is too
    late -- see README.md, "Deploy".
    """
    aas = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{endpoint}/variant/{variant}"

    print(f"[autoscale] registering {resource_id} min={min_cap} max={max_cap}")
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_cap,
        MaxCapacity=max_cap,
    )
    aas.put_scaling_policy(
        PolicyName=f"{endpoint}-invocations-target",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": target,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            # Scale out readily, scale in slowly. A new instance pays the full cold start
            # (image pull + weight load + kernel compile), so flapping is expensive in a
            # way that holding a warm instance is not.
            "ScaleOutCooldown": 60,
            "ScaleInCooldown": 600,
        },
    )
    print("[autoscale] target tracking policy applied")


def delete_all(sm, region: str, endpoint: str, config: str, model: str, variant: str) -> None:
    aas = boto3.client("application-autoscaling", region_name=region)
    try:
        aas.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=f"endpoint/{endpoint}/variant/{variant}",
            ScalableDimension="sagemaker:variant:DesiredInstanceCount")
        print("[autoscale] deregistered")
    except ClientError as exc:
        print(f"[autoscale] {exc.response['Error']['Code']} (ignoring)")
    try:
        sm.delete_endpoint(EndpointName=endpoint)
        print("[endpoint] deleted")
    except ClientError as exc:
        print(f"[endpoint] {exc.response['Error']['Code']} (ignoring)")

    # Configs are named `<base>-config-<hash>`, one per distinct image/instance/engine, so
    # teardown has to sweep the prefix rather than delete a single known name -- otherwise
    # every previous deploy's config is orphaned. `config` is the one this invocation would
    # have created; it may not exist, and others probably do.
    prefix = config.rsplit("-config-", 1)[0] + "-config"
    try:
        found = sm.list_endpoint_configs(NameContains=prefix, MaxResults=100)
        for item in found.get("EndpointConfigs", []):
            try:
                sm.delete_endpoint_config(
                    EndpointConfigName=item["EndpointConfigName"])
                print(f"[config] deleted {item['EndpointConfigName']}")
            except ClientError as exc:
                print(f"[config] {exc.response['Error']['Code']} (ignoring)")
    except ClientError as exc:
        print(f"[config] list failed: {exc.response['Error']['Code']} (ignoring)")

    # Models are swept by prefix for the same reason configs are: `--model-data-url`
    # appends an overlay hash to the model name, so a sweep that ran several overlays has
    # several models and `--delete` is usually invoked WITHOUT the overlay argument. Deleting
    # only the one name computed from this invocation's flags would leave the rest behind.
    #
    # `NameContains` is a substring match, so the exact name is deleted first and then the
    # prefix sweep picks up `<model>-<hash>` siblings. Narrow by design: it matches this
    # deployment's own model name, not a bare project prefix that could reach a neighbour's
    # resources.
    try:
        sm.delete_model(ModelName=model)
        print(f"[model] deleted {model}")
    except ClientError as exc:
        print(f"[model] {exc.response['Error']['Code']} (ignoring)")
    base_model = model.split("-model", 1)[0] + "-model"
    try:
        for item in sm.list_models(NameContains=base_model,
                                   MaxResults=100).get("Models", []):
            if not item["ModelName"].startswith(base_model):
                continue
            try:
                sm.delete_model(ModelName=item["ModelName"])
                print(f"[model] deleted {item['ModelName']}")
            except ClientError as exc:
                print(f"[model] {exc.response['Error']['Code']} (ignoring)")
    except ClientError as exc:
        print(f"[model] list failed: {exc.response['Error']['Code']} (ignoring)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="ECR image URI (required unless --delete)")
    ap.add_argument("--role", help="SageMaker execution role ARN")
    ap.add_argument("--name", default="strands-decider", help="base name for all resources")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE)
    ap.add_argument("--instance-pools", default="",
                    help="comma-separated instance types in priority order, e.g. "
                         "'ml.g6e.4xlarge,ml.g6e.2xlarge,ml.g6e.xlarge'. SageMaker falls "
                         "back automatically on InsufficientInstanceCapacity instead of "
                         "failing after ~31 minutes. Max 5; overrides --instance-type")
    ap.add_argument("--model-data-url", default="",
                    help="S3 URI of a model-repository overlay tarball (built by "
                         "tools/make_overlay.py). Tunes config.pbtxt with no image "
                         "rebuild. Must contain decider/1/model.py as well as "
                         "decider/config.pbtxt -- it REPLACES /opt/ml/model")
    ap.add_argument("--variant", default="AllTraffic")
    ap.add_argument("--min-capacity", type=int, default=DEFAULT_MIN_CAPACITY)
    ap.add_argument("--max-capacity", type=int, default=DEFAULT_MAX_CAPACITY)
    ap.add_argument("--target-invocations", type=float, default=0.0,
                    help="invocations per instance per MINUTE to hold; 0 disables "
                         "autoscaling. The measured ceiling is ~840/min, so ~250 (~30%%) "
                         "is the recommended target -- a new instance needs ~12 min to "
                         "serve, so scale-out must be requested early")
    ap.add_argument("--engine", default="merged", choices=["merged", "hf"],
                    help="which forward pass the Triton backend builds")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    endpoint_name = args.name

    pools = parse_instance_pools(args.instance_pools) if args.instance_pools else None
    if pools and not instance_pools_supported():
        raise SystemExit(
            "--instance-pools needs a botocore that knows the InstancePools parameter "
            "(1.43.0 or later; this one does not have it). botocore validates against its "
            "bundled service model, so the request would be refused client-side with a "
            "misleading 'unknown parameter' error rather than by SageMaker. Upgrade with "
            "`pip install -U botocore boto3`, or drop --instance-pools.")

    # The model name carries a hash of the OVERLAY, when there is one, for exactly the
    # reason `cfg_fingerprint` below hashes the config: `ModelDataUrl` lives on the model,
    # `ensure_model` reuses an existing model by name, and a sweep that points the same
    # model name at a new overlay would therefore reuse the OLD overlay and report success.
    # That is the same silent "deploy did nothing" failure, one object along.
    model_name = f"{args.name}-model"
    if args.model_data_url:
        overlay_tag = hashlib.sha256(args.model_data_url.encode()).hexdigest()[:10]
        model_name = f"{model_name}-{overlay_tag}"

    # The endpoint config name carries a hash of what it configures, so changing the
    # image (or instance type, or env) produces a NEW name.
    #
    # This is load-bearing, not cosmetic. `UpdateEndpoint` is the only way to move a live
    # endpoint, and it refuses a config it is already serving:
    #   ValidationException: Cannot update endpoint with the currently in use endpoint
    #   configuration "strands-decider-g6-config"
    # With a fixed name, `--replace` deletes and recreates the config *under the same
    # name*, so there is nothing new to point at -- and a name-equality check (which is
    # all `DescribeEndpoint` gives you) then concludes the endpoint is already correct and
    # silently does nothing. Measured: a deploy of a new image reported success and left
    # the old image serving. Hashing the content makes the comparison meaningful.
    #
    # Old configs are left behind deliberately: they cost nothing, and keeping them means
    # a rollback is `--image <previous>`, which resolves to a config that already exists.
    cfg_fingerprint = hashlib.sha256(
        json.dumps([args.image, args.instance_type, args.variant, args.engine,
                    # Both of these change what gets served, so both must change the name.
                    # The pool list is ordered, and the order IS the priority, so
                    # reordering it is a different configuration and must not collide.
                    pools, args.model_data_url],
                   sort_keys=True).encode()).hexdigest()[:10]
    config_name = f"{args.name}-config-{cfg_fingerprint}"

    if args.delete:
        delete_all(sm, args.region, endpoint_name, config_name, model_name, args.variant)
        return 0

    if not args.image:
        ap.error("--image is required (or use --delete)")

    role = args.role
    if not role:
        # Convenience only: in a notebook/Studio context the caller often *is* the
        # execution role. Anywhere else, pass --role explicitly.
        ident = boto3.client("sts", region_name=args.region).get_caller_identity()
        ap.error("--role is required; no SageMaker execution role could be inferred "
                 f"(caller is {ident['Arn']})")

    env = {
        # The DLC reads the model repository from /opt/ml/model and needs the model name;
        # both are baked into the image but set here too so an override needs no rebuild.
        "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "decider",
        # Metrics, and the port they must live on. SAGEMAKER_TRITON_METRICS_PORT is NOT
        # optional once metrics are enabled: Triton otherwise defaults to 8002, SageMaker
        # restricts the range to [23000, 23999], and the container dies at start with
        #   "The server cannot listen to metrics requests at port 8002,
        #    allowed port range is [23000, 23999]"
        # followed by tritonserver dumping its --help. The DLC launcher reads the var at
        # /usr/bin/serve:82 and range-checks it against SAGEMAKER_SAFE_PORT_RANGE.
        "SAGEMAKER_TRITON_ALLOW_METRICS": "true",
        "SAGEMAKER_TRITON_METRICS_PORT": "23000",
        "SD_ENGINE": args.engine,
        "SD_PREFIX_CACHE": "1",
    }

    ensure_model(sm, model_name, args.image, role, env, args.replace,
                 model_data_url=args.model_data_url)
    ensure_endpoint_config(sm, config_name, model_name, args.instance_type,
                           args.variant, args.replace, pools=pools)
    ensure_endpoint(sm, endpoint_name, config_name)

    if args.no_wait:
        print(f"[endpoint] not waiting; poll with: aws sagemaker describe-endpoint "
              f"--endpoint-name {endpoint_name} --region {args.region}")
        return 0

    status = wait_in_service(sm, endpoint_name)
    if status != "InService":
        return 1

    if pools:
        # Which type actually served matters for every number measured against it.
        print(f"[endpoint] placed on {describe_placement(sm, endpoint_name)}")

    if args.target_invocations > 0:
        max_capacity = (resolve_pooled_max_capacity(args.region, pools, args.max_capacity)
                        if pools else
                        resolve_max_capacity(args.region, args.instance_type,
                                             args.max_capacity))
        configure_autoscaling(args.region, endpoint_name, args.variant,
                              args.min_capacity, max_capacity,
                              args.target_invocations)
    else:
        print("[autoscale] skipped: no --target-invocations given. The measured ceiling "
              "is ~840 invocations/instance/minute, so ~250 is a sensible target")

    print(json.dumps({"endpoint": endpoint_name, "status": status,
                      "invoke": f"aws sagemaker-runtime invoke-endpoint "
                                f"--endpoint-name {endpoint_name} "
                                f"--content-type application/octet-stream "
                                f"--body fileb://request.json out.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
