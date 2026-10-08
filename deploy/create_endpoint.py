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
# ml.g6.xlarge (x2), ml.g6e.xlarge, ml.g6.2xlarge and ml.g6.4xlarge, ~31 minutes each -- the
# remedy is to run several CreateEndpoint attempts concurrently under different --name
# values and keep whichever lands, not to fall back to g5. See README.md, "Deploy".
DEFAULT_INSTANCE = "ml.g6.xlarge"

# A warm floor of 1, deliberately. Inference Components can scale a model to zero, but GPU
# cold start here is the image pull plus weight load plus the Gated DeltaNet kernel compile
# -- measured 78.8 s for a fresh 23.5 GB pull, and the Triton image is larger. The brief
# called this out and it is why scale-to-zero is not the default.
DEFAULT_MIN_CAPACITY = 1
DEFAULT_MAX_CAPACITY = 4  # the ml.g6.xlarge endpoint-usage quota in this account (verified)

# `/ping` returns 503 for the whole pull + load + warm-up window, and SageMaker kills the
# container if it does not pass before this expires. Generous on purpose.
STARTUP_HEALTH_CHECK_TIMEOUT = 1800
MODEL_DATA_DOWNLOAD_TIMEOUT = 1800


def _tags(extra: dict | None = None) -> list[dict]:
    tags = {"Project": "strands-decider", "ManagedBy": "decider-server/deploy"}
    tags.update(extra or {})
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def ensure_model(sm, name: str, image: str, role: str, env: dict, replace: bool) -> str:
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
    sm.create_model(
        ModelName=name,
        ExecutionRoleArn=role,
        # No ModelDataUrl: the weights are baked into the image, so there is no 4.6 GB S3
        # download on every scale-out. If you do supply one, SageMaker untars it to
        # /opt/ml/model and the entrypoint prefers it -- but the archive must have
        # strands_decider_config.json (or hobson_config.json) at its ROOT, not nested.
        PrimaryContainer={"Image": image, "Environment": env},
        Tags=_tags(),
    )
    return name


def ensure_endpoint_config(sm, name: str, model_name: str, instance: str,
                           variant: str, replace: bool) -> str:
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

    print(f"[config] creating {name} ({instance})")
    sm.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[{
            "VariantName": variant,
            "ModelName": model_name,
            "InitialInstanceCount": DEFAULT_MIN_CAPACITY,
            "InstanceType": instance,
            # Weight matters only once there is a second variant for an A/B.
            "InitialVariantWeight": 1.0,
            "ContainerStartupHealthCheckTimeoutInSeconds": STARTUP_HEALTH_CHECK_TIMEOUT,
            "ModelDataDownloadTimeoutInSeconds": MODEL_DATA_DOWNLOAD_TIMEOUT,
        }],
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

    try:
        sm.delete_model(ModelName=model)
        print("[model] deleted")
    except ClientError as exc:
        print(f"[model] {exc.response['Error']['Code']} (ignoring)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="ECR image URI (required unless --delete)")
    ap.add_argument("--role", help="SageMaker execution role ARN")
    ap.add_argument("--name", default="strands-decider", help="base name for all resources")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE)
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
    model_name = f"{args.name}-model"
    endpoint_name = args.name

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
        json.dumps([args.image, args.instance_type, args.variant, args.engine],
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

    ensure_model(sm, model_name, args.image, role, env, args.replace)
    ensure_endpoint_config(sm, config_name, model_name, args.instance_type,
                           args.variant, args.replace)
    ensure_endpoint(sm, endpoint_name, config_name)

    if args.no_wait:
        print(f"[endpoint] not waiting; poll with: aws sagemaker describe-endpoint "
              f"--endpoint-name {endpoint_name} --region {args.region}")
        return 0

    status = wait_in_service(sm, endpoint_name)
    if status != "InService":
        return 1

    if args.target_invocations > 0:
        configure_autoscaling(args.region, endpoint_name, args.variant,
                              args.min_capacity, args.max_capacity,
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
