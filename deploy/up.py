#!/usr/bin/env python3
"""One command from a fresh clone plus AWS credentials to a working endpoint.

`create_endpoint.py` already does the SageMaker half well, but it is not a starting point:
it requires `--role <arn>` and `--image <uri>`, and a fresh clone has neither. Getting
them meant creating an execution role by hand, knowing the ECR registry for your own
account, and knowing that `build_on_box.sh` wants a build bucket that it does not create.
That is four undocumented steps between "I cloned this" and "I can call it", which is the
gap this file closes. It ORCHESTRATES the existing scripts rather than reimplementing them
-- `build_on_box.sh` still owns the build and `create_endpoint.py` still owns the endpoint,
so there is one implementation of each to keep correct.

    deploy/up.py --image <acct>.dkr.ecr.<region>.amazonaws.com/strands-decider-serving:v23-triton-onepass
    deploy/up.py --build --box-id i-0123456789abcdef0 --tag v23-triton-onepass
    deploy/up.py --down

What each step is for:

  1. the execution role     created if absent, with the narrowest policy that works
  2. the build bucket       created if absent, and ONLY when --build is given
  3. the image              delegated to build_on_box.sh, only with --build
  4. the endpoint           delegated to create_endpoint.py
  5. a real inference       because `InService` is not the same as "answers correctly"

Step 5 is the one worth defending. An endpoint reaches `InService` as soon as `/ping`
returns 200, which this container does *after* warm-up -- but a wrong `SD_ENGINE`, a model
repository the DLC could not find, or a CUDA 13 base on an old host all produce an endpoint
that is `InService` and fails every invocation. So this exits non-zero unless a real
three-primitive request came back with real probabilities, through `decider_triton.wire`,
which is the same envelope encoder the client and the tests use.

Costs money. One `ml.g6.xlarge` is ~$1.13/hr and one `ml.g6e.xlarge` ~$1.86/hr, charged
whether or not anything calls it, so `--down` is part of the workflow rather than an
afterthought. `--down` deletes the endpoint, its configs, the model and (with
`--delete-role`) the role -- it does NOT delete the ECR image or the build bucket, because
those are what make the next `up.py` fast and cost cents rather than dollars.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The role this creates when none is given. Named rather than random so a second run finds
# and reuses it instead of leaving a trail of roles behind.
DEFAULT_ROLE_NAME = "strands-decider-sagemaker-execution"
DEFAULT_REPO = "strands-decider-serving"

# Pulling the image is the only thing a SageMaker endpoint strictly needs from AWS here:
# the weights are baked into the image, so there is no ModelDataUrl and therefore no S3
# read. AmazonSageMakerFullAccess is the usual copy-paste and is far too broad for that --
# it grants S3, IAM PassRole and the whole SageMaker API to a role whose only job is to
# pull a container and write logs.
ECR_READ_POLICY = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"

# Logs and metrics, written by the DLC rather than by this repository's code. Inline and
# scoped to the SageMaker log group prefix instead of CloudWatchLogsFullAccess.
OBSERVABILITY_POLICY_NAME = "strands-decider-logs-and-metrics"


def trust_policy(account: str) -> dict:
    """Let SageMaker assume the role, and only on behalf of this account.

    The `aws:SourceAccount` condition is the confused-deputy guard. Without it the trust
    policy says "any SageMaker in any account may assume this role", and a role ARN is not
    a secret -- it appears in CloudTrail, in error messages and in this repository's own
    README examples. The condition costs nothing and removes the class entirely.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": account}},
        }],
    }


def observability_policy(region: str, account: str) -> dict:
    """Write endpoint logs and the per-variant metrics, and nothing else.

    Scoped to `/aws/sagemaker/*` log groups. `cloudwatch:PutMetricData` cannot be scoped by
    resource -- it takes `*` -- so it is narrowed by namespace condition instead, which is
    the only lever the action offers.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream",
                           "logs:PutLogEvents", "logs:DescribeLogStreams"],
                "Resource": [
                    f"arn:aws:logs:{region}:{account}:log-group:/aws/sagemaker/*",
                    f"arn:aws:logs:{region}:{account}:log-group:/aws/sagemaker/*:log-stream:*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*",
                "Condition": {"StringEquals": {
                    "cloudwatch:namespace": ["AWS/SageMaker", "/aws/sagemaker/Endpoints"]}},
            },
        ],
    }


MODEL_DATA_POLICY_NAME = "strands-decider-model-data"


def model_data_policy(url: str) -> dict:
    """Read exactly one S3 archive, for a deploy that passes `--model-data-url`.

    Normally this role needs NO S3 at all -- the weights are baked into the image, which is
    why `ECR_READ_POLICY` plus logs is the whole grant. A model-repository overlay (the
    no-rebuild route to a Triton-level knob like `instance_group count`) is the one case
    that adds an S3 read, and without it `CreateEndpoint` fails minutes later with
      Could not access model data at s3://... Please ensure that the role can access it
    which reads like a bad URL rather than a missing permission.

    Scoped to the single object, and `ListBucket` to its key alone: SageMaker issues a
    HeadObject/ListObjectsV2 before the GET, and a bucket-wide list grant is a wider hole
    than this needs.
    """
    bucket, _, key = url.removeprefix("s3://").partition("/")
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "s3:GetObject",
             "Resource": f"arn:aws:s3:::{bucket}/{key}"},
            {"Effect": "Allow", "Action": "s3:ListBucket",
             "Resource": f"arn:aws:s3:::{bucket}",
             "Condition": {"StringEquals": {"s3:prefix": key}}},
        ],
    }


def ensure_execution_role(region: str, account: str, name: str,
                          model_data_url: str = "") -> str:
    """Create or reuse the SageMaker execution role, and return its ARN.

    The sleep at the end is not superstition. IAM is eventually consistent across its own
    control plane, and `CreateModel` with a role created seconds earlier fails with
      ValidationException: Could not access model ... validate that the role ...
    or an AccessDenied naming a role that demonstrably exists. There is no waiter for role
    propagation, so a freshly created role gets a fixed pause; a reused one needs none,
    which is why the pause is inside the create branch only.
    """
    iam = boto3.client("iam", region_name=region)
    created = False
    try:
        existing = iam.get_role(RoleName=name)
        print(f"[role] {name} exists; reusing")
        arn = existing["Role"]["Arn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        print(f"[role] creating {name}")
        arn = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust_policy(account)),
            Description="Pulls the strands-decider Triton image and writes endpoint logs.",
            Tags=[{"Key": "Project", "Value": "strands-decider"},
                  {"Key": "ManagedBy", "Value": "deploy/up.py"}],
        )["Role"]["Arn"]
        created = True

    # Idempotent: attaching an already-attached managed policy is a no-op, and putting an
    # inline policy overwrites it, so a re-run repairs a half-configured role rather than
    # failing on it.
    iam.attach_role_policy(RoleName=name, PolicyArn=ECR_READ_POLICY)
    iam.put_role_policy(
        RoleName=name,
        PolicyName=OBSERVABILITY_POLICY_NAME,
        PolicyDocument=json.dumps(observability_policy(region, account)),
    )
    policies = [Path(ECR_READ_POLICY).name, OBSERVABILITY_POLICY_NAME]
    if model_data_url:
        iam.put_role_policy(
            RoleName=name,
            PolicyName=MODEL_DATA_POLICY_NAME,
            PolicyDocument=json.dumps(model_data_policy(model_data_url)),
        )
        policies.append(MODEL_DATA_POLICY_NAME)
    print(f"[role] policies in place ({', '.join(policies)})")

    if created:
        print("[role] waiting 15s for IAM propagation before SageMaker uses it")
        time.sleep(15)
    return arn


def delete_execution_role(region: str, name: str) -> None:
    """Detach everything first: IAM refuses to delete a role that still has policies."""
    iam = boto3.client("iam", region_name=region)
    try:
        for pol in iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=name, PolicyArn=pol["PolicyArn"])
        for pol in iam.list_role_policies(RoleName=name)["PolicyNames"]:
            iam.delete_role_policy(RoleName=name, PolicyName=pol)
        iam.delete_role(RoleName=name)
        print(f"[role] deleted {name}")
    except ClientError as exc:
        print(f"[role] {exc.response['Error']['Code']} (ignoring)")


def ensure_build_bucket(region: str, bucket: str) -> None:
    """The build-context bucket `build_on_box.sh` syncs through.

    It is created here rather than there because `build_on_box.sh` runs `aws s3 sync` as its
    first real action and a missing bucket surfaces as a bare `NoSuchBucket` after the
    script has already printed its plan -- which reads like a credentials problem.

    `us-east-1` must NOT be given a LocationConstraint; every other region must. Passing it
    in us-east-1 fails with InvalidLocationConstraint, which is the one special case in
    this API.
    """
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[bucket] s3://{bucket} exists; reusing")
        return
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket", "403"):
            raise
        if exc.response["Error"]["Code"] == "403":
            # Someone else owns this name. Bucket names are global, so say so plainly.
            raise SystemExit(
                f"s3://{bucket} exists but is not yours. Pass --bucket <name> with a name "
                "you own; S3 bucket names are globally unique.") from exc

    print(f"[bucket] creating s3://{bucket}")
    kwargs: dict = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    # The context holds application code, and briefly an HF token file. Neither should be
    # publicly reachable or unversioned-on-overwrite by accident.
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={"BlockPublicAcls": True, "IgnorePublicAcls": True,
                                        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={"Rules": [
            {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})


def run(cmd: list[str], env: dict | None = None) -> None:
    """Run a child script, streaming its output, and stop on failure.

    Streamed rather than captured: the build takes tens of minutes and a silent subprocess
    is indistinguishable from a hung one.
    """
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed with exit code {result.returncode}")


def verify(region: str, endpoint: str) -> dict:
    """Invoke the endpoint for real, with all three primitives, and validate the shape.

    Uses `decider_triton.wire` rather than hand-building the KServe v2 envelope, for the
    same reason every other caller here does: there is one implementation of the wrapping
    and it is the one the tests cover. A hand-rolled `shape: [1]` is the single most common
    way to get an opaque `Unable to parse 'inputs'` out of this endpoint.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from decider_triton.wire import decode_v2_response, encode_v2_request

    body = {
        "state": "Help! My payouts have been failing for 3 days and nobody has replied.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "payments", "technical": "bugs"}},
            "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                            "criteria": ["calm", "frustrated", "depressed"]},
        },
    }
    rt = boto3.client("sagemaker-runtime", region_name=region)
    started = time.perf_counter()
    raw = rt.invoke_endpoint(
        EndpointName=endpoint, ContentType="application/json",
        Body=json.dumps(encode_v2_request(body)),
    )["Body"].read().decode()
    elapsed = (time.perf_counter() - started) * 1000
    out = decode_v2_response(json.loads(raw))

    if "error" in out:
        raise SystemExit(f"endpoint answered with an error: {out['error']}")
    answers = out.get("answers") or {}
    missing = {"urgent", "team", "frustration"} - set(answers)
    if missing:
        raise SystemExit(f"endpoint answered without {sorted(missing)}: {out}")
    # Shape, not value: the correctness gate is tools/reference_check.py against the
    # model's published reference values, not a smoke test's opinion of one request.
    if not 0.0 <= answers["urgent"]["noul"] <= 1.0:
        raise SystemExit(f"noul out of range: {answers['urgent']}")
    if answers["team"]["choice"] not in {"billing", "technical"}:
        raise SystemExit(f"choice not one of the options: {answers['team']}")

    print(f"\n[verify] OK in {elapsed:.0f} ms round trip, "
          f"{out.get('latency_ms')} ms server-side")
    print(f"[verify] urgent={answers['urgent']['noul']:.4f} "
          f"team={answers['team']['choice']} "
          f"frustration={answers['frustration']['score']:.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="strands-decider",
                    help="base name for every resource (default: strands-decider)")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--image", default="",
                    help="ECR image URI. Omit it with --build, or to accept the default "
                         "registry/repo/tag for your own account")
    ap.add_argument("--instance-type", default="ml.g6.xlarge",
                    help="ml.g6.xlarge (L4, measured) or ml.g6e.xlarge (L40S). NOT ml.g5: "
                         "its host drivers are too old for any current Triton DLC")
    ap.add_argument("--role", default="",
                    help="an existing execution role ARN; one is created if omitted")
    ap.add_argument("--role-name", default=DEFAULT_ROLE_NAME)
    # Passed straight through to create_endpoint.py, which owns what they mean. They are
    # here because `up.py` is the path that also VERIFIES the endpoint with a real
    # invocation, and a tuning-knob deploy is exactly the kind that needs verifying: an
    # env var the backend refuses (SD_FUSE_LAYERS=1 on SD_ENGINE=hf) fails inside
    # initialize(), and without step 5 that looks like a healthy endpoint.
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra container environment, repeatable "
                         "(e.g. --env SD_FUSE_LAYERS=1)")
    ap.add_argument("--model-data-url", default="",
                    help="s3:// archive extracted over the Triton model repository at "
                         "/opt/ml/model; the no-rebuild route to instance_group count")
    ap.add_argument("--target-invocations", type=float, default=0.0,
                    help="invocations/instance/MINUTE for autoscaling; 0 disables it. See "
                         "create_endpoint.py for how to pick this from the measured ceiling")
    ap.add_argument("--build", action="store_true",
                    help="build and push the image first (needs --box-id)")
    ap.add_argument("--box-id", default=os.environ.get("BOX_ID", ""),
                    help="in-region amd64 GPU instance to build on; required with --build")
    ap.add_argument("--tag", default="v23-triton-onepass")
    # Forwarded to build_on_box.sh, which forwards them to Dockerfile.triton's ARGs. Empty
    # means "whatever the Dockerfile defaults to", so the default build is unchanged.
    # Only meaningful with --build: the checkpoint is baked into the image at build time,
    # so pointing an EXISTING image at another checkpoint is not a thing you can do.
    ap.add_argument("--checkpoint-repo", default="",
                    help="Hub id of the checkpoint to bake in, e.g. "
                         "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610. Needs --build; "
                         "tag the result for the model (one image serves one checkpoint)")
    ap.add_argument("--checkpoint-revision", default="",
                    help="Hub revision of --checkpoint-repo. Needs --build")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--bucket", default="", help="build-context bucket; defaulted per account")
    ap.add_argument("--down", action="store_true", help="delete what this created")
    ap.add_argument("--delete-role", action="store_true",
                    help="with --down, also delete the execution role")
    args = ap.parse_args()

    ident = boto3.client("sts", region_name=args.region).get_caller_identity()
    account = ident["Account"]
    print(f"[aws] account {account}, region {args.region}, caller {ident['Arn']}")

    registry = f"{account}.dkr.ecr.{args.region}.amazonaws.com"
    image = args.image or f"{registry}/{args.repo}:{args.tag}"
    bucket = args.bucket or f"strands-decider-build-{account}-{args.region}"

    if args.down:
        # Delegate so there is one teardown implementation. create_endpoint.py sweeps every
        # `<name>-config-<hash>` rather than one known name, which matters because each
        # image/instance combination left a config behind.
        run([sys.executable, str(HERE / "create_endpoint.py"),
             "--name", args.name, "--region", args.region, "--delete"])
        if args.delete_role:
            delete_execution_role(args.region, args.role_name)
        else:
            print(f"[role] keeping {args.role_name} (--delete-role to remove it)")
        print("\n[down] endpoint, configs and model deleted. The ECR image and the build "
              "bucket are deliberately left: they cost cents and make the next `up.py` "
              "a few minutes instead of an hour.")
        return 0

    if args.build:
        if not args.box_id:
            raise SystemExit(
                "--build needs --box-id: an in-region amd64 GPU instance with docker, the "
                "SSM agent and ~200 GB free on /. A laptop cannot do this build -- the "
                "base image alone is 27.7 GB and an arm64 cross-build emulates every RUN "
                "step. To list candidates:\n"
                f"  aws ec2 describe-instances --region {args.region} \\\n"
                "    --filters Name=instance-state-name,Values=running \\\n"
                "    --query 'Reservations[].Instances[].[InstanceId,InstanceType]' "
                "--output text")
        ensure_build_bucket(args.region, bucket)
        build_env = {"BOX_ID": args.box_id, "REGION": args.region,
                     "BUCKET": bucket, "REPO": args.repo}
        # Only set when non-empty: build_on_box.sh forwards a --build-arg per variable it
        # sees, and an empty one would override the Dockerfile's ARG with an empty string.
        if args.checkpoint_repo:
            build_env["SD_CHECKPOINT_REPO"] = args.checkpoint_repo
        if args.checkpoint_revision:
            build_env["SD_CHECKPOINT_REVISION"] = args.checkpoint_revision
        run([str(HERE / "build_on_box.sh"), args.tag], env=build_env)
    elif args.checkpoint_repo or args.checkpoint_revision:
        # Fail rather than ignore it. The checkpoint is baked into the image, so without
        # --build these flags describe a build that is not happening and the endpoint would
        # come up serving whatever the named tag already contains -- a silent mismatch
        # between what you asked for and what answers.
        raise SystemExit(
            "--checkpoint-repo/--checkpoint-revision only apply with --build: the "
            "checkpoint is baked into the image, so selecting one means building one. "
            f"Either add --build --box-id <id>, or drop the flag and accept whatever "
            f"{image} already contains.")

    if args.role and args.model_data_url:
        # Not modified here on purpose: a role passed in belongs to the operator, and
        # silently widening someone else's role is worse than a clear instruction.
        print(f"[role] NOTE: --model-data-url needs s3:GetObject on "
              f"{args.model_data_url} and {args.role} is yours to grant it on. Without it "
              "CreateEndpoint fails with 'Could not access model data at s3://...', which "
              "reads like a bad URL rather than a missing permission.")
    role = args.role or ensure_execution_role(args.region, account, args.role_name,
                                              model_data_url=args.model_data_url)

    endpoint_cmd = [sys.executable, str(HERE / "create_endpoint.py"),
                    "--image", image, "--role", role, "--name", args.name,
                    "--region", args.region, "--instance-type", args.instance_type]
    if args.target_invocations > 0:
        endpoint_cmd += ["--target-invocations", str(args.target_invocations)]
    for item in args.env:
        endpoint_cmd += ["--env", item]
    if args.model_data_url:
        endpoint_cmd += ["--model-data-url", args.model_data_url]
    run(endpoint_cmd)

    verify(args.region, args.name)

    print(f"\n[up] endpoint {args.name} is serving {image} on {args.instance_type}.")
    print("[up] call it:  see CLIENT.md, or notebooks/decider_playground.ipynb")
    print(f"[up] tear down: deploy/up.py --name {args.name} "
          f"--region {args.region} --down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
