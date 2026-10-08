"""The deploy scripts' decisions, with AWS stubbed out.

`deploy/` creates billable resources, so almost none of it is testable without an account.
What IS testable is the part that decides *what* to create, and three of those decisions are
worth a regression test because each has a silent failure mode:

  * the execution role's trust policy. A missing `aws:SourceAccount` condition is not
    visible in any console view that matters and cannot be detected from the outside.
  * the autoscaling ceiling against the account's real quota. Getting this wrong produces an
    endpoint that reports healthy, never scales, and records the refusal only in a scaling
    activity log (see `create_endpoint.endpoint_usage_quota`).
  * what identifies a model and an endpoint config. Both are immutable, both are reused by
    NAME, and both therefore have a failure mode where a deploy reports success and leaves
    the previous settings serving -- which turns an A/B into a comparison of one thing with
    itself (see `create_endpoint.config_fingerprint` and `container_drift`).
  * `tools/loadsweep_triton.py` being importable without running a load sweep.

The scripts are loaded by path rather than imported, because `deploy/` and `tools/` are
directories of entry points, not packages -- the same approach `test_triton_backend.py` uses
for `model_repository/decider/1/model.py`.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def up():
    pytest.importorskip("boto3", reason="deploy/up.py imports boto3 at module level")
    return _load(ROOT / "deploy" / "up.py", "deploy_up")


@pytest.fixture(scope="module")
def create_endpoint():
    pytest.importorskip("boto3")
    return _load(ROOT / "deploy" / "create_endpoint.py", "deploy_create_endpoint")


# ----------------------------------------------------------------- the execution role


def test_trust_policy_is_scoped_to_this_account(up):
    """The confused-deputy guard. A role ARN is not a secret -- it is in CloudTrail, in
    error messages and in this repo's own README examples -- so the trust policy must say
    *which* account's SageMaker may assume it, not just "SageMaker"."""
    policy = up.trust_policy("111122223333")
    (statement,) = policy["Statement"]
    assert statement["Principal"] == {"Service": "sagemaker.amazonaws.com"}
    assert statement["Action"] == "sts:AssumeRole"
    assert statement["Condition"]["StringEquals"]["aws:SourceAccount"] == "111122223333"


def test_trust_policy_does_not_allow_a_wildcard_principal(up):
    policy = up.trust_policy("111122223333")
    assert "*" not in str(policy["Statement"][0]["Principal"])


def test_observability_policy_scopes_logs_to_sagemaker_groups(up):
    """Logs are scoped by resource; PutMetricData cannot be (the action takes `*`), so it is
    narrowed by namespace condition instead. If either loosens to a bare `*` with no
    condition, this role stops being least-privilege and nothing else would notice."""
    policy = up.observability_policy("us-west-2", "111122223333")
    logs = [s for s in policy["Statement"] if "logs:PutLogEvents" in s["Action"]]
    assert logs, "the role must be able to write endpoint logs"
    assert all(r.startswith("arn:aws:logs:us-west-2:111122223333:log-group:/aws/sagemaker/")
               for r in logs[0]["Resource"])

    metrics = [s for s in policy["Statement"] if s["Action"] == "cloudwatch:PutMetricData"]
    assert metrics, "the DLC publishes per-variant metrics"
    assert metrics[0]["Condition"]["StringEquals"]["cloudwatch:namespace"]


def test_model_data_policy_is_scoped_to_one_object(up):
    """This role normally needs no S3 at all -- the weights are in the image. The one case
    that adds a read is a model-repository overlay, and it must stay an object-level grant:
    a bucket-wide `s3:GetObject` on `*` would be a far wider hole than the one archive the
    deploy named."""
    policy = up.model_data_policy("s3://sd-exchange-123/overlays/ig2.tar.gz")
    get = [s for s in policy["Statement"] if s["Action"] == "s3:GetObject"]
    assert get[0]["Resource"] == "arn:aws:s3:::sd-exchange-123/overlays/ig2.tar.gz"
    assert "*" not in get[0]["Resource"]
    listing = [s for s in policy["Statement"] if s["Action"] == "s3:ListBucket"]
    assert listing[0]["Resource"] == "arn:aws:s3:::sd-exchange-123"
    assert (listing[0]["Condition"]["StringEquals"]["s3:prefix"]
            == "overlays/ig2.tar.gz")


def test_execution_role_does_not_grant_sagemaker_full_access(up):
    """AmazonSageMakerFullAccess is the usual copy-paste and grants S3, IAM PassRole and the
    whole SageMaker API to a role whose only job is to pull a container and write logs. The
    weights are baked into the image, so there is no ModelDataUrl and no S3 read at all."""
    assert "AmazonSageMakerFullAccess" not in up.ECR_READ_POLICY
    assert up.ECR_READ_POLICY.endswith("AmazonEC2ContainerRegistryReadOnly")


# ------------------------------------------------------------- the autoscaling ceiling


def test_max_capacity_is_clamped_to_the_quota(create_endpoint, monkeypatch, capsys):
    """The g6e case that motivated this. DEFAULT_MAX_CAPACITY is 4 because that is the
    measured ml.g6.xlarge quota; ml.g6e.xlarge is 1 in the same account, and
    application-autoscaling would accept 4 without complaint."""
    monkeypatch.setattr(create_endpoint, "endpoint_usage_quota",
                        lambda region, instance_type: 1.0)
    assert create_endpoint.resolve_max_capacity("us-west-2", "ml.g6e.xlarge", 4) == 1
    # The operator has to be told that autoscaling is now inert, not just that it clamped.
    assert "cannot add an instance" in capsys.readouterr().out


def test_max_capacity_is_left_alone_when_it_fits(create_endpoint, monkeypatch):
    monkeypatch.setattr(create_endpoint, "endpoint_usage_quota",
                        lambda region, instance_type: 4.0)
    assert create_endpoint.resolve_max_capacity("us-west-2", "ml.g6.xlarge", 4) == 4
    assert create_endpoint.resolve_max_capacity("us-west-2", "ml.g6.xlarge", 2) == 2


def test_unreadable_quota_does_not_block_a_deploy(create_endpoint, monkeypatch):
    """Not being able to *check* the ceiling is not a reason to refuse to deploy: the
    caller may simply lack servicequotas:ListServiceQuotas."""
    monkeypatch.setattr(create_endpoint, "endpoint_usage_quota",
                        lambda region, instance_type: None)
    assert create_endpoint.resolve_max_capacity("us-west-2", "ml.g6.xlarge", 4) == 4


def test_zero_quota_refuses_rather_than_deploying_something_unplaceable(
        create_endpoint, monkeypatch):
    """A quota of 0 is real -- ml.g6.24xlarge and ml.g6e.48xlarge are both 0 in the account
    this was built in. CreateEndpoint would take ~31 minutes to fail with
    InsufficientInstanceCapacity-shaped noise, so say so in one second instead."""
    monkeypatch.setattr(create_endpoint, "endpoint_usage_quota",
                        lambda region, instance_type: 0.0)
    with pytest.raises(SystemExit, match="quota in this account is 0"):
        create_endpoint.resolve_max_capacity("us-west-2", "ml.g6.24xlarge", 4)


# ----------------------------------------------- what identifies a model and a config


def test_config_fingerprint_moves_with_every_field_it_configures(create_endpoint):
    """A config name that does not move makes UpdateEndpoint a no-op, and the deploy then
    reports success while the old container keeps serving. `--env` and `--model-data-url`
    are the two newest ways to reach that bug, so they are asserted explicitly."""
    fp = create_endpoint.config_fingerprint
    base = fp("img:a", "ml.g6e.2xlarge", "AllTraffic", "merged", [], "")
    assert base == fp("img:a", "ml.g6e.2xlarge", "AllTraffic", "merged", [], ""), \
        "the fingerprint must be stable, or every re-run leaks a config"
    assert base != fp("img:b", "ml.g6e.2xlarge", "AllTraffic", "merged", [], "")
    assert base != fp("img:a", "ml.g6e.4xlarge", "AllTraffic", "merged", [], "")
    assert base != fp("img:a", "ml.g6e.2xlarge", "AllTraffic", "hf", [], "")
    assert base != fp("img:a", "ml.g6e.2xlarge", "AllTraffic", "merged",
                      ["SD_FUSE_LAYERS=1"], "")
    assert base != fp("img:a", "ml.g6e.2xlarge", "AllTraffic", "merged", [],
                      "s3://b/ig2.tar.gz")


def test_config_fingerprint_ignores_the_order_env_was_given_in(create_endpoint):
    """Two deploys that differ only in argument order are the same deployment, and giving
    them different config names would leave an orphan behind for nothing."""
    fp = create_endpoint.config_fingerprint
    a = fp("i", "t", "v", "merged", ["A=1", "B=2"], "")
    b = fp("i", "t", "v", "merged", ["B=2", "A=1"], "")
    assert a == b


def test_env_overrides_split_on_the_first_equals_only(create_endpoint):
    """A value may contain `=`. Splitting on all of them would silently truncate it."""
    merged = create_endpoint.apply_env_overrides({"SD_ENGINE": "merged"},
                                                 ["SD_FUSE_LAYERS=1", "X=a=b"])
    assert merged == {"SD_ENGINE": "merged", "SD_FUSE_LAYERS": "1", "X": "a=b"}


def test_env_override_can_replace_a_default(create_endpoint):
    """The point of the flag: the backend's param() lets the environment win over
    config.pbtxt, so this is how a knob is flipped without a 20.6 GB rebuild."""
    merged = create_endpoint.apply_env_overrides({"SD_PREFIX_CACHE": "1"},
                                                 ["SD_PREFIX_CACHE=0"])
    assert merged["SD_PREFIX_CACHE"] == "0"


def test_env_override_refuses_a_bare_key(create_endpoint):
    """`--env SD_FUSE_LAYERS` would otherwise set it to "", which param() reads as OFF --
    i.e. it would deploy the default while the command line asked for the opposite."""
    with pytest.raises(ValueError, match="KEY=VALUE"):
        create_endpoint.apply_env_overrides({}, ["SD_FUSE_LAYERS"])


def test_container_drift_catches_an_environment_only_change(create_endpoint):
    """There is no UpdateModel, and `ensure_model` reuses a model by name. Before this
    guard, `--env SD_FUSE_LAYERS=1` found the existing model, reused it, and benchmarked
    the UNFUSED torso while reporting a successful fused deploy."""
    current = {"Image": "img:a", "Environment": {"SD_ENGINE": "merged"}}
    desired = {"Image": "img:a", "Environment": {"SD_ENGINE": "merged",
                                                 "SD_FUSE_LAYERS": "1"}}
    assert create_endpoint.container_drift(current, desired) == ["Environment"]


def test_container_drift_is_empty_when_nothing_moved(create_endpoint):
    """Idempotence matters: a re-run must not delete and recreate a model that is correct,
    because deleting one that an endpoint config references is a trap for the next deploy."""
    same = {"Image": "img:a", "Environment": {"SD_ENGINE": "merged"}}
    assert create_endpoint.container_drift(same, dict(same)) == []
    # An absent ModelDataUrl and an empty one are the same deployment.
    assert create_endpoint.container_drift({**same, "ModelDataUrl": ""}, dict(same)) == []


def test_container_drift_catches_a_model_data_url(create_endpoint):
    """The model repository overlay. It is how `instance_group count` is changed without a
    rebuild, so a reused model here would measure count: 1 three times and call it a sweep."""
    current = {"Image": "img:a", "Environment": {}}
    desired = {"Image": "img:a", "Environment": {}, "ModelDataUrl": "s3://b/ig2.tar.gz"}
    assert create_endpoint.container_drift(current, desired) == ["ModelDataUrl"]


# ------------------------------------------------------------------- the tool guards


def test_every_tool_has_a_main_guard():
    """`tools/loadsweep_triton.py` called `main()` at module scope, so importing it started
    a real 5-cell load sweep against localhost:8100 and blocked for minutes -- which is what
    a test collector, a `--help` wrapper or an editor's symbol indexer does on import.

    Asserted by source inspection rather than by importing: importing the broken version is
    exactly the thing that hangs, so a test that detected it by doing so would hang too.
    """
    offenders = []
    for path in sorted((ROOT / "tools").glob("*.py")):
        text = path.read_text()
        if 'if __name__ == "__main__":' not in text:
            offenders.append(path.name)
    assert not offenders, (
        f"these tools run work at import time: {offenders}. Wrap the entry point in "
        'an `if __name__ == "__main__":` guard.')


def test_loadsweep_import_does_not_run_a_sweep(monkeypatch):
    """The behavioural half of the test above, for the one tool that was broken.

    `main()` is replaced before the module is executed is not possible (it is defined by the
    module), so instead the network client is poisoned: if import reaches `main()`, it
    builds an httpx client and this raises rather than silently running a sweep.
    """
    httpx = pytest.importorskip("httpx")
    called: list[str] = []

    def _boom(*args, **kwargs):
        called.append("AsyncClient")
        raise AssertionError("importing loadsweep_triton must not start a sweep")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    module = _load(ROOT / "tools" / "loadsweep_triton.py", "tools_loadsweep_triton")
    assert not called
    # It must still be a usable entry point, not merely inert.
    assert isinstance(module.main, types.FunctionType)
    assert callable(module.cell)
