"""The deploy scripts' decisions, with AWS stubbed out.

`deploy/` creates billable resources, so almost none of it is testable without an account.
What IS testable is the part that decides *what* to create, and three of those decisions are
worth a regression test because each has a silent failure mode:

  * the execution role's trust policy. A missing `aws:SourceAccount` condition is not
    visible in any console view that matters and cannot be detected from the outside.
  * the autoscaling ceiling against the account's real quota. Getting this wrong produces an
    endpoint that reports healthy, never scales, and records the refusal only in a scaling
    activity log (see `create_endpoint.endpoint_usage_quota`).
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
