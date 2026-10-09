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


@pytest.fixture(scope="module")
def make_overlay():
    return _load(ROOT / "tools" / "make_overlay.py", "tools_make_overlay")


@pytest.fixture(scope="module")
def bench_tickets():
    pytest.importorskip("boto3")
    return _load(ROOT / "tools" / "bench_tickets.py", "tools_bench_tickets")


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


def test_overlay_policy_separates_object_and_bucket_grants(up):
    """`bucket/prefix/*` and `bucket` are different ARNs for different actions, and
    conflating them is the usual way this policy ends up either broken or far too broad.
    GetObject needs the key wildcard; ListBucket needs the bucket itself."""
    policy = up.overlay_read_policy("my-bucket")
    objects = [s for s in policy["Statement"] if "s3:GetObject" in s["Action"]]
    assert objects[0]["Resource"] == "arn:aws:s3:::my-bucket/decider-overlays/*"
    listing = [s for s in policy["Statement"] if s["Action"] == "s3:ListBucket"]
    assert listing[0]["Resource"] == "arn:aws:s3:::my-bucket"
    assert listing[0]["Condition"]["StringLike"]["s3:prefix"] == "decider-overlays/*"


def test_overlay_policy_grants_no_write_and_no_wildcard_bucket(up):
    """The role only ever reads an overlay. A write grant would let a compromised endpoint
    replace the model configuration it is about to load."""
    policy = up.overlay_read_policy("my-bucket")
    actions = {a for s in policy["Statement"] for a in
               ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])}
    assert not {a for a in actions if "Put" in a or "Delete" in a}
    assert "arn:aws:s3:::*" not in str(policy)


# --------------------------------------------------------------------- instance pools
#
# The point of pools is that six consecutive `InsufficientInstanceCapacity` deploys cost
# ~31 minutes EACH before surfacing -- over three hours for no information. SageMaker falls
# back across up to five types inside one CreateEndpoint instead. None of that is testable
# without an account; what is testable is the request body, which is where the two ways to
# get this wrong live.


def test_pool_priority_comes_from_list_order(create_endpoint):
    """Priority 1 is tried first, so the written order must be the attempted order."""
    pools = create_endpoint.parse_instance_pools(
        "ml.g6e.4xlarge,ml.g6e.2xlarge,ml.g6e.xlarge")
    assert pools == [
        {"InstanceType": "ml.g6e.4xlarge", "Priority": 1},
        {"InstanceType": "ml.g6e.2xlarge", "Priority": 2},
        {"InstanceType": "ml.g6e.xlarge", "Priority": 3},
    ]


def test_pool_spec_tolerates_whitespace(create_endpoint):
    pools = create_endpoint.parse_instance_pools(" ml.g6e.4xlarge , ml.g6e.2xlarge ")
    assert [p["InstanceType"] for p in pools] == ["ml.g6e.4xlarge", "ml.g6e.2xlarge"]


def test_more_than_five_pools_is_refused(create_endpoint):
    """SageMaker's limit is 5. Refused locally so the error names the limit instead of
    arriving as a server-side validation failure."""
    with pytest.raises(SystemExit, match="at most 5"):
        create_endpoint.parse_instance_pools(",".join(f"ml.g6e.{n}xlarge"
                                                      for n in range(1, 7)))


def test_duplicate_pools_are_refused_not_deduplicated(create_endpoint):
    """A repeated type burns one of five fallback slots on a type that already failed, and
    the likeliest way to write one is a typo in a longer list. Silently de-duplicating
    would leave the operator believing they had more fallbacks than they do."""
    with pytest.raises(SystemExit, match="repeats"):
        create_endpoint.parse_instance_pools("ml.g6e.4xlarge,ml.g6e.2xlarge,ml.g6e.4xlarge")


def test_empty_pool_spec_is_refused(create_endpoint):
    with pytest.raises(SystemExit, match="no instance types"):
        create_endpoint.parse_instance_pools(" , ")


def test_pools_replace_instance_type_rather_than_accompany_it(create_endpoint):
    """The API treats them as alternatives -- "you replace the InstanceType parameter in
    your production variant with an InstancePools list" -- and sending both is a validation
    error. Getting this wrong costs a deploy attempt, so it is asserted directly."""
    pools = create_endpoint.parse_instance_pools("ml.g6e.4xlarge,ml.g6e.2xlarge")
    body = create_endpoint.production_variant("m", "ml.g6.xlarge", "AllTraffic", pools)
    assert "InstanceType" not in body, "InstancePools and InstanceType are mutually exclusive"
    assert body["InstancePools"] == pools


def test_pools_bound_the_provisioning_wait(create_endpoint):
    """The whole value here is a FAST failure: capacity that has not appeared in 15 minutes
    across five types is not about to. Without this the pooled attempt can still burn the
    same ~31 minutes the serial attempts did."""
    pools = create_endpoint.parse_instance_pools("ml.g6e.4xlarge")
    body = create_endpoint.production_variant("m", "ml.g6.xlarge", "AllTraffic", pools)
    timeout = body["VariantInstanceProvisionTimeoutInSeconds"]
    assert 300 <= timeout <= 3600, "outside the range the API accepts"
    assert timeout < 1800, "a pooled attempt must fail faster than the serial one did"


def test_without_pools_the_variant_is_unchanged(create_endpoint):
    """The flag is additive: the default single-type path must not grow pool-only keys,
    which an older botocore would reject outright."""
    body = create_endpoint.production_variant("m", "ml.g6.xlarge", "AllTraffic", None)
    assert body["InstanceType"] == "ml.g6.xlarge"
    assert "InstancePools" not in body
    assert "VariantInstanceProvisionTimeoutInSeconds" not in body


def test_instance_pools_support_is_detected_from_the_service_model(create_endpoint):
    """Checked against the bundled service model rather than a version string, because the
    question is about the bytes on this machine. botocore validates client-side, so a stale
    SDK rejects the request with a misleading 'unknown parameter' that reads like the
    feature does not exist in SageMaker."""
    assert isinstance(create_endpoint.instance_pools_supported(), bool)


def test_pooled_capacity_takes_the_best_quota_across_pools(create_endpoint, monkeypatch,
                                                           capsys):
    """Best, not sum (the sum is a ceiling that may not exist, since one variant's
    instances are not guaranteed to spread) and not minimum (that refuses headroom the
    preferred type actually has)."""
    quotas = {"ml.g6e.4xlarge": 1.0, "ml.g6.xlarge": 4.0}
    monkeypatch.setattr(create_endpoint, "endpoint_usage_quota",
                        lambda region, itype: quotas.get(itype))
    pools = create_endpoint.parse_instance_pools("ml.g6e.4xlarge,ml.g6.xlarge")
    assert create_endpoint.resolve_pooled_max_capacity("us-west-2", pools, 8) == 4
    assert "ml.g6e.4xlarge=1" in capsys.readouterr().out


def test_pooled_capacity_of_one_warns_that_update_endpoint_cannot_work(create_endpoint,
                                                                      monkeypatch, capsys):
    """Quota 1 means blue/green has nowhere to put the new fleet, so `UpdateEndpoint` is
    refused and the endpoint keeps serving the old config -- i.e. the deploy looks like it
    did nothing. Worth saying out loud at the moment the ceiling is discovered."""
    monkeypatch.setattr(create_endpoint, "endpoint_usage_quota", lambda region, itype: 1.0)
    pools = create_endpoint.parse_instance_pools("ml.g6e.4xlarge")
    assert create_endpoint.resolve_pooled_max_capacity("us-west-2", pools, 4) == 1
    assert "UpdateEndpoint" in capsys.readouterr().out


def test_unreadable_pool_quotas_do_not_block_a_deploy(create_endpoint, monkeypatch):
    monkeypatch.setattr(create_endpoint, "endpoint_usage_quota", lambda region, itype: None)
    pools = create_endpoint.parse_instance_pools("ml.g6e.4xlarge")
    assert create_endpoint.resolve_pooled_max_capacity("us-west-2", pools, 3) == 3


# ----------------------------------------------------------- the model-repository overlay
#
# `config.pbtxt` lives inside the image, so tuning batch geometry would normally mean a
# ~25-minute rebuild. An overlay tarball swaps it in at deploy time. Two failure modes,
# both asserted: an archive without `model.py` leaves the python backend with nothing to
# run, and a geometry that exceeds `max_rows` chunks silently instead of failing.


def test_overlay_always_carries_model_py(make_overlay):
    """The overlay REPLACES /opt/ml/model rather than merging into it, so an archive of
    just config.pbtxt leaves Triton with no python model and the load fails."""
    import io
    import tarfile

    blob = make_overlay.make_tarball(make_overlay.build_config(16, [8, 16], None, None))
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        names = sorted(tar.getnames())
    assert names == ["decider/1/model.py", "decider/config.pbtxt"]


def test_overlay_is_byte_identical_for_identical_settings(make_overlay):
    """Determinism is what makes a re-run of a sweep cell reuse the deployment it already
    validated, instead of creating one that differs only by timestamp."""
    first = make_overlay.make_tarball(make_overlay.build_config(16, None, None, None))
    second = make_overlay.make_tarball(make_overlay.build_config(16, None, None, None))
    assert first == second


def test_overlay_rewrites_max_batch_size(make_overlay):
    text = make_overlay.build_config(16, None, None, None)
    assert "\nmax_batch_size: 16\n" in text


def test_overlay_leaves_the_comments_intact(make_overlay):
    """The comments in config.pbtxt carry every measurement behind every number in it, so
    the editor is a targeted regex rather than a protobuf round-trip."""
    text = make_overlay.build_config(16, None, None, None)
    assert "WHY TRITON IS HERE AT ALL" in text
    assert "measured 4,474 MiB per stub" in text or "4,474 MiB" in text


def test_overlay_does_not_rewrite_max_batch_size_inside_comments(make_overlay):
    """`max_batch_size` appears many times in the prose above the field. Only the
    top-level field may be rewritten; a greedy substitution would corrupt the rationale."""
    text = make_overlay.build_config(12, None, None, None)
    assert text.count("\nmax_batch_size: 12\n") == 1
    assert "Raising this to 32 was" in text


def test_overlay_rewrites_preferred_batch_size_list(make_overlay):
    text = make_overlay.build_config(None, [8], None, None)
    assert "preferred_batch_size: [ 8 ]" in text


def test_overlay_rewrites_max_rows_parameter(make_overlay):
    text = make_overlay.build_config(None, None, 256, None)
    assert 'string_value: "256"' in text


def test_overlay_refuses_a_geometry_that_would_chunk(make_overlay):
    """`max_batch_size: 32` against `max_rows: 128` is 224 rows: Triton assembles the wide
    batch, the engine splits it into two pass pairs anyway, and the queueing delay buys no
    amortisation. MEASURED at 56 decisions/s against 101.5, p50 611 -> 2,031 ms. It does
    not fail at runtime, so it has to be refused here or not at all."""
    text = make_overlay.build_config(32, None, None, None)
    with pytest.raises(SystemExit, match="exceeds max_rows"):
        make_overlay.validate(text, questions=7)


def test_overlay_accepts_a_geometry_that_fits(make_overlay):
    """16 x 7 = 112 rows against 128. The interesting half of the sweep needs no rebuild."""
    make_overlay.validate(make_overlay.build_config(16, None, None, None), questions=7)


def test_overlay_geometry_check_follows_a_raised_row_budget(make_overlay):
    """Raising both together is the documented remedy, so the check must accept it."""
    make_overlay.validate(make_overlay.build_config(32, None, 256, None), questions=7)


def test_overlay_flags_settings_an_older_image_cannot_honour(make_overlay):
    """`SD_MAX_ROWS != 128` needs a package newer than v23-triton-onepass. model.py
    deliberately REFUSES to start in that case, which SageMaker reports as a ping
    health-check failure ~31 minutes later -- so it is worth knowing before the upload."""
    needs = make_overlay.requires_newer_image(
        make_overlay.build_config(None, None, 256, None))
    assert needs == ["SD_MAX_ROWS=256"]


def test_overlay_at_shipped_defaults_needs_no_rebuild(make_overlay):
    """The corollary, and the reason a geometry sweep is affordable at all: changing only
    config.pbtxt fields works against the image that already exists."""
    assert make_overlay.requires_newer_image(
        make_overlay.build_config(16, [8, 16], None, None)) == []


# --------------------------------------------------------------- the load generator
#
# Every throughput number from a load generator is really `min(server capacity, client
# capacity)` and the two look identical in the output. This project has already published
# one figure that measured a 4-vCPU client rather than the endpoint.


def test_shard_threads_sum_to_the_offered_concurrency(bench_tickets):
    """The sum IS the concurrency the cell reports. Rounding every shard up would offer
    more load than the label claims -- c=10 over 4 processes would really run 12."""
    for conc in range(1, 40):
        for procs in (1, 2, 3, 4, 8, 16):
            shards = bench_tickets.split_threads(conc, procs)
            assert sum(shards) == conc, (conc, procs, shards)
            assert all(t >= 1 for t in shards), (conc, procs, shards)


def test_shards_never_outnumber_threads(bench_tickets):
    """An empty shard pays process startup, issues nothing, and makes --processes look
    like it changed the result when all it changed was the number of idle children."""
    assert bench_tickets.split_threads(1, 16) == [1]
    assert len(bench_tickets.split_threads(4, 16)) == 4


def test_shard_threads_are_balanced(bench_tickets):
    """Within one, so no single process is the bottleneck for the whole cell."""
    shards = bench_tickets.split_threads(10, 4)
    assert max(shards) - min(shards) <= 1
    assert sorted(shards) == [2, 2, 3, 3]


def test_cpu_sampler_reports_a_usable_percentage(bench_tickets):
    """`loadgen_cpu_pct` is the evidence that a headline number is not client-bound, so it
    must be populated on every platform -- there is no /proc on macOS, where the harness's
    own rusage is the fallback."""
    sampler = bench_tickets.CpuSampler()
    reading = sampler.read()
    assert reading["loadgen_cpu_pct"] is not None
    assert 0.0 <= reading["loadgen_cpu_pct"] <= 100.0 * (reading["loadgen_vcpu"] + 1)
    assert reading["loadgen_vcpu"] >= 1


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
