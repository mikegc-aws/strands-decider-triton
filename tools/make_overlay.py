#!/usr/bin/env python3
"""Build a Triton model-repository overlay tarball, for tuning `config.pbtxt` with no rebuild.

WHY THIS EXISTS. Changing `max_batch_size` or `preferred_batch_size` is a one-line edit to
`config.pbtxt`, but the model repository is baked into the image (`COPY model_repository
/opt/ml/model`), so shipping that edit normally means a ~25-minute image build and push.
SageMaker untars a `ModelDataUrl` archive over `/opt/ml/model` -- which is precisely where
the DLC looks (`SAGEMAKER_SINGLE_MODEL_REPO=/opt/ml/model/`, /usr/bin/serve:28) -- so an
archive containing just the model repository swaps the configuration in at deploy time.
The weights are NOT in the archive: they live at /opt/strands-decider in the image and are
untouched, so this stays a few kilobytes rather than 4.6 GB.

THE TWO WAYS THIS GOES WRONG, both of which this script refuses rather than lets you ship.

1. **An archive with no `model.py`.** The overlay REPLACES the repository; it does not merge
   into it. A tarball carrying only `decider/config.pbtxt` leaves the python backend with no
   model to execute, and Triton fails to load. `--config-only` is therefore not an option:
   `decider/1/model.py` is always included.

2. **A `model.py` newer than the image's `strands_decider`.** This is the expensive one. The
   newest published image (`v23-triton-onepass`, pushed 2026-10-07 11:05) predates
   `fuse_layers`, `cuda_graphs` and `max_rows`, so a model.py that passes one of those as a
   kwarg dies in `initialize()` with

       TypeError: load_merged_engine() got an unexpected keyword argument

   Triton then never reports ready, `/ping` never passes, and SageMaker fails the endpoint
   ~31 MINUTES later with "did not pass the ping health check". That happened once and cost
   a deploy and ~$5. `model.py::_engine_kwargs` is the fix -- it filters optional kwargs
   against the installed signature -- and `--target-image-predates` below is the second
   belt: it refuses to build an overlay whose parameters REQUIRE a newer package than the
   image you are about to deploy it against.

Usage:
    tools/make_overlay.py --max-batch-size 16 --bucket my-bucket --upload
    tools/make_overlay.py --max-batch-size 16 --preferred-batch-size 8,16 --out /tmp/o.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "model_repository"
MODEL = "decider"

# Parameters that the image's `strands_decider` must already understand. Keyed by the
# config.pbtxt parameter name, valued by the default that is safe against ANY image --
# `model.py::_engine_kwargs` drops an unsupported option silently only when it still holds
# its default, and raises otherwise. So "equal to the default" is exactly the test for
# "safe against an older image".
NEWER_THAN_V23 = {
    "SD_MAX_ROWS": "128",
    "SD_FUSE_LAYERS": "0",
    "SD_CUDA_GRAPHS": "0",
}


def set_scalar(text: str, key: str, value: str) -> str:
    """Replace a top-level `key: value` line in config.pbtxt.

    A targeted regex rather than a protobuf parse, because the comments in `config.pbtxt`
    are the file's main content -- they carry every measurement behind every number -- and
    round-tripping through a parser would discard all of them. Anchored at the start of a
    line so the many mentions of `max_batch_size` inside those comments are not rewritten.
    """
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"{key} not found as a top-level field in config.pbtxt")
    return pattern.sub(f"{key}: {value}", text, count=1)


def set_list(text: str, key: str, values: list[int]) -> str:
    """Replace a `key: [ a, b ]` line, e.g. `preferred_batch_size`."""
    pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*\[.*?\]\s*$", re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"{key} not found as a list field in config.pbtxt")
    body = ", ".join(str(v) for v in values)
    return pattern.sub(lambda m: f"{m.group(1)}{key}: [ {body} ]", text, count=1)


def set_parameter(text: str, key: str, value: str) -> str:
    """Replace the `string_value` of an existing `parameters` entry."""
    pattern = re.compile(
        r'(key:\s*"' + re.escape(key) + r'"\s*\n\s*value:\s*\{\s*string_value:\s*")'
        r'[^"]*(")')
    if not pattern.search(text):
        raise SystemExit(
            f"parameter {key!r} not found in config.pbtxt. Add it to the `parameters` "
            "block first -- this script edits existing keys rather than inventing them, so "
            "that a typo cannot become a silently ignored setting.")
    return pattern.sub(lambda m: m.group(1) + value + m.group(2), text, count=1)


def build_config(max_batch_size: int | None, preferred: list[int] | None,
                 max_rows: int | None, instance_count: int | None) -> str:
    text = (REPO / MODEL / "config.pbtxt").read_text()
    if max_batch_size is not None:
        text = set_scalar(text, "max_batch_size", str(max_batch_size))
    if preferred is not None:
        text = set_list(text, "preferred_batch_size", preferred)
    if max_rows is not None:
        text = set_parameter(text, "SD_MAX_ROWS", str(max_rows))
    if instance_count is not None:
        # `count` is indented inside `instance_group`, so it needs the in-group form.
        text = set_scalar_in_group(text, "count", str(instance_count))
    return text


def set_scalar_in_group(text: str, key: str, value: str) -> str:
    """Replace an indented `key: value` inside a block such as `instance_group`."""
    pattern = re.compile(rf"^(\s+){re.escape(key)}:\s*\d+\s*$", re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"{key} not found as an indented field in config.pbtxt")
    return pattern.sub(lambda m: f"{m.group(1)}{key}: {value}", text, count=1)


def requires_newer_image(config_text: str) -> list[str]:
    """Which parameters in this config an image older than this commit cannot honour.

    Reported rather than guessed at deploy time: every one of these makes
    `model.py::_engine_kwargs` RAISE on an older package (deliberately -- see its
    docstring), and a raise in `initialize()` surfaces as a ping-health-check failure 31
    minutes after you asked. Knowing it before the upload is worth the ten lines.
    """
    needs = []
    for key, safe_default in NEWER_THAN_V23.items():
        match = re.search(
            r'key:\s*"' + re.escape(key) + r'"\s*\n\s*value:\s*\{\s*string_value:\s*"'
            r'([^"]*)"', config_text)
        if match and match.group(1).strip() != safe_default:
            needs.append(f"{key}={match.group(1)}")
    return needs


def validate(config_text: str, questions: int) -> None:
    """The batch-geometry rule, checked before a deploy rather than after a benchmark.

    `max_batch_size x typical questions <= max_rows`. Overshooting it does not fail at
    runtime -- the engine CHUNKS -- so the only symptom is a throughput number that is worse
    than the narrower setting for no visible reason. That is exactly what `max_batch_size:
    32` against `max_rows: 128` produced: Triton formed batches of 26 requests (182 rows),
    chunked them into two pass pairs, and measured 56 decisions/s against 101.5.
    """
    mbs = re.search(r"^max_batch_size:\s*(\d+)", config_text, re.MULTILINE)
    rows = re.search(r'key:\s*"SD_MAX_ROWS"\s*\n\s*value:\s*\{\s*string_value:\s*"(\d+)"',
                     config_text)
    if not (mbs and rows):
        return
    mbs, rows = int(mbs.group(1)), int(rows.group(1))
    used = mbs * questions
    verdict = (f"[overlay] geometry: max_batch_size {mbs} x {questions} questions "
               f"= {used} rows against max_rows {rows} ({100 * used // rows}% used)")
    if used > rows:
        print(verdict)
        raise SystemExit(
            f"[overlay] REFUSED: {used} rows exceeds max_rows {rows}, so Triton would "
            f"assemble a batch of {mbs} and the engine would chunk it into "
            f"{-(-used // rows)} pass pairs anyway -- paying the queueing delay for no "
            "amortisation. Raise SD_MAX_ROWS (needs an image built from this commit) or "
            "lower --max-batch-size. Pass --allow-chunking to measure it deliberately.")
    print(verdict)


def make_tarball(config_text: str) -> bytes:
    """`decider/config.pbtxt` + `decider/1/model.py`, gzipped, deterministic.

    Deterministic (fixed mtime, fixed uid/gid, sorted order) so the same settings produce
    the same bytes and therefore the same S3 key and the same model name -- which is what
    makes a re-run of a sweep cell reuse the deployment it already validated instead of
    building a new one that differs only by timestamp.
    """
    buf = io.BytesIO()
    model_py = (REPO / MODEL / "1" / "model.py").read_bytes()
    entries = [
        (f"{MODEL}/config.pbtxt", config_text.encode()),
        # Always included: the overlay REPLACES the repository, so an archive without this
        # leaves the python backend with nothing to run. See the module docstring.
        (f"{MODEL}/1/model.py", model_py),
    ]
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
        for name, payload in sorted(entries):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-batch-size", type=int)
    ap.add_argument("--preferred-batch-size", default="",
                    help="comma-separated, e.g. '8' or '4,8'")
    ap.add_argument("--max-rows", type=int,
                    help="SD_MAX_ROWS. Needs an image built from this commit or later")
    ap.add_argument("--instance-count", type=int, help="instance_group count")
    ap.add_argument("--questions", type=int, default=7,
                    help="typical questions per request, for the geometry check")
    ap.add_argument("--allow-chunking", action="store_true",
                    help="build even if max_batch_size x questions exceeds max_rows")
    ap.add_argument("--out", default="", help="write the tarball here")
    ap.add_argument("--bucket", default="", help="S3 bucket to upload to")
    ap.add_argument("--prefix", default="decider-overlays")
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()

    preferred = ([int(v) for v in args.preferred_batch_size.split(",") if v.strip()]
                 if args.preferred_batch_size else None)
    config_text = build_config(args.max_batch_size, preferred, args.max_rows,
                               args.instance_count)

    if not args.allow_chunking:
        validate(config_text, args.questions)

    needs = requires_newer_image(config_text)
    if needs:
        print(f"[overlay] WARNING: {', '.join(needs)} requires an image built from this "
              "commit or later. Against an older image (v23-triton-onepass and before) "
              "model.py REFUSES to start rather than serve a setting you did not ask for, "
              "and SageMaker reports that as a ping health-check failure ~31 minutes in.")

    blob = make_tarball(config_text)
    digest = hashlib.sha256(blob).hexdigest()[:12]
    print(f"[overlay] {len(blob)} bytes, sha256 {digest}")

    if args.out:
        Path(args.out).write_bytes(blob)
        print(f"[overlay] wrote {args.out}")

    if args.bucket:
        import boto3

        key = f"{args.prefix}/overlay-{digest}.tar.gz"
        boto3.client("s3", region_name=args.region).put_object(
            Bucket=args.bucket, Key=key, Body=blob)
        uri = f"s3://{args.bucket}/{key}"
        print(f"[overlay] uploaded {uri}")
        print(f"[overlay] --model-data-url {uri}")
    elif not args.out:
        print("[overlay] nothing written: pass --out and/or --bucket", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
