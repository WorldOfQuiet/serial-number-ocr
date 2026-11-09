from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

from . import __version__
from .io import (
    ResultWriter,
    fingerprint,
    load_samples,
    output_available,
    sha256,
    write_json,
)
from .metrics import summarize
from .policy import SERIAL_PATTERN, catalog_rows, select


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def unit_interval(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return number


def parser():
    root = argparse.ArgumentParser(
        prog="marking-ocr",
        description="Read printed serial numbers, export CSV and measure OCR quality.",
        epilog="Run marking-ocr COMMAND --help for input schemas and command options.",
    )
    root.add_argument(
        "--version", action="version", version=f"marking-ocr {__version__}"
    )
    commands = root.add_subparsers(dest="command", required=True)
    for command, description in [
        ("predict", "Recognize one image, a directory or an input manifest."),
        ("benchmark", "Evaluate a labeled manifest and record CPU timings."),
        (
            "calibrate",
            "Choose a score threshold on a separate labeled calibration split.",
        ),
    ]:
        item = commands.add_parser(
            command,
            help=description,
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog="CSV: sample_id,image,split,expected_text,readable,scenario,eligible. Paths are relative to the manifest. "
            "Only image is required for predict; benchmark requires expected_text. "
            "Empty expected_text is excluded from text accuracy unless readable=true. "
            "Calibration additionally requires eligible=true/false for every row. "
            "Exit codes: 0 success, 1 model/runtime error, 2 invalid arguments/input, 3 image failures.",
        )
        item.add_argument(
            "--input",
            type=Path,
            required=True,
            help="PNG/JPEG, recursive directory, CSV or JSONL manifest",
        )
        item.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Destination predictions CSV; policy JSON for calibrate",
        )
        item.add_argument("--split", help="Exact split value to select from a manifest")
        item.add_argument(
            "--model",
            type=Path,
            help="Local model profile JSON; defaults to the bundled profile",
        )
        item.add_argument(
            "--device",
            choices=["cpu"],
            default="cpu",
            help="ONNX Runtime execution device",
        )
        item.add_argument(
            "--threads",
            type=positive,
            default=4,
            help="CPU threads for ONNX Runtime and OpenCV",
        )
        item.add_argument(
            "--catalog",
            type=Path,
            help="Optional serial,item_code,status CSV for exact lookup",
        )
        item.add_argument(
            "--decoder",
            choices=["raw", "serial"],
            default="serial",
            help="serial resolves 0 to O only in the two-letter prefix; raw text remains in CSV",
        )
        item.add_argument(
            "--limit",
            type=positive,
            help="Process at most this many selected images; recorded in metrics",
        )
        item.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace existing CSV/JSON outputs; artifact directories must remain empty",
        )
        item.add_argument(
            "--fail-fast",
            action="store_true",
            help="Stop inference after the first image failure; record remaining inputs as skipped",
        )
        item.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress progress on stderr; keep final stdout summary",
        )
        if command != "calibrate":
            policy = item.add_mutually_exclusive_group()
            policy.add_argument(
                "--policy",
                type=Path,
                help="Calibrated policy tied to this model, decoder and catalog",
            )
            policy.add_argument(
                "--min-score",
                type=unit_interval,
                help="Manual acceptance threshold; does not imply calibration",
            )
            item.add_argument(
                "--artifacts-dir",
                type=Path,
                help="Empty/new directory for per-image overlays, detected crops and raw JSON",
            )
            item.add_argument(
                "--overlay-alpha",
                type=unit_interval,
                default=0.25,
                help="Opacity of the red region fill; does not change predictions",
            )
        if command == "benchmark":
            item.add_argument(
                "--metrics",
                type=Path,
                help="Metrics JSON path; defaults to OUTPUT.metrics.json",
            )
            item.add_argument(
                "--warmup",
                type=nonnegative,
                default=3,
                help="Warm-up inference calls excluded from latency statistics",
            )
        if command == "calibrate":
            item.add_argument(
                "--predictions",
                type=Path,
                help="Calibration predictions CSV; defaults to OUTPUT.predictions.csv",
            )
            item.add_argument(
                "--max-error",
                type=unit_interval,
                default=0.01,
                help="Maximum observed error among accepted calibration rows",
            )
            item.add_argument(
                "--min-accepted",
                type=positive,
                default=30,
                help="Minimum accepted calibration count",
            )
    info = commands.add_parser(
        "model-info",
        help="Print bundled/local model provenance and verify weight checksums",
    )
    info.add_argument("--model", type=Path)
    info.add_argument(
        "--json", action="store_true", help="Print full model profile as JSON"
    )
    dataset = commands.add_parser(
        "dataset", help="Generate the versioned synthetic evaluation dataset"
    )
    dataset.add_argument(
        "--output",
        type=Path,
        required=True,
        help="A new dataset directory; never overwritten",
    )
    dataset.add_argument(
        "--seed", type=int, default=20261109, help="Independent synthetic-label seed"
    )
    return root


def print_values(values):
    for key, value in values.items():
        if value is None:
            text = "n/a"
        elif isinstance(value, float):
            text = f"{value:.6f}"
        elif isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            text = str(value)
        print(f"{key}: {text}")


def runtime_info(threads):
    import psutil

    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == "darwin" else 1024
        memory_kind = "peak_process_rss"
    except ImportError:
        rss = psutil.Process().memory_info().rss
        memory_kind = "process_rss_at_end"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "device": "cpu",
        "threads": threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        memory_kind + "_mb": rss / (1024**2),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "rapidocr",
                "onnxruntime",
                "numpy",
                "opencv-python-headless",
                "Pillow",
            ]
        },
    }


def policy_identity(model_identity, decoder, catalog):
    return {
        "model_fingerprint": model_identity,
        "decoder": decoder,
        "catalog_sha256": sha256(catalog) if catalog else None,
        "pattern": SERIAL_PATTERN,
    }


def predict_one(sample, engine, args, catalog, threshold):
    from .engine import decode_image
    from .artifacts import save_artifacts

    begin = time.perf_counter()
    image = decode_image(sample.path)
    candidates, latency = engine.recognize(image)
    result = select(
        candidates, catalog=catalog, threshold=threshold, decoder=args.decoder
    )
    result.update(
        status="ok",
        sample_id=sample.sample_id,
        image=sample.image,
        split=sample.split,
        input_sha256=sha256(sample.path),
        model_id=engine.model_id,
        model_fingerprint=engine.identity,
        inference_ms=latency,
        error="",
        artifact_dir="",
        expected_text=sample.expected_text,
        text_correct=result["text"] == sample.expected_text
        if sample.expected_text is not None
        else None,
    )
    artifact_root = getattr(args, "artifacts_dir", None)
    if artifact_root is not None:
        safe = (
            re.sub(r"[^a-zA-Z0-9_.-]", "_", sample.sample_id)[:64].strip(".") or "image"
        )
        destination = artifact_root / (safe + "-" + fingerprint(sample.image)[:8])
        result["artifact_dir"] = str(destination)
        result["total_ms"] = (time.perf_counter() - begin) * 1000
        # Rendering uses copies of the image and never updates the OCR candidate list.
        save_artifacts(image, result, destination, args.overlay_alpha)
    result["total_ms"] = (time.perf_counter() - begin) * 1000
    result["candidates_json"] = json.dumps(
        candidates, ensure_ascii=False, separators=(",", ":")
    )
    return result


def batch(args, samples, engine, catalog, threshold, output):
    predictions, stopped = [], False
    with ResultWriter(output, args.overwrite) as writer:
        for index, sample in enumerate(samples, 1):
            try:
                if stopped:
                    raise OSError("Not processed after --fail-fast failure")
                with contextlib.redirect_stdout(sys.stderr):
                    result = predict_one(sample, engine, args, catalog, threshold)
            except Exception as exc:
                result = {
                    "sample_id": sample.sample_id,
                    "image": sample.image,
                    "split": sample.split,
                    "status": "skipped" if stopped else "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_text": "",
                    "text": "",
                    "expected_text": sample.expected_text,
                    "text_correct": False if sample.expected_text is not None else None,
                    "model_id": engine.model_id,
                    "model_fingerprint": engine.identity,
                    "candidates": [],
                    "candidates_json": "[]",
                }
                stopped = stopped or args.fail_fast
            predictions.append(result)
            writer.write(result)
            if not args.quiet and (index % 100 == 0 or index == len(samples)):
                print(f"processed: {index}/{len(samples)}", file=sys.stderr, flush=True)
    return predictions


def preflight(args):
    samples = load_samples(args.input, args.split, labeled=args.command != "predict")
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_rows(args.catalog)
    outputs = [args.output]
    if args.command == "benchmark":
        args.metrics = args.metrics or args.output.with_suffix(".metrics.json")
        outputs.append(args.metrics)
        if not any(s.expected_text is not None for s in samples):
            raise ValueError("Benchmark requires at least one readable expected_text")
    if args.command == "calibrate":
        if args.catalog is None or any(s.eligible is None for s in samples):
            raise ValueError(
                "Calibration requires --catalog and eligible=true/false on every input"
            )
        args.predictions = args.predictions or args.output.with_suffix(
            ".predictions.csv"
        )
        outputs.append(args.predictions)
        hashes = [sha256(s.path) for s in samples if s.path.is_file()]
        if len(hashes) != len(set(hashes)):
            raise ValueError("Calibration contains duplicate image contents")
    if getattr(args, "policy", None) is not None:
        policy = json.loads(args.policy.read_text())
        if policy.get("schema_version") != 1:
            raise ValueError("Unsupported policy schema")
        value = policy.get("threshold")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Policy threshold must be null or finite in [0,1]")
        if policy.get("status") not in {"calibrated", "review_only"}:
            raise ValueError("Policy status must be calibrated or review_only")
        if (policy["status"] == "review_only") != (value is None):
            raise ValueError("Policy status and threshold disagree")
        if args.command == "benchmark":
            calibration = set(policy.get("calibration_images", []))
            if any(sha256(s.path) in calibration for s in samples if s.path.is_file()):
                raise ValueError(
                    "Benchmark overlaps calibration images; use a disjoint evaluation manifest"
                )
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("Each output must use a different path")
    protected = {args.input.resolve(), *(s.path for s in samples)}
    if args.catalog:
        protected.add(args.catalog.resolve())
    if getattr(args, "policy", None):
        protected.add(args.policy.resolve())
    from .engine import DEFAULT_MODEL, RESOURCES

    model_path = (args.model or DEFAULT_MODEL).resolve()
    protected.add(model_path)
    protected.add((RESOURCES / "fonts/JetBrainsMono-Regular.ttf").resolve())
    if model_path.is_file():
        definition = json.loads(model_path.read_text())
        for key in ("detector", "recognizer"):
            protected.add((model_path.parent / definition[key]["path"]).resolve())
    for output in outputs:
        if output.resolve() in protected:
            raise ValueError(
                "An output must not overwrite an input, model, policy or catalog"
            )
        output_available(output, args.overwrite)
    artifacts = getattr(args, "artifacts_dir", None)
    if artifacts is not None:
        if artifacts.exists() and (not artifacts.is_dir() or any(artifacts.iterdir())):
            raise ValueError("--artifacts-dir must be a new or empty directory")
        artifacts.mkdir(parents=True, exist_ok=True)
    return samples, catalog


def run(args, samples, catalog):
    from .engine import Engine, decode_image

    start = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        engine = Engine(args.model, threads=args.threads)
    model_load_ms = (time.perf_counter() - start) * 1000
    identity = policy_identity(engine.identity, args.decoder, args.catalog)
    threshold, policy_mode = (
        getattr(args, "min_score", None),
        "manual" if getattr(args, "min_score", None) is not None else "uncalibrated",
    )
    if getattr(args, "policy", None) is not None:
        policy = json.loads(args.policy.read_text())
        if policy["identity"] != identity:
            raise ValueError(
                "Policy does not match the selected model, decoder or catalog"
            )
        threshold, policy_mode = policy["threshold"], policy["status"]
    warmup = getattr(args, "warmup", 0)
    warmed = 0
    if warmup:
        for sample in samples:
            try:
                image = decode_image(sample.path)
                with contextlib.redirect_stdout(sys.stderr):
                    for _ in range(warmup - warmed):
                        engine.recognize(image)
                        warmed += 1
            except Exception as exc:
                if not args.quiet:
                    print(
                        f"warmup_skipped: {sample.sample_id}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                continue
            break
    predictions = batch(
        args,
        samples,
        engine,
        catalog,
        threshold,
        args.predictions if args.command == "calibrate" else args.output,
    )
    metrics = summarize(samples, predictions)
    metrics.update(
        environment=runtime_info(args.threads),
        model_id=engine.model_id,
        identity=identity,
        input_manifest_sha256=sha256(args.input) if args.input.is_file() else None,
        evaluated_inputs_sha256=fingerprint(
            [
                {
                    "id": s.sample_id,
                    "sha256": p.get("input_sha256"),
                    "expected_text": s.expected_text,
                    "scenario": s.scenario,
                }
                for s, p in zip(samples, predictions, strict=True)
            ]
        ),
        split=args.split,
        limit=args.limit,
        model_load_ms=model_load_ms,
        warmup=warmed,
        warmup_requested=warmup,
        artifacts_saved=getattr(args, "artifacts_dir", None) is not None,
        policy_mode=policy_mode,
        decoder=args.decoder,
        benchmark_version=__version__,
    )
    if args.command == "calibrate":
        if metrics["counts"]["failed"]:
            print_values(
                {
                    "status": "incomplete",
                    "failed": metrics["counts"]["failed"],
                    "predictions_csv": str(args.predictions),
                }
            )
            return 3
        eligible_candidates = [
            (sample, result)
            for sample, result in zip(samples, predictions, strict=True)
            if result["format_valid"]
            and result["catalog_status"] == "active"
            and result["score"] is not None
        ]
        curve = []
        for candidate in sorted({result["score"] for _, result in eligible_candidates}):
            accepted = [
                (sample, result)
                for sample, result in eligible_candidates
                if result["score"] >= candidate
            ]
            wrong = sum(
                not sample.eligible or result["text"] != sample.expected_text
                for sample, result in accepted
            )
            curve.append(
                {
                    "threshold": candidate,
                    "accepted": len(accepted),
                    "wrong": wrong,
                    "observed_error": wrong / len(accepted),
                }
            )
        feasible = [
            x
            for x in curve
            if x["accepted"] >= args.min_accepted
            and x["observed_error"] <= args.max_error
        ]
        selected = (
            max(feasible, key=lambda x: (x["accepted"], x["threshold"]))
            if feasible
            else None
        )
        result = {
            "schema_version": 1,
            "identity": identity,
            "threshold": selected["threshold"] if selected else None,
            "status": "calibrated" if selected else "review_only",
            "selection": selected,
            "counts": metrics["counts"],
            "max_observed_error": args.max_error,
            "min_accepted": args.min_accepted,
            "curve": curve,
            "input_manifest_sha256": metrics["input_manifest_sha256"],
            "evaluated_inputs_sha256": metrics["evaluated_inputs_sha256"],
            "calibration_images": [p["input_sha256"] for p in predictions],
            "environment": metrics["environment"],
        }
        write_json(args.output, result, overwrite=args.overwrite)
        print_values(
            {
                "status": result["status"],
                "images": len(samples),
                "threshold": result["threshold"],
                "accepted": selected["accepted"] if selected else 0,
                "wrong_accepted": selected["wrong"] if selected else 0,
                "policy_json": str(args.output),
                "predictions_csv": str(args.predictions),
            }
        )
        return 0
    if args.command == "benchmark":
        write_json(args.metrics, metrics, overwrite=args.overwrite)
    summary = {
        "status": "complete" if not metrics["counts"]["failed"] else "incomplete",
        "images": len(samples),
        "processed": metrics["counts"]["processed"],
        "failed": metrics["counts"]["failed"],
        "device": "cpu",
        "threads": args.threads,
        "model": engine.model_id,
        "decoder": args.decoder,
        "policy_mode": policy_mode,
    }
    if args.command == "benchmark":
        summary.update(
            {
                key: metrics[key]
                for key in [
                    "raw_exact_match",
                    "exact_match",
                    "valid_serial_exact_match",
                    "character_error_rate",
                    "acceptance_coverage",
                    "acceptance_error",
                ]
            }
        )
        summary.update(
            readable_images=metrics["counts"]["readable"],
            valid_serial_images=metrics["counts"]["valid_serial_inputs"],
            metrics_json=str(args.metrics),
        )
    if len(predictions) == 1 and not predictions[0].get("error"):
        summary.update(
            {
                key: predictions[0][key]
                for key in [
                    "raw_text",
                    "text",
                    "format_decoded",
                    "score",
                    "decision",
                    "reason",
                ]
            }
        )
    summary.update(
        latency_p50_ms=metrics["inference_ms"]["p50"],
        latency_p95_ms=metrics["inference_ms"]["p95"],
        output_csv=str(args.output),
    )
    print_values(summary)
    return 3 if metrics["counts"]["failed"] else 0


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "model-info":
            from .engine import profile

            _, value = profile(args.model)
            if args.json:
                print(json.dumps(value, ensure_ascii=False, indent=2))
            else:
                print_values(
                    {
                        "model": value["model_id"],
                        "fingerprint": fingerprint(value),
                        "weights_verified": True,
                        "recognizer": value["recognizer"]["ocr_version"],
                        "language": value["recognizer"]["language"],
                        "recognition_preprocess": value["recognition_preprocess"],
                        "device": "cpu",
                    }
                )
            return 0
        if args.command == "dataset":
            from .synthesis import generate

            value = generate(args.output, args.seed)
            print_values(
                {
                    "status": "generated",
                    "output": str(args.output),
                    "seed": args.seed,
                    "protocol": value["protocol"],
                    "images": sum(1 for _ in (args.output / "images").glob("*.png")),
                    "manifest": str(args.output / "dataset.csv"),
                    "catalog": str(args.output / "catalog.csv"),
                }
            )
            return 0
        samples, catalog = preflight(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        root.error(str(exc))
    try:
        return run(args, samples, catalog)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
