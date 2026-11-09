"""Integration checks using the shipped ONNX weights and development examples."""

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "marking_ocr", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_real_batch_preserves_raw_and_ambiguity(tmp_path):
    output = tmp_path / "predictions.csv"
    result = invoke("predict", "--input", "examples/manifest.csv", "--output", output, "--threads", 1)
    assert result.returncode == 0, result.stderr
    rows = {row["sample_id"]: row for row in read_rows(output)}
    assert len(rows) == 5
    assert rows["label"]["raw_text"] == rows["label"]["text"] == "DX140765"
    assert rows["prefix-zero"]["raw_text"] == "J0054740"
    assert rows["prefix-zero"]["text"] == "JO054740"
    assert rows["two-numbers"]["reason"] == "ambiguous"
    assert rows["two-numbers"]["text"] == ""
    assert rows["blank"]["reason"] == "no_text"
    assert all(not row["artifact_dir"] for row in rows.values())
    assert not list(tmp_path.rglob("*.png"))


def test_real_opacity_changes_only_rendering(tmp_path):
    rows = []
    overlays = []
    for alpha in (0, 1):
        output = tmp_path / f"alpha-{alpha}.csv"
        artifacts = tmp_path / f"alpha-{alpha}"
        result = invoke("predict", "--input", "examples/images/label.png", "--output", output,
                        "--artifacts-dir", artifacts, "--overlay-alpha", alpha, "--threads", 1)
        assert result.returncode == 0, result.stderr
        rows.append(read_rows(output)[0])
        overlays.append(next(artifacts.rglob("overlay.png")))
    for key in ("raw_text", "text", "score", "candidates_json", "input_sha256", "decision"):
        assert rows[0][key] == rows[1][key]
    with Image.open(overlays[0]) as left, Image.open(overlays[1]) as right:
        assert ImageChops.difference(left, right).getbbox() is not None


def test_real_bad_image_counts_in_benchmark(tmp_path):
    manifest = tmp_path / "input.csv"
    manifest.write_text("sample_id,image,expected_text,split\n"
                        "missing,missing.png,AA123456,test\n"
                        f"valid,{ROOT / 'examples/images/label.png'},DX140765,test\n")
    output = tmp_path / "predictions.csv"
    result = invoke("benchmark", "--input", manifest, "--output", output,
                    "--split", "test", "--threads", 1, "--warmup", 0)
    assert result.returncode == 3, result.stderr
    rows = read_rows(output)
    assert len(rows) == 2 and rows[0]["status"] == "error" and rows[1]["status"] == "ok"
    import json
    metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["counts"]["readable"] == 2
    assert metrics["counts"]["failed"] == 1
    assert metrics["exact_match"] == .5
