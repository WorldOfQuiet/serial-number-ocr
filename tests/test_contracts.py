import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from marking_ocr.io import ResultWriter, Sample, load_samples
from marking_ocr.metrics import summarize
from marking_ocr.policy import decode_serial, select


def candidate(text, score=0.99):
    return {
        "raw_text": text,
        "score": score,
        "polygon": [[0, 0], [100, 0], [100, 30], [0, 30]],
    }


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "marking_ocr", *map(str, args)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "command",
    [[], ["predict"], ["benchmark"], ["calibrate"], ["dataset"], ["model-info"]],
)
def test_help_is_available(command):
    result = invoke(*command, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert not result.stderr


def test_serial_decoder_is_explicit_and_keeps_raw():
    candidates = [candidate("J0123456")]
    result = select(candidates, decoder="serial")
    assert (result["raw_text"], result["text"], result["format_decoded"]) == (
        "J0123456",
        "JO123456",
        True,
    )
    assert result["reason"] == "uncalibrated"
    assert candidates[0]["raw_text"] == "J0123456"
    assert decode_serial("JO12O456") == "JO12O456"
    assert decode_serial("J01234567") == "J01234567"


def test_two_decodable_candidates_remain_ambiguous():
    result = select(
        [candidate("00123456"), candidate("AB123456")], decoder="serial", threshold=0.5
    )
    assert result["raw_text"] == "AB123456"
    assert result["text"] == ""
    assert result["reason"] == "ambiguous"
    assert result["decision"] == "review"


def test_catalog_is_exact_and_does_not_rewrite_text():
    catalog = {"AB123456": {"item_code": "one", "status": "active"}}
    result = select(
        [candidate("AB123458")], catalog=catalog, threshold=0.5, decoder="serial"
    )
    assert result["text"] == "AB123458"
    assert result["reason"] == "unknown_serial"


def test_blocked_catalog_entry_is_never_accepted():
    result = select(
        [candidate("AB123456")],
        catalog={"AB123456": {"item_code": "one", "status": "blocked"}},
        threshold=0.1,
    )
    assert result["reason"] == "blocked_item"


def test_failed_readable_input_remains_in_denominator():
    samples = [
        Sample("a", Path("a.png"), "a.png", expected_text="JO123456"),
        Sample("b", Path("b.png"), "b.png", expected_text="AB123456"),
    ]
    prediction = {
        **select([candidate("J0123456")], decoder="serial"),
        "inference_ms": 1.0,
        "total_ms": 2.0,
        "error": "",
    }
    metrics = summarize(samples, [prediction, {"error": "decode failed"}])
    assert metrics["counts"]["readable"] == 2
    assert metrics["exact_match"] == 0.5
    assert metrics["raw_exact_match"] == 0
    assert metrics["character_error_rate"] == 0.5


def test_relative_paths_labels_and_duplicate_detection(tmp_path):
    (tmp_path / "inputs.csv").write_text(
        "sample_id,image,expected_text,split\na,image.png,AB123456,test\n"
    )
    row = load_samples(tmp_path / "inputs.csv", "test", labeled=True)[0]
    assert row.path == tmp_path / "image.png"
    assert row.expected_text == "AB123456"
    (tmp_path / "inputs.csv").write_text(
        "sample_id,image\na,image.png\nb,./image.png\n"
    )
    with pytest.raises(ValueError, match="Duplicate image"):
        load_samples(tmp_path / "inputs.csv")


def test_existing_results_are_preserved(tmp_path):
    output = tmp_path / "results.csv"
    output.write_text("keep me")
    with pytest.raises(ValueError, match="Output exists"):
        ResultWriter(output)
    assert output.read_text() == "keep me"


def test_atomic_csv_quotes_values_and_rolls_back_on_failure(tmp_path):
    output = tmp_path / "results.csv"
    with ResultWriter(output) as writer:
        writer.write({"sample_id": "a", "raw_text": "A, B\nC"})
    with output.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["raw_text"] == "A, B\nC"
    before = output.read_bytes()
    with pytest.raises(RuntimeError):
        with ResultWriter(output, True) as writer:
            writer.write({"sample_id": "b"})
            raise RuntimeError("model failed")
    assert output.read_bytes() == before


@pytest.mark.parametrize("alpha", ["nan", "inf", "-0.1", "1.1"])
def test_invalid_alpha_fails_before_loading_models(tmp_path, alpha):
    result = invoke(
        "predict",
        "--input",
        tmp_path / "missing.png",
        "--output",
        tmp_path / "out.csv",
        "--overlay-alpha",
        alpha,
    )
    assert result.returncode == 2
    assert "between 0 and 1" in result.stderr
    assert not (tmp_path / "out.csv").exists()


def test_unknown_split_is_an_error(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image,expected_text,split\nx.png,AB123456,test\n")
    result = invoke(
        "benchmark",
        "--input",
        manifest,
        "--split",
        "missing",
        "--output",
        tmp_path / "out.csv",
    )
    assert result.returncode == 2
    assert "No matching images" in result.stderr


def test_nan_policy_is_rejected_before_processing(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image,expected_text\nx.png,AB123456\n")
    policy = tmp_path / "policy.json"
    policy.write_text('{"schema_version":1,"threshold":NaN}')
    result = invoke(
        "benchmark",
        "--input",
        manifest,
        "--output",
        tmp_path / "out.csv",
        "--policy",
        policy,
    )
    assert result.returncode == 2
    assert "Policy threshold" in result.stderr
    assert not (tmp_path / "out.csv").exists()


def test_failed_blank_expected_is_not_counted_correct():
    metrics = summarize(
        [Sample("x", Path("x.png"), "x.png", expected_text="")],
        [{"status": "error", "error": ""}],
    )
    assert metrics["counts"]["failed"] == 1
    assert metrics["exact_match"] == 0
    assert metrics["raw_exact_match"] == 0


def test_model_profile_cannot_be_overwritten_by_predictions(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {"detector": {"path": "det.onnx"}, "recognizer": {"path": "rec.onnx"}}
        )
    )
    source = tmp_path / "inputs.csv"
    source.write_text("image\nx.png\n")
    before = profile.read_bytes()
    result = invoke(
        "predict",
        "--input",
        source,
        "--model",
        profile,
        "--output",
        profile,
        "--overwrite",
    )
    assert result.returncode == 2
    assert "must not overwrite" in result.stderr
    assert profile.read_bytes() == before


def test_model_paths_cannot_be_overridden_after_checksum_validation(tmp_path):
    from marking_ocr.engine import profile
    from marking_ocr.io import sha256

    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"checksum-only fixture")
    spec = {
        "schema_version": 1,
        "model_id": "test",
        "detector": {"path": "weights.onnx", "sha256": sha256(weights)},
        "recognizer": {"path": "weights.onnx", "sha256": sha256(weights)},
        "engine_parameters": {"Rec.model_path": "other.onnx"},
    }
    location = tmp_path / "model.json"
    location.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Unsupported engine parameters"):
        profile(location)
