import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from marking_ocr.synthesis import (
    FONTS,
    FONT_DIRECTORY,
    SERIAL_PATTERN,
    build_samples,
    generate,
    legacy_exclusions,
    protocol_definition,
    render_sample,
    validate_samples,
    _png_bytes,
)


@pytest.fixture(scope="module")
def planned():
    return build_samples()


def test_split_independence_and_declared_counts(planned):
    samples, catalog = planned
    assert Counter(row["split"] for row in samples) == {
        "calibration": 400, "test_clean": 400, "test_degraded": 400, "test_negative": 200,
    }
    assert len({row["sample_id"] for row in samples}) == 1400
    assert len({row["physical_id"] for row in samples}) == 1400
    all_texts = [text for row in samples for text in row["all_rendered_texts"]]
    assert len(all_texts) == len(set(all_texts))
    assert set(all_texts).isdisjoint(legacy_exclusions())
    assert set(row["serial"] for row in catalog).isdisjoint(legacy_exclusions())
    for row in samples:
        assert Path(row["image"]).name == row["sample_id"] + ".png"
        assert len(row["sample_id"]) == 24
        assert all(text not in row["image"] for text in row["all_rendered_texts"])
    assert validate_samples(samples, catalog)["samples"] == 1400


def test_fixed_scenarios_catalog_and_negative_labels(planned):
    samples, catalog = planned
    known = {row["serial"]: row["status"] for row in catalog}
    assert Counter(row["scenario"] for row in samples if row["split"] == "calibration") == {
        "active": 280, "unknown": 40, "blocked": 20, "invalid_format": 20,
        "blank": 10, "distractor": 10, "occluded": 10, "ambiguous": 10,
    }
    assert Counter(row["scenario"] for row in samples if row["split"] == "test_negative") == {
        "invalid_format": 50, "distractor": 40, "blank": 40, "ambiguous": 40, "occluded": 30,
    }
    for row in samples:
        scenario, texts = row["scenario"], row["all_rendered_texts"]
        assert row["eligible"] is (scenario == "active")
        if scenario in {"blank", "ambiguous", "occluded"}:
            assert row["expected_text"] == "" and row["readable"] is False
        else:
            assert row["expected_text"] == texts[0] and row["readable"] is True
        if row["split"] in {"test_clean", "test_degraded"}:
            assert scenario in {"active", "unknown", "blocked"}
            assert SERIAL_PATTERN.fullmatch(texts[0])
        for text in texts:
            if scenario in {"active", "blocked", "ambiguous", "occluded"}:
                assert known[text] == ("blocked" if scenario == "blocked" else "active")
            else:
                assert text not in known
    for split, expected_per_kind in (("calibration", 10), ("test_negative", 25)):
        invalid = [row for row in samples if row["split"] == split and row["scenario"] == "invalid_format"]
        assert Counter(row["invalid_kind"] for row in invalid) == {
            "letter_in_digits": expected_per_kind, "digit_in_prefix": expected_per_kind,
        }
        for row in invalid:
            text = row["expected_text"]
            assert not SERIAL_PATTERN.fullmatch(text)
            if row["invalid_kind"] == "letter_in_digits":
                assert text[:2].isalpha()
                assert sum(character.isalpha() for character in text[2:]) == 1
            else:
                assert sum(character.isdigit() for character in text[:2]) == 1
                assert text[2:].isdigit()


def test_duplicate_identity_text_and_catalog_leak_are_rejected(planned):
    original, catalog = planned
    samples = json.loads(json.dumps(original))
    samples[1]["physical_id"] = samples[0]["physical_id"]
    with pytest.raises(ValueError, match="physical label"):
        validate_samples(samples, catalog)
    samples = json.loads(json.dumps(original))
    active = next(row for row in samples if row["scenario"] == "active")
    unknown = next(row for row in samples if row["scenario"] == "unknown")
    active["all_rendered_texts"] = unknown["all_rendered_texts"]
    active["expected_text"] = unknown["expected_text"]
    with pytest.raises(ValueError):
        validate_samples(samples, catalog)
    extra_catalog = catalog + [{"serial": unknown["expected_text"], "status": "active"}]
    with pytest.raises(ValueError, match="must not enter catalog"):
        validate_samples(original, extra_catalog)


def test_balanced_fonts_and_recipe_groups(planned):
    samples, _ = planned
    assert Counter(row["recipe"]["family"] for row in samples if row["split"] == "test_degraded") == {
        "degraded_blur": 100, "degraded_noise": 100, "degraded_perspective": 100, "degraded_low_contrast": 100,
    }
    for split in {row["split"] for row in samples}:
        fonts = Counter(row["font"] for row in samples if row["split"] == split)
        assert set(fonts) == {font["file"] for font in FONTS}
        assert max(fonts.values()) - min(fonts.values()) <= 1
    for row in samples:
        if row["split"] == "test_clean":
            assert -5 <= row["recipe"]["angle_degrees"] <= 5
    for font in FONTS:
        assert hashlib.sha256((FONT_DIRECTORY / font["file"]).read_bytes()).hexdigest() == font["sha256"]
        assert hashlib.sha256((FONT_DIRECTORY / font["license"]).read_bytes()).hexdigest() == font["license_sha256"]


def test_manifest_roundtrip_replays_png_and_polygons(planned):
    samples, _ = planned
    representatives = {}
    for sample in samples:
        representatives.setdefault((sample["recipe"]["family"], sample["scenario"]), sample)
    for sample in representatives.values():
        image, polygons = render_sample(sample)
        saved = json.loads(json.dumps(sample))
        replay, replay_polygons = render_sample(saved)
        assert image.size == (640, 320) and image.mode == "RGB"
        assert hashlib.sha256(_png_bytes(image)).digest() == hashlib.sha256(_png_bytes(replay)).digest()
        assert polygons == replay_polygons
        assert len(polygons) == len(sample["all_rendered_texts"])
        for polygon in polygons:
            points = np.asarray(polygon)
            assert (points[:, 0] >= 0).all() and (points[:, 0] < 640).all()
            assert (points[:, 1] >= 0).all() and (points[:, 1] < 320).all()


def test_v1_reservation_includes_catalog_examples_and_old_negatives():
    reserved = legacy_exclusions()
    assert {"FH123894", "EO771437", "EZ335821", "IM945476"} <= reserved
    assert {"FH12O894", "INSPECTED 0", "INSPECTED 2799"} <= reserved
    assert sum(bool(SERIAL_PATTERN.fullmatch(text)) for text in reserved) == 5600


def test_same_seed_is_stable_and_other_seed_changes_samples(planned):
    assert build_samples() == planned
    different, _ = build_samples(20261110)
    assert different[0]["sample_id"] != planned[0][0]["sample_id"]
    assert different[0]["all_rendered_texts"] != planned[0][0]["all_rendered_texts"]
    assert "not_evaluated" == protocol_definition()["real_photos"]


def test_existing_output_is_never_overwritten(tmp_path):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate(tmp_path)
    with pytest.raises(FileExistsError):
        generate(sentinel)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("seed", [-1, True, "20261109"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        build_samples(seed)
