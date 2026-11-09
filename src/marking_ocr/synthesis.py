"""A new, independently generated synthetic serial-number evaluation protocol.

This module renders images only. It never imports or runs an OCR engine, reads
predictions, or changes its recipe in response to recognition results. The old
protocol's random string pool is reconstructed to reserve it, without reading
old holdout rows. ``generate`` refuses every existing output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import re
import string
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, features


PROTOCOL = "printed-serial-independent-synthetic-v2"
DEFAULT_SEED = 20261109
SERIAL_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{6}$")
CSV_FIELDS = ("sample_id", "image", "split", "expected_text", "readable", "scenario", "eligible")
CATALOG_FIELDS = ("serial", "item_code", "name", "batch", "status")
FONT_DIRECTORY = Path(__file__).parent / "resources" / "fonts"
FONTS = (
    {"file": "JetBrainsMono-Regular.ttf", "sha256": "e6fd0d7e91550b3ed2b735d4312474362c4716edc4fc0577a0f61ed782d5aed1",
     "license": "JetBrainsMono-OFL.txt", "license_sha256": "a76abf002c49097d146e86740a3105a5d00450b1592e820a1109a8c5680cd697"},
    {"file": "DejaVuSansMono.ttf", "sha256": "a54dca07c76d6289e717e75e0a58c0128f6d7269ef3faf76417c9d7d3bba37ab",
     "license": "DejaVuSansMono-copyright.txt", "license_sha256": "63d3ba759d12804c5b31a9d5940d855c1820d1f5999e6b0872eb1c7ff045fbc9"},
    {"file": "LiberationMono-Regular.ttf", "sha256": "d18235f1e874771f4eaec3ddd6ddb4db849d334bdc6930c5c2185d9344834e62",
     "license": "LiberationMono-copyright.txt", "license_sha256": "f32df582a2aab128c07b2f7e36e7b928dc5f6bd82657b905c04f91459f9e5cf9"},
)
SPLITS = {
    "calibration": {"active": 280, "unknown": 40, "blocked": 20, "invalid_format": 20,
                    "blank": 10, "distractor": 10, "occluded": 10, "ambiguous": 10},
    "test_clean": {"active": 320, "unknown": 40, "blocked": 40},
    "test_degraded": {"active": 320, "unknown": 40, "blocked": 40},
    "test_negative": {"invalid_format": 50, "distractor": 40, "blank": 40,
                      "ambiguous": 40, "occluded": 30},
}
# Each bound is inclusive for integer draws and half-open for real draws.
# All four degraded families retain large glyphs and moderate distortions.
RECIPE_FAMILIES = {
    "clean": {"font_px": [36, 54], "angle_degrees": [-5, 5], "corner_jitter_px": 2,
              "ink": [20, 65], "blur_sigma": [0, 0.25], "sensor_noise_sd": [0, 1.5]},
    "mild_perspective": {"font_px": [34, 50], "angle_degrees": [-8, 8], "corner_jitter_px": 10,
                         "ink": [20, 70], "blur_sigma": [0, 0.3], "sensor_noise_sd": [0, 2]},
    "mild_low_contrast": {"font_px": [34, 50], "angle_degrees": [-5, 5], "corner_jitter_px": 3,
                          "ink": [115, 150], "blur_sigma": [0, 0.3], "sensor_noise_sd": [0, 2]},
    "mild_blur": {"font_px": [34, 50], "angle_degrees": [-5, 5], "corner_jitter_px": 3,
                  "ink": [20, 70], "blur_sigma": [0.5, 0.9], "sensor_noise_sd": [0, 2]},
    "degraded_blur": {"font_px": [34, 48], "angle_degrees": [-6, 6], "corner_jitter_px": 5,
                      "ink": [30, 80], "blur_sigma": [0.8, 1.3], "sensor_noise_sd": [2, 5]},
    "degraded_noise": {"font_px": [34, 48], "angle_degrees": [-6, 6], "corner_jitter_px": 5,
                       "ink": [30, 80], "blur_sigma": [0.1, 0.4], "sensor_noise_sd": [8, 14]},
    "degraded_perspective": {"font_px": [34, 48], "angle_degrees": [-9, 9], "corner_jitter_px": 18,
                             "ink": [30, 80], "blur_sigma": [0.2, 0.6], "sensor_noise_sd": [2, 5]},
    "degraded_low_contrast": {"font_px": [34, 48], "angle_degrees": [-6, 6], "corner_jitter_px": 5,
                              "ink": [125, 155], "blur_sigma": [0.2, 0.6], "sensor_noise_sd": [2, 5]},
    "negative_clear": {"font_px": [36, 52], "angle_degrees": [-5, 5], "corner_jitter_px": 3,
                       "ink": [20, 65], "blur_sigma": [0, 0.3], "sensor_noise_sd": [0, 2]},
}
SPLIT_FAMILIES = {
    "calibration": ("clean", "mild_perspective", "mild_low_contrast", "mild_blur"),
    "test_clean": ("clean",),
    "test_degraded": ("degraded_blur", "degraded_noise", "degraded_perspective", "degraded_low_contrast"),
    "test_negative": ("negative_clear",),
}


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str:
    return _digest(path.read_bytes())


def _seed(seed: int, purpose: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL}:{seed}:{purpose}".encode()).digest()[:8], "big")


def legacy_exclusions() -> set[str]:
    """Reserve every v1 serial plus a superset of its invalid/distractor strings.

    Reproduces only the old generator's initial random string loop (seed 314159,
    2800 labels, two unique serials per label); no old split or image is loaded.
    """
    rng = random.Random(314159)
    serials: set[str] = set()
    while len(serials) < 5600:
        serials.add("".join(rng.choices(string.ascii_uppercase, k=2)) + f"{rng.randrange(1_000_000):06d}")
    return serials | {s[:4] + "O" + s[5:] for s in serials} | {f"INSPECTED {i}" for i in range(2800)}


def _check_fonts() -> None:
    for font in FONTS:
        for filename, expected in ((font["file"], font["sha256"]), (font["license"], font["license_sha256"])):
            path = FONT_DIRECTORY / filename
            if not path.is_file() or _file_hash(path) != expected:
                raise ValueError(f"Missing or changed protocol resource: {filename}")


def protocol_definition(seed: int = DEFAULT_SEED) -> dict:
    """Return the fixed design, including the executable generator fingerprint."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _check_fonts()
    reserved = legacy_exclusions()
    return {
        "protocol": PROTOCOL, "seed": seed, "synthetic_only": True,
        "real_photos": "not_evaluated", "split_counts": SPLITS,
        "total_images": sum(sum(counts.values()) for counts in SPLITS.values()),
        "selection_policy": {
            "recipe_fixed_before_inference": True,
            "calibration": "Select OCR preprocessing and acceptance thresholds on calibration only.",
            "tests": "Freeze the OCR model, preprocessing, candidate rule, and thresholds before any test inference. Never select parameters on test results.",
            "immutable_outputs": "Existing output directories are refused. A revised recipe requires a new protocol and new held-out data.",
            "readability": "Readability is assigned by controlled generation, not inferred from OCR success. Ambiguous has two independently readable serials but no unique target; occluded hides central glyphs.",
            "claims": "Procedural images are not evidence of accuracy on real photographs.",
        },
        "serial_format": "^[A-Z]{2}[0-9]{6}$",
        "split_assignment": "Fixed scenario counts, independently shuffled within each split; each physical label and rendered nonempty string belongs to exactly one split.",
        "fonts": list(FONTS),
        "font_assignment": "Round-robin three fonts within each shuffled split, offset by split index; counts differ by at most one.",
        "recipe_families": RECIPE_FAMILIES,
        "split_recipe_assignment": {name: list(values) for name, values in SPLIT_FAMILIES.items()},
        "common_rendering": {
            "image_size": [640, 320], "label_size": [480, 180],
            "background_gray": [180, 218], "background_noise_sd": [1.5, 4.0],
            "label_rgb": "base in [236,249], RGB=(base,base-1,base-6)",
            "label_center_jitter_px": [-8, 8], "lighting_gradient_amplitude": [-6, 6],
            "image_format": "PNG RGB, compress_level=6, optimize=false",
            "font_layout": "Pillow BASIC; centered glyph bounds; distractor font size capped at 42; ambiguous uses two centered lines at y=51 and y=129, font size capped at 38.",
            "occlusion": "Label-colored rectangle hides the middle 27%-40% of the single rendered text width, including its full glyph height.",
            "geometry": "Rotate label corners, add independently sampled per-corner XY jitter, project label and polygons with OpenCV homography; blend a warped mask.",
            "noise": "NumPy PCG64 per-image texture seed; RGB Gaussian background and final sensor noise; explicit parameters saved in manifest.jsonl.",
            "bounds": "Integer bounds inclusive; real bounds half-open.",
        },
        "invalid_format": {
            "calibration": {"letter_in_digits": 10, "digit_in_prefix": 10},
            "test_negative": {"letter_in_digits": 25, "digit_in_prefix": 25},
            "mutation": "Uniform uppercase A-Z in one of six digit positions, or uniform digit 0-9 in one of two letter positions; exactly one mutation, alternating kinds.",
        },
        "distractors": "Unique INSPECTED, LOT, PASS, or QC OK followed by a random seven-digit number; preserved verbatim as readable expected_text.",
        "catalog": "Contains active/blocked serials and all underlying ambiguous/occluded serials. Unknown, invalid, distractor and blank have no catalog entry. Only active is eligible.",
        "legacy_exclusion": {
            "protocol": "printed-serial-synthetic-v1", "seed": 314159, "physical_label_count": 2800,
            "reserved_serial_count": 5600, "reserved_text_count": len(reserved),
            "reserved_texts_sha256": _digest(("\n".join(sorted(reserved)) + "\n").encode()),
            "method": "Replay only initial serial creation from the published v1 generator. Reserve all serials, every position-4 O mutation, and INSPECTED 0..2799. Do not read v1 holdout rows, images, or predictions.",
            "source_generator_sha256": "68a7b617f802aa737fca85e51efe22b3ae618cd7053b427afa392661d9aa2b1b",
            "source_config_sha256": "08fc11c0500a0322bf0d4d25eb43f6663deb6564e9326e8b7ca639afa3675f09",
            "source_catalog_sha256": "18dadcf2f9f4972a3c08e3f4b514531a2dbbc9e94ded0329bdac5fb79b73c972",
        },
        "files": {"all_samples": "dataset.csv", "catalog": "catalog.csv", "manifest": "manifest.jsonl",
                  "design": "recipe.json", "summary": "protocol.json",
                  "splits": {split: f"{split}.csv" for split in SPLITS}},
        "sample_csv_fields": list(CSV_FIELDS),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
                    "opencv": cv2.__version__, "freetype": features.version("freetype2"),
                    "zlib": features.version("zlib"), "machine": platform.machine()},
        "generator_sha256": _file_hash(Path(__file__)),
    }


def _sample_recipe(seed: int, family: str, scenario: str) -> dict:
    rng = np.random.default_rng(seed)
    bounds = RECIPE_FAMILIES[family]
    uniform = lambda key: float(rng.uniform(*bounds[key]))
    base = int(rng.integers(236, 250))
    size = int(rng.integers(bounds["font_px"][0], bounds["font_px"][1] + 1))
    if scenario == "ambiguous":
        size = min(size, 38)
    elif scenario == "distractor":
        size = min(size, 42)
    return {
        "family": family, "font_px": size,
        "background_gray": float(rng.uniform(180, 218)),
        "background_noise_sd": float(rng.uniform(1.5, 4.0)),
        "label_rgb": [base, base - 1, base - 6],
        "ink": int(rng.integers(bounds["ink"][0], bounds["ink"][1] + 1)),
        "angle_degrees": uniform("angle_degrees"),
        "corner_offsets_xy": rng.uniform(-bounds["corner_jitter_px"], bounds["corner_jitter_px"], (4, 2)).tolist(),
        "center_xy": (np.array([320, 160]) + rng.uniform(-8, 8, 2)).tolist(),
        "lighting_gradient_amplitude": float(rng.uniform(-6, 6)),
        "blur_sigma": uniform("blur_sigma"), "sensor_noise_sd": uniform("sensor_noise_sd"),
        "occlusion_width_fraction": float(rng.uniform(0.27, 0.40)) if scenario == "occluded" else None,
    }


def build_samples(seed: int = DEFAULT_SEED) -> tuple[list[dict], list[dict]]:
    """Plan all labels before rendering; this does not inspect images or OCR."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(_seed(seed, "strings"))
    used = legacy_exclusions()
    samples, catalog = [], []

    def serial() -> str:
        while True:
            value = "".join(rng.choices(string.ascii_uppercase, k=2)) + f"{rng.randrange(1_000_000):06d}"
            if value not in used:
                used.add(value)
                return value

    for split_index, (split, counts) in enumerate(SPLITS.items()):
        scenarios = [scenario for scenario, count in counts.items() for _ in range(count)]
        random.Random(_seed(seed, f"shuffle:{split}")).shuffle(scenarios)
        invalid_index = 0
        for split_position, scenario in enumerate(scenarios):
            index = len(samples)
            identity = _digest(f"{PROTOCOL}:{seed}:physical:{index}".encode())[:24]
            base_serial = serial()
            texts = [base_serial]
            invalid_kind = None
            if scenario == "blank":
                texts = []
            elif scenario == "ambiguous":
                texts = [base_serial, serial()]
            elif scenario == "invalid_format":
                invalid_kind = "letter_in_digits" if invalid_index % 2 == 0 else "digit_in_prefix"
                invalid_index += 1
                while True:
                    position = rng.randrange(2, 8) if invalid_kind == "letter_in_digits" else rng.randrange(2)
                    replacement = rng.choice(string.ascii_uppercase if invalid_kind == "letter_in_digits" else string.digits)
                    changed = base_serial[:position] + replacement + base_serial[position + 1:]
                    if changed not in used:
                        used.add(changed)
                        texts = [changed]
                        break
            elif scenario == "distractor":
                while True:
                    changed = rng.choice(("INSPECTED", "LOT", "PASS", "QC OK")) + f" {rng.randrange(1_000_000, 10_000_000)}"
                    if changed not in used:
                        used.add(changed)
                        texts = [changed]
                        break
            readable = scenario not in {"blank", "ambiguous", "occluded"}
            item = f"MO-{index:05d}"
            if scenario in {"active", "blocked", "ambiguous", "occluded"}:
                for text in texts:
                    catalog.append({"serial": text, "item_code": item, "name": f"Synthetic module {index % 12 + 1}",
                                    "batch": f"NEW-{index // 100 + 1:03d}",
                                    "status": "blocked" if scenario == "blocked" else "active"})
            family = SPLIT_FAMILIES[split][split_position % len(SPLIT_FAMILIES[split])]
            font = FONTS[(split_position + split_index) % len(FONTS)]
            recipe_seed = _seed(seed, f"recipe:{index}")
            samples.append({
                "sample_id": identity, "image": f"images/{identity}.png", "split": split,
                "expected_text": texts[0] if readable else "", "readable": readable,
                "scenario": scenario, "eligible": scenario == "active",
                "physical_id": f"new-{identity}", "group_id": f"new-{identity}",
                "all_rendered_texts": texts, "invalid_kind": invalid_kind,
                "font": font["file"], "font_sha256": font["sha256"],
                "recipe_seed": recipe_seed, "texture_seed": _seed(seed, f"texture:{index}"),
                "recipe": _sample_recipe(recipe_seed, family, scenario),
            })
    validate_samples(samples, catalog)
    return samples, catalog


def validate_samples(samples: list[dict], catalog: list[dict]) -> dict:
    """Enforce split, identity, text, catalog, and negative-label contracts."""
    identities, groups, text_owner = set(), set(), {}
    reserved = legacy_exclusions()
    by_serial = {row["serial"]: row for row in catalog}
    if len(by_serial) != len(catalog):
        raise ValueError("Catalog serials must be unique")
    expected_catalog = set()
    for row in samples:
        identity, group = row["sample_id"], row["physical_id"]
        if identity in identities or group in groups:
            raise ValueError("Duplicate physical label or sample identity")
        identities.add(identity)
        groups.add(group)
        if row["group_id"] != group or row["image"] != f"images/{identity}.png":
            raise ValueError("Physical group and opaque filename contract violated")
        scenario, texts = row["scenario"], row["all_rendered_texts"]
        expected_count = 0 if scenario == "blank" else (2 if scenario == "ambiguous" else 1)
        if len(texts) != expected_count:
            raise ValueError("Wrong number of rendered texts")
        for value in texts:
            if value in text_owner or value in reserved:
                raise ValueError("Text overlaps another label or the v1 reserved universe")
            text_owner[value] = row["split"]
            if scenario in {"active", "unknown", "blocked", "ambiguous", "occluded"} and not SERIAL_PATTERN.fullmatch(value):
                raise ValueError("Valid-serial scenario has an invalid serial")
            if scenario in {"invalid_format", "distractor"} and SERIAL_PATTERN.fullmatch(value):
                raise ValueError("Negative text unexpectedly matches serial format")
        readable = scenario not in {"blank", "ambiguous", "occluded"}
        if row["readable"] != readable or row["expected_text"] != (texts[0] if readable else ""):
            raise ValueError("Readability and expected_text disagree")
        if row["eligible"] != (scenario == "active"):
            raise ValueError("Only active serials may be eligible")
        if scenario in {"active", "blocked", "ambiguous", "occluded"}:
            expected_catalog.update(texts)
            for text in texts:
                if text not in by_serial or by_serial[text]["status"] != ("blocked" if scenario == "blocked" else "active"):
                    raise ValueError("Catalog status does not match scenario")
        elif any(text in by_serial for text in texts):
            raise ValueError("Unknown or negative text must not enter catalog")
    if set(by_serial) != expected_catalog:
        raise ValueError("Catalog has missing or extraneous serials")
    actual = {split: dict(Counter(row["scenario"] for row in samples if row["split"] == split)) for split in SPLITS}
    if actual != SPLITS or set(row["split"] for row in samples) != set(SPLITS):
        raise ValueError("Protocol split/scenario counts differ from the fixed design")
    for split, count in (("calibration", 10), ("test_negative", 25)):
        kinds = Counter(row["invalid_kind"] for row in samples if row["split"] == split and row["scenario"] == "invalid_format")
        if kinds != {"letter_in_digits": count, "digit_in_prefix": count}:
            raise ValueError("Invalid-format mutation balance changed")
    return {"samples": len(samples), "physical_labels": len(groups), "unique_rendered_texts": len(text_owner),
            "catalog_size": len(catalog), "split_counts": actual,
            "cross_split_text_overlap": 0, "cross_split_physical_overlap": 0,
            "legacy_reserved_text_overlap": 0}


def render_sample(sample: dict) -> tuple[Image.Image, list]:
    """Render directly from a saved manifest row, with no hidden mutable state."""
    recipe = sample["recipe"]
    font_path = FONT_DIRECTORY / sample["font"]
    if _file_hash(font_path) != sample["font_sha256"]:
        raise ValueError("Saved font hash does not match the bundled font")
    rng = np.random.default_rng(sample["texture_seed"])
    background = rng.normal(recipe["background_gray"], recipe["background_noise_sd"], (320, 640, 3))
    label = Image.new("RGB", (480, 180), tuple(recipe["label_rgb"]))
    draw = ImageDraw.Draw(label)
    font = ImageFont.truetype(str(font_path), recipe["font_px"], layout_engine=ImageFont.Layout.BASIC)
    polygons = []
    texts = sample["all_rendered_texts"]
    for index, value in enumerate(texts):
        left, top, right, bottom = draw.textbbox((0, 0), value, font=font)
        width, height = right - left, bottom - top
        center_y = 90 if len(texts) == 1 else 51 + index * 78
        x, y = (480 - width) / 2 - left, center_y - height / 2 - top
        if width > 456 or height > 60:
            raise ValueError("Text exceeds the protocol's legible label bounds")
        draw.text((x, y), value, font=font, fill=(recipe["ink"],) * 3)
        polygons.append([[x + left - 4, y + top - 4], [x + right + 4, y + top - 4],
                         [x + right + 4, y + bottom + 4], [x + left - 4, y + bottom + 4]])
        if sample["scenario"] == "occluded":
            hidden_width = width * recipe["occlusion_width_fraction"]
            draw.rectangle((240 - hidden_width / 2, y + top - 3, 240 + hidden_width / 2, y + bottom + 3),
                           fill=tuple(recipe["label_rgb"]))
    corners = np.array([[-240, -90], [240, -90], [240, 90], [-240, 90]], dtype=np.float64)
    angle = np.deg2rad(recipe["angle_degrees"])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    target = (corners @ rotation.T + recipe["center_xy"] + np.asarray(recipe["corner_offsets_xy"])).astype(np.float32)
    transform = cv2.getPerspectiveTransform(np.array([[0, 0], [479, 0], [479, 179], [0, 179]], dtype=np.float32), target)
    warped = cv2.warpPerspective(np.asarray(label), transform, (640, 320), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpPerspective(np.full((180, 480), 255, dtype=np.uint8), transform, (640, 320), flags=cv2.INTER_LINEAR).astype(np.float64)[:, :, None] / 255
    background = warped * alpha + background * (1 - alpha)
    gradient = np.linspace(-1, 1, 640) * recipe["lighting_gradient_amplitude"]
    background += gradient[None, :, None]
    if recipe["blur_sigma"] > 0:
        background = cv2.GaussianBlur(background, (0, 0), recipe["blur_sigma"])
    background += rng.normal(0, recipe["sensor_noise_sd"], background.shape)
    image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8))
    projected = [cv2.perspectiveTransform(np.array([p], dtype=np.float32), transform)[0].tolist() for p in polygons]
    return image, projected


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=6, optimize=False)
    return stream.getvalue()


def _write_csv(path: Path, rows: list[dict], fields: tuple) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(value).lower() if type(value) is bool else value for key, value in row.items()})


def generate(output: Path, seed: int = DEFAULT_SEED) -> dict:
    """Create all 1,400 samples in a new directory and return portable metadata.

    ``recipe.json`` is written before any rendering. ``protocol.json`` records
    its SHA, every manifest/CSV SHA, validation, and a one-image replay SHA.
    A partial run remains inspectable and is never overwritten by this API.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    definition = protocol_definition(seed)
    samples, catalog = build_samples(seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    recipe_bytes = _json_bytes(definition)
    (output / "recipe.json").write_bytes(recipe_bytes)
    with (output / "manifest.jsonl").open("x", encoding="utf-8", newline="") as manifest:
        for sample in samples:
            image, polygons = render_sample(sample)
            content = _png_bytes(image)
            (output / sample["image"]).write_bytes(content)
            sample["sha256"] = _digest(content)
            sample["gt_polygons"] = polygons
            manifest.write(json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    _write_csv(output / "dataset.csv", samples, CSV_FIELDS)
    for split in SPLITS:
        _write_csv(output / f"{split}.csv", [sample for sample in samples if sample["split"] == split], CSV_FIELDS)
    _write_csv(output / "catalog.csv", catalog, CATALOG_FIELDS)
    validation = validate_samples(samples, catalog)
    for sample in samples:
        if _file_hash(output / sample["image"]) != sample["sha256"]:
            raise ValueError(f"Image hash verification failed: {sample['sample_id']}")
    replay_image, replay_polygons = render_sample(json.loads(json.dumps(samples[0])))
    replay_sha = _digest(_png_bytes(replay_image))
    if replay_sha != samples[0]["sha256"] or replay_polygons != samples[0]["gt_polygons"]:
        raise ValueError("Saved recipe did not reproduce the first PNG byte-for-byte")
    files = ["recipe.json", "manifest.jsonl", "dataset.csv", "catalog.csv"] + [f"{split}.csv" for split in SPLITS]
    summary = {
        "protocol": PROTOCOL, "seed": seed, "synthetic_only": True, "real_photos": "not_evaluated",
        "recipe_sha256": _digest(recipe_bytes), "definition": definition,
        "artifacts": {filename: {"path": filename, "sha256": _file_hash(output / filename)} for filename in files},
        "validation": {**validation, "all_image_hashes_verified": True,
                       "replay": {"sample_id": samples[0]["sample_id"], "sha256": replay_sha, "identical_png_and_polygons": True}},
        "font_counts": {split: dict(Counter(row["font"] for row in samples if row["split"] == split)) for split in SPLITS},
        "recipe_counts": {split: dict(Counter(row["recipe"]["family"] for row in samples if row["split"] == split)) for split in SPLITS},
    }
    (output / "protocol.json").write_bytes(_json_bytes(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        metadata = generate(args.output, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"protocol": metadata["protocol"], "recipe_sha256": metadata["recipe_sha256"],
                      "validation": metadata["validation"]}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
