from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CSV_FIELDS = [
    "sample_id",
    "image",
    "split",
    "status",
    "raw_text",
    "selected_raw_text",
    "text",
    "format_decoded",
    "score",
    "candidate_count",
    "format_valid",
    "catalog_status",
    "item_code",
    "decision",
    "reason",
    "expected_text",
    "text_correct",
    "inference_ms",
    "total_ms",
    "input_sha256",
    "model_id",
    "artifact_dir",
    "candidates_json",
    "error",
]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    image: str
    split: str = ""
    expected_text: str | None = None
    scenario: str = "unspecified"
    eligible: bool | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def boolean(value, *, optional=False):
    if value in (None, "") and optional:
        return None
    if value in (True, 1, "1", "true", "True"):
        return True
    if value in (False, 0, "0", "false", "False"):
        return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def load_samples(
    source: Path, split: str | None = None, *, labeled=False
) -> list[Sample]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    suffix = source.suffix.lower()
    manifest = source.is_file() and suffix in {".csv", ".jsonl"}
    if (labeled or split is not None) and not manifest:
        raise ValueError(
            "A CSV or JSONL manifest is required for --split and benchmark/calibrate"
        )
    if source.is_dir():
        records = [
            {"image": str(p.relative_to(source))}
            for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        base = source
    elif manifest:
        base = source.parent
        with source.open(encoding="utf-8-sig", newline="") as stream:
            if suffix == ".csv":
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not (
                    {"image", "path"} & set(reader.fieldnames)
                ):
                    raise ValueError(
                        "Manifest requires an image column (path is also accepted)"
                    )
                records = list(reader)
            else:
                records = [json.loads(line) for line in stream if line.strip()]
    elif suffix in IMAGE_SUFFIXES:
        base = source.parent
        records = [{"image": source.name}]
    else:
        raise ValueError("Input must be a PNG/JPEG, directory, CSV or JSONL manifest")
    result, identities, paths = [], set(), set()
    for index, row in enumerate(records, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {index} must be an object")
        row_split = str(row.get("split") or "")
        if split is not None and row_split != split:
            continue
        image = row.get("image") or row.get("path")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"Missing image at manifest row {index}")
        filename = Path(image).expanduser()
        if not filename.is_absolute():
            filename = base / filename
        filename = filename.resolve()
        if filename in paths:
            raise ValueError(f"Duplicate image path: {filename}")
        paths.add(filename)
        sample_id = str(
            row.get("sample_id") or hashlib.sha256(image.encode()).hexdigest()[:16]
        )
        if sample_id in identities:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        identities.add(sample_id)
        has_label = "expected_text" in row or "ground_truth_text" in row
        if labeled and not has_label:
            raise ValueError(f"Missing expected_text column at manifest row {index}")
        expected = row.get("expected_text", row.get("ground_truth_text"))
        if expected is not None and not isinstance(expected, str):
            raise ValueError(f"expected_text must be text or null at row {index}")
        readable = boolean(row.get("readable"), optional=True)
        if readable is False or (readable is None and expected in (None, "")):
            expected = None
        if readable is True and expected is None:
            raise ValueError(f"readable=true needs expected_text at row {index}")
        result.append(
            Sample(
                sample_id,
                filename,
                image,
                row_split,
                expected,
                str(row.get("scenario") or row.get("negative_type") or "unspecified"),
                boolean(
                    row.get("eligible", row.get("eligible_for_auto_accept")),
                    optional=True,
                ),
            )
        )
    if not result:
        raise ValueError(
            f"No matching images{f' in split {split!r}' if split is not None else ''}"
        )
    return result


def output_available(path: Path, overwrite=False):
    if path.exists() and not overwrite:
        raise ValueError(f"Output exists: {path}; use --overwrite to replace it")
    if path.exists() and not path.is_file():
        raise ValueError(f"Output is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value, *, overwrite=False):
    output_available(path, overwrite)
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    atomic_text(path, data, overwrite=overwrite)


def atomic_text(path: Path, data: str, *, overwrite=False):
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(data)
        if overwrite:
            os.replace(name, path)
        else:
            os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class ResultWriter:
    """Write a complete CSV atomically; preserve per-image failure rows."""

    def __init__(self, path, overwrite=False):
        self.path = Path(path)
        self.overwrite = overwrite
        output_available(self.path, overwrite)

    def __enter__(self):
        descriptor, name = tempfile.mkstemp(
            prefix="." + self.path.name + ".", dir=self.path.parent
        )
        self.temporary = Path(name)
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.stream, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        self.writer.writeheader()
        return self

    def write(self, row):
        clean = {key: ("" if value is None else value) for key, value in row.items()}
        self.writer.writerow(clean)
        self.stream.flush()

    def __exit__(self, kind, value, traceback):
        self.stream.close()
        try:
            if kind is None:
                if self.overwrite:
                    os.replace(self.temporary, self.path)
                else:
                    os.link(self.temporary, self.path)
        finally:
            self.temporary.unlink(missing_ok=True)
