from __future__ import annotations

import csv
import math
import re
from pathlib import Path

SERIAL_PATTERN = r"^[A-Z]{2}[0-9]{6}$"


def decode_serial(text: str) -> str:
    """Resolve zero only in the two-letter prefix; retain every other character."""
    if re.fullmatch(r"[A-Z0]{2}[0-9]{6}", text):
        return text[:2].replace("0", "O") + text[2:]
    return text


def catalog_rows(path: Path | None, pattern=SERIAL_PATTERN):
    if path is None:
        return None
    serial = re.compile(pattern)
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"serial", "item_code", "status"}.issubset(reader.fieldnames or []):
            raise ValueError(
                "Catalog requires serial,item_code,status columns; name and batch are optional"
            )
        for row in reader:
            key = row["serial"]
            if not serial.fullmatch(key) or key in result:
                raise ValueError(f"Invalid or duplicate catalog serial: {key!r}")
            if row["status"] not in {"active", "blocked"} or not row["item_code"]:
                raise ValueError(f"Invalid catalog entry: {key!r}")
            result[key] = row
    return result


def select(
    candidates, pattern=SERIAL_PATTERN, catalog=None, threshold=None, decoder="raw"
):
    if decoder not in {"raw", "serial"}:
        raise ValueError(f"Unsupported decoder: {decoder}")
    if decoder == "serial" and pattern != SERIAL_PATTERN:
        raise ValueError(
            "The serial decoder requires the default two-letters/six-digits pattern"
        )
    expression = re.compile(pattern)
    for candidate in candidates:
        if not math.isfinite(candidate["score"]) or not 0 <= candidate["score"] <= 1:
            raise ValueError("OCR returned an invalid score")
    texts = [
        decode_serial(c["raw_text"]) if decoder == "serial" else c["raw_text"]
        for c in candidates
    ]
    matching = [i for i, text in enumerate(texts) if expression.fullmatch(text)]
    chosen = (
        matching[0]
        if len(matching) == 1
        else (0 if not matching and len(candidates) == 1 else None)
    )
    candidate = candidates[chosen] if chosen is not None else None
    raw = candidate["raw_text"] if candidate else ""
    text = texts[chosen] if chosen is not None else ""
    score = candidate["score"] if candidate else None
    valid = bool(expression.fullmatch(text))
    item = catalog.get(text) if catalog is not None and valid else None
    catalog_status = (
        "not_requested" if catalog is None else (item["status"] if item else "missing")
    )
    if not candidates:
        reason = "no_text"
    elif chosen is None:
        reason = "ambiguous"
    elif not valid:
        reason = "invalid_format"
    elif catalog_status == "missing":
        reason = "unknown_serial"
    elif catalog_status == "blocked":
        reason = "blocked_item"
    elif threshold is None:
        reason = "uncalibrated"
    elif score < threshold:
        reason = "low_score"
    else:
        reason = "accepted"
    raw_selection = (
        select(candidates, pattern=pattern, decoder="raw")["raw_text"]
        if decoder == "serial"
        else raw
    )
    return {
        "raw_text": raw_selection,
        "selected_raw_text": raw,
        "text": text,
        "format_decoded": text != raw,
        "decoder": decoder,
        "score": score,
        "selected_candidate": chosen,
        "candidate_count": len(candidates),
        "format_valid": valid,
        "catalog_status": catalog_status,
        "item_code": item["item_code"] if item else "",
        "catalog_item": item,
        "decision": "accept" if reason == "accepted" else "review",
        "reason": reason,
        "candidates": candidates,
    }
