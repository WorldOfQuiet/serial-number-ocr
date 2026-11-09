from __future__ import annotations

import math


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentiles(values):
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    ordered = sorted(values)

    def value(q):
        index = (len(ordered) - 1) * q
        lower, upper = math.floor(index), math.ceil(index)
        return (
            ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
            if lower != upper
            else ordered[lower]
        )

    return {"p50": value(0.5), "p95": value(0.95), "mean": sum(values) / len(values)}


def summarize(samples, predictions):
    if len(samples) != len(predictions):
        raise ValueError(
            "Every input must have one prediction, including failed inputs"
        )
    counts = {
        "inputs": len(samples),
        "processed": 0,
        "failed": 0,
        "readable": 0,
        "exact": 0,
        "raw_exact": 0,
        "character_edits": 0,
        "raw_character_edits": 0,
        "characters": 0,
        "accepted": 0,
        "accepted_labeled": 0,
        "wrong_accepted": 0,
        "valid_serial_inputs": 0,
        "valid_serial_exact": 0,
        "valid_serial_raw_exact": 0,
        "format_decoded": 0,
    }
    groups, inference, total = {}, [], []
    import re
    from .policy import SERIAL_PATTERN, select

    for sample, result in zip(samples, predictions, strict=True):
        error = result.get("status") in {"error", "skipped"} or bool(
            result.get("error")
        )
        counts["failed" if error else "processed"] += 1
        text = "" if error else result["text"]
        # Evaluate the raw-only candidate selection independently of decoding.
        raw = "" if error else select(result["candidates"], decoder="raw")["raw_text"]
        counts["format_decoded"] += bool(result.get("format_decoded"))
        group = groups.setdefault(
            sample.scenario,
            {"inputs": 0, "readable": 0, "exact": 0, "raw_exact": 0, "failed": 0},
        )
        group["inputs"] += 1
        group["failed"] += error
        if sample.expected_text is not None:
            counts["readable"] += 1
            counts["exact"] += not error and text == sample.expected_text
            counts["raw_exact"] += not error and raw == sample.expected_text
            counts["characters"] += len(sample.expected_text)
            counts["character_edits"] += edit_distance(text, sample.expected_text)
            counts["raw_character_edits"] += edit_distance(raw, sample.expected_text)
            group["readable"] += 1
            group["exact"] += not error and text == sample.expected_text
            group["raw_exact"] += not error and raw == sample.expected_text
            if re.fullmatch(SERIAL_PATTERN, sample.expected_text):
                counts["valid_serial_inputs"] += 1
                counts["valid_serial_exact"] += (
                    not error and text == sample.expected_text
                )
                counts["valid_serial_raw_exact"] += (
                    not error and raw == sample.expected_text
                )
        if not error:
            inference.append(result["inference_ms"])
            total.append(result["total_ms"])
        if result.get("decision") == "accept":
            counts["accepted"] += 1
            if sample.eligible is not None:
                counts["accepted_labeled"] += 1
                counts["wrong_accepted"] += (
                    not sample.eligible or text != sample.expected_text
                )
    return {
        "counts": counts,
        "exact_match": ratio(counts["exact"], counts["readable"]),
        "raw_exact_match": ratio(counts["raw_exact"], counts["readable"]),
        "character_error_rate": ratio(counts["character_edits"], counts["characters"]),
        "raw_character_error_rate": ratio(
            counts["raw_character_edits"], counts["characters"]
        ),
        "valid_serial_exact_match": ratio(
            counts["valid_serial_exact"], counts["valid_serial_inputs"]
        ),
        "valid_serial_raw_exact_match": ratio(
            counts["valid_serial_raw_exact"], counts["valid_serial_inputs"]
        ),
        "acceptance_coverage": ratio(counts["accepted"], counts["inputs"]),
        "acceptance_error": ratio(counts["wrong_accepted"], counts["accepted_labeled"]),
        "inference_ms": percentiles(inference),
        "total_ms": percentiles(total),
        "per_scenario": {
            k: {**v, "exact_match": ratio(v["exact"], v["readable"])}
            for k, v in groups.items()
        },
        "raw_normalization": "none",
        "failure_policy": "Failed readable inputs count as empty predictions.",
    }
