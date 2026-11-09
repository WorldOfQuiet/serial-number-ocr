"""Render a static recognition trace from saved marking-ocr output.

Install docs/figure-requirements.txt to rebuild. This script does not run or
change the recognizer; it lays out the original pixels and recorded values.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image
from matplotlib.patches import Polygon

from figure_style import GREEN, INK, LINE, MUTED, arrow, box, canvas, photo, save, step, text

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record_path = DOCS / "demo-source" / "prediction.json"
    result = json.loads(record_path.read_text())
    input_path = ROOT / result["input"]
    assert digest(input_path) == result["input_sha256"]
    crop_path = DOCS / "demo-source" / "crop.png"
    original = np.asarray(Image.open(input_path).convert("RGB"))
    crop = np.asarray(Image.open(crop_path).convert("RGB"))
    assert result["candidate_count"] == 1

    fig, ax = canvas()
    text(ax, 80, 39, "MARKING-OCR  /  RECOGNITION TRACE", 12, GREEN, mono=True)
    text(ax, 80, 75, "Serial number OCR", 34, weight="bold")
    text(ax, 80, 132, "Text detection     →     Rectified crop     →     Recognition     →     Exact catalog lookup", 15, MUTED)
    ax.plot([80, 1720], [173, 173], color=LINE, linewidth=1)

    step(ax, 80, 208, "01", "Detected region")
    step(ax, 850, 208, "02", "Rectified crop")
    box(ax, 80, 250, 690, 490)
    left, top, scale = photo(ax, original, 100, 290, 650, 350)
    points = np.asarray(result["candidates"][0]["polygon"]) * scale + [left, top]
    ax.add_patch(Polygon(points, closed=True, fill=False, edgecolor=GREEN, linewidth=2.2, zorder=4))
    text(ax, points[0][0], points[0][1] - 28, "region 01", 11, GREEN, mono=True,
         bbox={"facecolor":"#f5f6f3", "edgecolor":"none", "pad":3})
    text(ax, 110, 680, "label.png  ·  synthetic development sample", 12, MUTED, mono=True)

    box(ax, 850, 250, 870, 220)
    photo(ax, crop, 880, 283, 810, 150)
    arrow(ax, (788, 474), (833, 365))
    arrow(ax, (1285, 482), (1285, 532))

    step(ax, 850, 515, "03", "Recognized serial")
    box(ax, 850, 553, 870, 187)
    text(ax, 880, 586, result["text"], 38, INK, mono=True)
    ax.plot([1430, 1430], [584, 660], color=LINE, linewidth=1)
    text(ax, 1460, 587, "recognizer score", 11, MUTED)
    text(ax, 1460, 617, f"{result['score']:.5f}", 22, GREEN, mono=True)
    agreement = "raw_text = text  ·  no text correction" if result["raw_text"] == result["text"] else f"raw_text: {result['raw_text']}"
    text(ax, 880, 694, agreement, 13, MUTED, mono=True)

    ax.plot([80, 1720], [813, 813], color=LINE, linewidth=1)
    for x, label, value in [
        (80, "format", "valid" if result["format_valid"] else "invalid"),
        (640, "catalog", result["catalog_status"]),
        (1200, "item_code", result["item_code"]),
    ]:
        text(ax, x, 843, label, 13, MUTED, mono=True)
        text(ax, x, 875, value, 30, GREEN, mono=True)
    box(ax, 80, 962, 1640, 63, fill="#e9eee7", edge="#e9eee7", radius=9)
    command = "marking-ocr predict --input examples/images/label.png --output results.csv --catalog examples/catalog.csv --artifacts-dir regions --overlay-alpha 0.25"
    text(ax, 102, 984, "$ " + command, 12, INK, mono=True)
    text(ax, 80, 1050, "PP-OCRv6 detector + English PP-OCRv5 recognizer · CPU · Recognizer score is not a correctness probability.", 11, MUTED)
    text(ax, 80, 1078, "Synthetic input and illustrative catalog. The figure preserves the recorded text, region coordinates and lookup result.", 11, MUTED)
    save(fig, DOCS / "images" / "recognition-demo")
    provenance = {"input_sha256": digest(input_path), "prediction_record_sha256": digest(record_path),
                  "crop_sha256": digest(crop_path), "raw_text": result["raw_text"], "text": result["text"],
                  "score": result["score"], "catalog_status": result["catalog_status"], "synthetic": True,
                  "changes": "Layout, image scaling and recorded polygon outline only; no prediction values changed"}
    (DOCS / "images" / "recognition-demo.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
