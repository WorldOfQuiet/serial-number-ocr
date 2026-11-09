from __future__ import annotations

from pathlib import Path

from .io import write_json


def rectified_crop(image, polygon):
    import cv2
    import numpy as np
    from PIL import Image

    points = np.asarray(polygon, dtype=np.float32)
    width = max(
        16,
        round(
            max(
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[3]),
            )
        ),
    )
    height = max(
        16,
        round(
            max(
                np.linalg.norm(points[3] - points[0]),
                np.linalg.norm(points[2] - points[1]),
            )
        ),
    )
    matrix = cv2.getPerspectiveTransform(
        points,
        np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            np.float32,
        ),
    )
    return Image.fromarray(
        cv2.warpPerspective(
            np.asarray(image), matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
    )


def save_artifacts(image, result, destination: Path, alpha: float):
    from PIL import ImageDraw, ImageFont
    from .engine import RESOURCES

    destination.mkdir(parents=True, exist_ok=False)
    image.save(destination / "original.png")
    shaded = image.copy()
    draw = ImageDraw.Draw(shaded)
    for candidate in result["candidates"]:
        draw.polygon(
            [tuple(point) for point in candidate["polygon"]], fill=(225, 45, 40)
        )
    from PIL import Image

    overlay = Image.blend(image, shaded, alpha)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(
        str(RESOURCES / "fonts/JetBrainsMono-Regular.ttf"), max(12, image.width // 48)
    )
    for i, candidate in enumerate(result["candidates"], 1):
        points = [tuple(p) for p in candidate["polygon"]]
        draw.line(
            points + [points[0]], fill=(190, 25, 20), width=max(2, image.width // 250)
        )
        label = candidate["raw_text"]
        x, y = (
            min(p[0] for p in points),
            max(0, min(p[1] for p in points) - font.size - 6),
        )
        box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x, y), label, fill=(130, 15, 10), font=font)
        rectified_crop(image, candidate["polygon"]).save(
            destination / f"crop-{i:03d}.png"
        )
    overlay.save(destination / "overlay.png")
    # Complete per-image timing is written by the caller to the CSV after rendering.
    write_json(destination / "result.json", {k: v for k, v in result.items() if k != "total_ms"})
