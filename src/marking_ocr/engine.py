from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

from .io import fingerprint, sha256

RESOURCES = Path(__file__).parent / "resources"
DEFAULT_MODEL = RESOURCES / "model.json"
ENGINE_PARAMETERS = {
    "Det.limit_side_len",
    "Det.limit_type",
    "Det.thresh",
    "Det.box_thresh",
    "Det.unclip_ratio",
    "Det.use_dilation",
    "Det.score_mode",
    "Rec.rec_img_shape",
    "Rec.rec_batch_num",
}


def profile(path: Path | None = None):
    location = (path or DEFAULT_MODEL).expanduser().resolve()
    data = json.loads(location.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported model profile schema")
    for name in ("detector", "recognizer"):
        asset = data[name]
        filename = (location.parent / asset["path"]).resolve()
        if not filename.is_file() or sha256(filename) != asset["sha256"]:
            raise ValueError(f"Missing or modified {name} weights: {filename}")
    if data.get("recognition_preprocess", "none") not in {"none", "widen1.5"}:
        raise ValueError("Unknown recognition preprocessing")
    unknown = set(data.get("engine_parameters", {})) - ENGINE_PARAMETERS
    if unknown:
        raise ValueError(f"Unsupported engine parameters: {', '.join(sorted(unknown))}")
    return location, data


def decode_image(path: Path):
    from PIL import Image, ImageOps

    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Image exceeds 20 MiB")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported")
            if min(image.size) < 16 or image.width * image.height > 20_000_000:
                raise ValueError(
                    "Image must be at least 16x16 and no larger than 20 megapixels"
                )
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")


class Engine:
    """Local detector and recognizer. Receives pixels, never a catalog or labels."""

    def __init__(self, model: Path | None = None, *, threads=4):
        import cv2
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

        location, self.profile = profile(model)
        if not (RESOURCES / "fonts/JetBrainsMono-Regular.ttf").is_file():
            raise ValueError("Bundled font is missing; reinstall the package")
        self.identity = fingerprint(self.profile)
        self.model_id = self.profile["model_id"]
        self.threads = threads
        cv2.setNumThreads(threads)
        rec = self.profile["recognizer"]
        params = {
            "Det.model_path": str(
                (location.parent / self.profile["detector"]["path"]).resolve()
            ),
            "Rec.model_path": str((location.parent / rec["path"]).resolve()),
            "Rec.ocr_version": OCRVersion(rec["ocr_version"]),
            "Rec.model_type": ModelType(rec["model_type"]),
            "Rec.lang_type": LangRec(rec["language"]),
            "Global.font_path": str(RESOURCES / "fonts/JetBrainsMono-Regular.ttf"),
            "Global.use_cls": False,
            "Global.text_score": 0.0,
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
        }
        params.update(self.profile.get("engine_parameters", {}))
        params["EngineConfig.onnxruntime.use_cuda"] = False
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
        params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
        self.engine = RapidOCR(params=params)
        self.engine._load_det_model()
        recognizer = self.engine._load_rec_model()
        if self.profile.get("recognition_preprocess") == "widen1.5":
            from rapidocr.ch_ppocr_rec import TextRecInput
            import numpy as np

            class WidenedRecognizer:
                def __getattr__(self, name):
                    return getattr(recognizer, name)

                def __call__(self, value):
                    images = (
                        [value.img] if isinstance(value.img, np.ndarray) else value.img
                    )
                    images = [
                        cv2.resize(
                            img,
                            (round(img.shape[1] * 1.5), img.shape[0]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for img in images
                    ]
                    return recognizer(
                        TextRecInput(img=images, return_word_box=value.return_word_box)
                    )

            self.engine.text_rec = WidenedRecognizer()

    def recognize(self, image):
        import numpy as np

        begin = time.perf_counter()
        output = self.engine(
            np.asarray(image)[:, :, ::-1].copy(),
            use_det=True,
            use_cls=False,
            use_rec=True,
        )
        duration = (time.perf_counter() - begin) * 1000
        texts = [] if output.txts is None else list(output.txts)
        scores = [] if output.scores is None else list(output.scores)
        boxes = [] if output.boxes is None else list(output.boxes)
        if not len(texts) == len(scores) == len(boxes):
            raise RuntimeError("OCR returned inconsistent text, score and box counts")
        candidates = []
        for text, score, polygon in zip(texts, scores, boxes, strict=True):
            coordinates = np.asarray(polygon, dtype=float)
            if (
                coordinates.shape != (4, 2)
                or not np.isfinite(coordinates).all()
                or not math.isfinite(float(score))
            ):
                raise RuntimeError("OCR returned an invalid detection")
            candidates.append(
                {
                    "raw_text": str(text),
                    "score": float(score),
                    "polygon": coordinates.tolist(),
                }
            )
        return candidates, duration
