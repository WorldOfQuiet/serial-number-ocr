# Third-party resources

| Resource | Source and terms |
| --- | --- |
| PP-OCRv6 small detector | PaddleOCR/Baidu model, ONNX conversion by RapidAI. Apache-2.0. [Official model card](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx). |
| English PP-OCRv5 mobile recognizer | PaddleOCR/Baidu model, ONNX conversion by RapidAI. Apache-2.0. [Official model card](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec). |
| RapidOCR 3.10.0 | RapidOCR Authors, Apache-2.0. [Project](https://github.com/RapidAI/RapidOCR). |
| ONNX Runtime | Microsoft, MIT. [Project](https://github.com/microsoft/onnxruntime). |
| JetBrains Mono | JetBrains s.r.o., SIL OFL 1.1. [Project](https://github.com/JetBrains/JetBrainsMono). |
| DejaVu Sans Mono | Bitstream Vera/DejaVu licenses; full copyright notice included. [Project](https://dejavu-fonts.github.io/). |
| Liberation Mono | Liberation Fonts contributors, SIL OFL 1.1; full package notice included. [Project](https://github.com/liberationfonts/liberation-fonts). |
| Synthetic images and catalog | Procedurally rendered examples and fictional product records. |

Model URLs, versions and SHA-256 hashes are stored in
`src/marking_ocr/resources/model.json`. Weights and embedded character lists are
unmodified. The horizontal crop expansion is part of this application's image
preprocessing; the optional serial decoder is separate from the model output.

Apache notices, the [RapidOCR model attribution notice](licenses/RapidOCR-model-notice.md),
and the recognition model card are in `licenses/`. Font files and their notices
are in `src/marking_ocr/resources/fonts/`. The repository's MIT license applies
to its application code, not to third-party weights or fonts.
