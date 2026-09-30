from pathlib import Path
from typing import List, Optional

from src.domain.entities.ocr import OcrLine, OcrResult
from src.domain.entities.text_region import TextRegion
from src.domain.services.ocr_engine import OcrEngine
from src.shared.exceptions import ProcessingDependencyError


class VietOcrEngine(OcrEngine):
    """Vietnamese OCR recognition adapter.

    VietOCR is strongest when the input image is already cropped to a text line
    or a compact text region. Full-page detection/layout should be handled by
    OpenCV, PaddleOCR, or a future document detection component before this
    adapter is used for Vietnamese recognition.
    """

    def __init__(self, config_name: str = "vgg_seq2seq", device: str = "cpu") -> None:
        self._config_name = config_name
        self._device = device
        self._predictor = None

    def extract_text(self, file_path: Path, regions: Optional[List[TextRegion]] = None) -> OcrResult:
        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")
            lines = [OcrLine(text=line.strip(), confidence=0.9) for line in text.splitlines() if line.strip()]
            return OcrResult(lines=lines)

        predictor = self._get_predictor()

        try:
            from PIL import Image
        except ImportError as exc:
            raise ProcessingDependencyError("Pillow is required for VietOCR image loading.") from exc

        target_paths = [region.crop_path for region in regions] if regions else [file_path]
        lines = []
        for target_path in target_paths:
            try:
                text = predictor.predict(Image.open(target_path))
            except Exception as exc:
                raise ProcessingDependencyError(
                    "VietOCR inference failed. Ensure VietOCR model files can be downloaded or are available locally."
                ) from exc

            cleaned = text.strip()
            if cleaned:
                lines.append(OcrLine(text=cleaned, confidence=0.75))
        return OcrResult(lines=lines)

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise ProcessingDependencyError(
                "VietOCR is not installed. Install optional OCR dependencies with: pip install -r requirements-ocr.txt"
            ) from exc

        config = Cfg.load_config_from_name(self._config_name)
        config["device"] = self._device
        config["predictor"]["beamsearch"] = False
        self._predictor = Predictor(config)
        return self._predictor
