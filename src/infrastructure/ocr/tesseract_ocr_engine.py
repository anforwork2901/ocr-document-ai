import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

from src.domain.entities.ocr import OcrLine, OcrResult
from src.domain.entities.text_region import TextRegion
from src.domain.services.ocr_engine import OcrEngine
from src.shared.exceptions import ProcessingDependencyError


class TesseractOcrEngine(OcrEngine):
    def __init__(self, language: str = "vie+eng") -> None:
        self._language = language

    def extract_text(self, file_path: Path, regions: Optional[List[TextRegion]] = None) -> OcrResult:
        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")
            lines = [OcrLine(text=line.strip(), confidence=0.9) for line in text.splitlines() if line.strip()]
            return OcrResult(lines=lines)

        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ProcessingDependencyError("pytesseract and Pillow are required for Tesseract OCR.") from exc

        full_page_result = self._extract_best_full_page_orientation(pytesseract, Image, file_path)
        if not regions:
            return full_page_result

        region_result = self._extract_from_paths(
            pytesseract,
            Image,
            [region.crop_path for region in regions],
            "--psm 7",
        )
        if region_result.average_confidence + 0.05 < full_page_result.average_confidence:
            return full_page_result
        return region_result

    def _extract_best_full_page_orientation(self, pytesseract, Image, file_path: Path) -> OcrResult:
        try:
            image = Image.open(file_path)
        except Exception as exc:
            raise ProcessingDependencyError(
                "Tesseract OCR failed. Ensure the input image can be opened."
            ) from exc

        candidates = [(0, file_path)]
        with TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            for angle in (90, 180, 270):
                rotated_path = temporary_path / f"{file_path.stem}_rotated_{angle}.png"
                image.rotate(angle, expand=True).save(rotated_path)
                candidates.append((angle, rotated_path))

            scored_results = []
            for angle, candidate_path in candidates:
                for psm in (3, 6, 11, 12):
                    result = self._extract_from_paths(pytesseract, Image, [candidate_path], f"--psm {psm}")
                    scored_results.append((self._score_result(result), angle, psm, result))

        return max(scored_results, key=lambda item: item[0])[3]

    def _extract_from_paths(self, pytesseract, Image, paths: List[Path], config: str) -> OcrResult:
        lines = []
        for path in paths:
            try:
                data = pytesseract.image_to_data(
                    Image.open(path),
                    lang=self._language,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
            except Exception as exc:
                raise ProcessingDependencyError(
                    "Tesseract OCR failed. Ensure the Tesseract binary and language data are installed."
                ) from exc
            lines.extend(self._lines_from_tesseract_data(data))
        return OcrResult(lines=lines)

    def _lines_from_tesseract_data(self, data) -> List[OcrLine]:
        lines_by_key = {}
        for index, text in enumerate(data.get("text", [])):
            cleaned = text.strip().strip("|").strip()
            if not cleaned:
                continue
            key = (
                data["page_num"][index],
                data["block_num"][index],
                data["par_num"][index],
                data["line_num"][index],
            )
            confidence = self._parse_confidence(data["conf"][index])
            lines_by_key.setdefault(key, {"words": [], "confidences": []})
            lines_by_key[key]["words"].append(cleaned)
            lines_by_key[key]["confidences"].append(confidence)

        lines = []
        for value in lines_by_key.values():
            confidence_values = value["confidences"]
            confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
            lines.append(OcrLine(text=" ".join(value["words"]), confidence=round(confidence, 4)))
        return lines

    def _parse_confidence(self, raw_confidence) -> float:
        try:
            value = float(raw_confidence)
        except (TypeError, ValueError):
            return 0.0
        if value < 0:
            return 0.0
        return min(value / 100, 1.0)

    def _score_result(self, result: OcrResult) -> float:
        text = result.text.lower()
        meaningful_chars = sum(char.isalnum() for char in text)
        receipt_terms = [
            "hóa đơn",
            "hoa don",
            "tổng",
            "tong",
            "tiền",
            "tien",
            "ngày",
            "ngay",
            "khách",
            "khach",
            "coffee",
            "mart",
            "vincommerce",
            "vinmart",
        ]
        term_bonus = sum(1 for term in receipt_terms if term in text)
        amount_patterns = len(re.findall(r"\b\d{1,3}(?:[.,]\d{3})+\b", text))
        date_patterns = len(re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text))
        line_bonus = min(len(result.lines), 30) / 30
        length_bonus = min(meaningful_chars / 250, 1.0)
        return (
            (result.average_confidence * 0.45)
            + (length_bonus * 0.20)
            + (line_bonus * 0.10)
            + (min(term_bonus, 6) * 0.04)
            + (min(amount_patterns, 6) * 0.06)
            + (min(date_patterns, 2) * 0.04)
        )
