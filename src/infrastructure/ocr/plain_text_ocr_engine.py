from pathlib import Path
from typing import List, Optional

from src.domain.entities.ocr import OcrLine, OcrResult
from src.domain.entities.text_region import TextRegion
from src.domain.services.ocr_engine import OcrEngine


class PlainTextOcrEngine(OcrEngine):
    """MVP OCR adapter.

    Text files are treated as already-OCRed documents so the pipeline can be
    demonstrated without heavyweight OCR dependencies.
    """

    def extract_text(self, file_path: Path, regions: Optional[List[TextRegion]] = None) -> OcrResult:
        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")
        else:
            text = (
                "MINI MART AN PHU\n"
                "Ngay ban: 12/08/2025\n"
                "Sua tuoi Vinamilk 2 x 28000\n"
                "Banh mi 1 x 15000\n"
                "Nuoc suoi 3 x 7000\n"
                "Tong cong: 92000 VND"
            )

        lines = [OcrLine(text=line.strip(), confidence=0.9) for line in text.splitlines() if line.strip()]
        return OcrResult(lines=lines)
