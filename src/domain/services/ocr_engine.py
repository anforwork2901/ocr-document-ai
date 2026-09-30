from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from src.domain.entities.ocr import OcrResult
from src.domain.entities.text_region import TextRegion


class OcrEngine(ABC):
    @abstractmethod
    def extract_text(self, file_path: Path, regions: Optional[List[TextRegion]] = None) -> OcrResult:
        raise NotImplementedError
