from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.text_region import TextDetectionResult


class TextDetector(ABC):
    @abstractmethod
    def detect(self, file_path: Path) -> TextDetectionResult:
        raise NotImplementedError

