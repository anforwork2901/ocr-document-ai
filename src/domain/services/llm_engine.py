from abc import ABC, abstractmethod
from typing import List

from src.domain.entities.extraction import ExtractionResult
from src.domain.entities.ocr import OcrResult


class LlmEngine(ABC):
    @abstractmethod
    def extract(self, ocr_result: OcrResult) -> ExtractionResult:
        raise NotImplementedError

    @abstractmethod
    def answer(self, question: str, context: List[str]) -> str:
        raise NotImplementedError
