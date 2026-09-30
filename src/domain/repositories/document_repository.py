from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities.document import Document
from src.domain.entities.extraction import ExtractionResult, ValidationResult
from src.domain.entities.image_processing import ImageProcessingResult
from src.domain.entities.ocr import OcrResult
from src.domain.entities.text_region import TextDetectionResult


class DocumentRepository(ABC):
    @abstractmethod
    def save_processing_result(
        self,
        document: Document,
        image_result: ImageProcessingResult,
        text_detection_result: TextDetectionResult,
        ocr_result: OcrResult,
        extraction_result: ExtractionResult,
        validation_result: ValidationResult,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_processing_result(self, document_id: str) -> Optional[dict]:
        raise NotImplementedError
