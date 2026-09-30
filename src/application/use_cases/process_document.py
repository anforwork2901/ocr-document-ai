from pathlib import Path

from src.application.dto.document_dto import ProcessDocumentResponse
from src.application.use_cases.validators import ExtractionValidator
from src.domain.entities.document import Document, DocumentStatus
from src.domain.entities.extraction import ExtractedField, ExtractionResult, ValidationResult
from src.domain.entities.image_processing import ImageProcessingResult
from src.domain.entities.ocr import OcrResult
from src.domain.entities.text_region import TextDetectionResult
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.services.file_storage import FileStorage
from src.domain.services.image_processor import ImageProcessor
from src.domain.services.llm_engine import LlmEngine
from src.domain.services.ocr_engine import OcrEngine
from src.domain.services.text_detector import TextDetector
from src.domain.services.vector_store import VectorStore
from src.shared.exceptions import InvalidFileError


class ProcessDocumentUseCase:
    def __init__(
        self,
        storage: FileStorage,
        image_processor: ImageProcessor,
        text_detector: TextDetector,
        ocr_engine: OcrEngine,
        llm_engine: LlmEngine,
        repository: DocumentRepository,
        vector_store: VectorStore,
        validator: ExtractionValidator,
        allowed_extensions: set,
    ) -> None:
        self._storage = storage
        self._image_processor = image_processor
        self._text_detector = text_detector
        self._ocr_engine = ocr_engine
        self._llm_engine = llm_engine
        self._repository = repository
        self._vector_store = vector_store
        self._validator = validator
        self._allowed_extensions = allowed_extensions

    def execute(self, file_name: str, content_type: str, content: bytes) -> ProcessDocumentResponse:
        self._validate_file(file_name, content)
        file_path = self._storage.save(file_name=file_name, content=content)
        document = Document(file_name=file_name, content_type=content_type, file_path=file_path)

        image_result = self._image_processor.process(file_path)
        text_detection_result = self._text_detector.detect(image_result.processed_path)
        ocr_result = self._ocr_engine.extract_text(image_result.processed_path, text_detection_result.regions)
        extraction_result = self._llm_engine.extract(ocr_result)
        validation_result = self._validator.validate(extraction_result)
        document.status = DocumentStatus.PROCESSED

        self._repository.save_processing_result(
            document=document,
            image_result=image_result,
            text_detection_result=text_detection_result,
            ocr_result=ocr_result,
            extraction_result=extraction_result,
            validation_result=validation_result,
        )
        self._vector_store.index(document_id=document.id, text=ocr_result.text)

        return self._to_response(
            document.id,
            document.status.value,
            image_result,
            text_detection_result,
            ocr_result,
            extraction_result,
            validation_result,
        )

    def _validate_file(self, file_name: str, content: bytes) -> None:
        extension = Path(file_name).suffix.lower()
        if extension not in self._allowed_extensions:
            raise InvalidFileError(f"Unsupported file type: {extension}")
        if not content:
            raise InvalidFileError("Uploaded file is empty")

    def _to_response(
        self,
        document_id: str,
        status: str,
        image_result: ImageProcessingResult,
        text_detection_result: TextDetectionResult,
        ocr_result: OcrResult,
        extraction_result: ExtractionResult,
        validation_result: ValidationResult,
    ) -> ProcessDocumentResponse:
        return ProcessDocumentResponse(
            document_id=document_id,
            status=status,
            ocr_text=ocr_result.text,
            average_ocr_confidence=round(ocr_result.average_confidence, 4),
            image_quality={
                "quality_score": image_result.quality_report.quality_score,
                "blur_score": image_result.quality_report.blur_score,
                "brightness": image_result.quality_report.brightness,
                "contrast": image_result.quality_report.contrast,
                "issues": image_result.quality_report.issues,
                "recommendation": image_result.quality_report.recommendation,
                "was_processed": image_result.was_processed,
            },
            text_detection={
                "region_count": text_detection_result.count,
                "regions": [
                    {"bbox": region.bbox, "crop_path": str(region.crop_path), "confidence": region.confidence}
                    for region in text_detection_result.regions
                ],
            },
            extraction=_serialize_extraction(extraction_result),
            validation={
                "status": validation_result.status.value,
                "errors": validation_result.errors,
                "needs_review": validation_result.needs_review,
            },
        )


def _serialize_extraction(result: ExtractionResult) -> dict:
    return {
        "document_type": result.document_type,
        "fields": {name: _serialize_field(field) for name, field in result.fields.items()},
    }


def _serialize_field(field: ExtractedField) -> dict:
    return {
        "value": field.value,
        "source_text": field.source_text,
        "confidence": field.confidence.value,
        "needs_review": field.needs_review,
    }
