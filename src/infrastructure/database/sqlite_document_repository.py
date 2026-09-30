import json
import sqlite3
from pathlib import Path
from typing import Optional

from src.application.use_cases.process_document import _serialize_extraction
from src.domain.entities.document import Document
from src.domain.entities.extraction import ExtractionResult, ValidationResult
from src.domain.entities.image_processing import ImageProcessingResult
from src.domain.entities.ocr import OcrResult
from src.domain.entities.text_region import TextDetectionResult
from src.domain.repositories.document_repository import DocumentRepository


class SQLiteDocumentRepository(DocumentRepository):
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_processing_result(
        self,
        document: Document,
        image_result: ImageProcessingResult,
        text_detection_result: TextDetectionResult,
        ocr_result: OcrResult,
        extraction_result: ExtractionResult,
        validation_result: ValidationResult,
    ) -> None:
        payload = {
            "document_id": document.id,
            "file_name": document.file_name,
            "content_type": document.content_type,
            "file_path": str(document.file_path),
            "processed_file_path": str(image_result.processed_path),
            "status": document.status.value,
            "created_at": document.created_at.isoformat(),
            "ocr_text": ocr_result.text,
            "average_ocr_confidence": ocr_result.average_confidence,
            "image_quality": {
                "quality_score": image_result.quality_report.quality_score,
                "blur_score": image_result.quality_report.blur_score,
                "brightness": image_result.quality_report.brightness,
                "contrast": image_result.quality_report.contrast,
                "issues": image_result.quality_report.issues,
                "recommendation": image_result.quality_report.recommendation,
                "was_processed": image_result.was_processed,
            },
            "text_detection": {
                "region_count": text_detection_result.count,
                "regions": [
                    {"bbox": region.bbox, "crop_path": str(region.crop_path), "confidence": region.confidence}
                    for region in text_detection_result.regions
                ],
            },
            "extraction": _serialize_extraction(extraction_result),
            "validation": {
                "status": validation_result.status.value,
                "errors": validation_result.errors,
                "needs_review": validation_result.needs_review,
            },
        }
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                INSERT INTO documents (id, payload)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET payload = excluded.payload
                """,
                (document.id, json.dumps(payload)),
            )

    def get_processing_result(self, document_id: str) -> Optional[dict]:
        with sqlite3.connect(self._database_path) as connection:
            row = connection.execute("SELECT payload FROM documents WHERE id = ?", (document_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def _initialize(self) -> None:
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
