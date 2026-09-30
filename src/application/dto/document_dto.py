from pydantic import BaseModel
from typing import List


class ProcessDocumentResponse(BaseModel):
    document_id: str
    status: str
    ocr_text: str
    average_ocr_confidence: float
    image_quality: dict
    text_detection: dict
    extraction: dict
    validation: dict


class AskDocumentResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    sources: List[str]
