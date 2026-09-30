import json
from typing import Any, Dict, List

from src.domain.entities.extraction import ExtractedField, ExtractionResult, FieldConfidence
from src.domain.entities.ocr import OcrResult
from src.domain.services.llm_engine import LlmEngine
from src.shared.exceptions import ProcessingDependencyError


class OpenAiLlmEngine(LlmEngine):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def extract(self, ocr_result: OcrResult) -> ExtractionResult:
        if not ocr_result.text.strip():
            return ExtractionResult(document_type="unknown", fields={})

        response_text = self._create_structured_response(
            instructions=self._extraction_instructions(),
            user_input=f"OCR_TEXT:\n{ocr_result.text}",
            schema=self._extraction_schema(),
        )
        payload = json.loads(response_text)
        return self._to_extraction_result(payload)

    def answer(self, question: str, context: List[str]) -> str:
        if not context:
            return "I could not find relevant information in this document."

        response_text = self._create_text_response(
            instructions=(
                "You answer questions using only the provided document context. "
                "If the context does not contain the answer, say that the document does not provide enough information. "
                "Keep the answer concise."
            ),
            user_input=f"QUESTION:\n{question}\n\nDOCUMENT_CONTEXT:\n{chr(10).join(context)}",
        )
        return response_text.strip()

    def _get_client(self):
        if not self._api_key:
            raise ProcessingDependencyError("OPENAI_API_KEY is required when APP_LLM_ENGINE=openai.")
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProcessingDependencyError("OpenAI SDK is not installed. Run: pip install -r requirements.txt") from exc
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _create_structured_response(self, instructions: str, user_input: str, schema: Dict[str, Any]) -> str:
        client = self._get_client()
        try:
            response = client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "document_extraction_result",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            raise ProcessingDependencyError(f"OpenAI extraction request failed: {exc}") from exc
        return response.output_text

    def _create_text_response(self, instructions: str, user_input: str) -> str:
        client = self._get_client()
        try:
            response = client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
            )
        except Exception as exc:
            raise ProcessingDependencyError(f"OpenAI answer request failed: {exc}") from exc
        return response.output_text

    def _extraction_instructions(self) -> str:
        return (
            "You are a document information extraction engine for Vietnamese OCR output. "
            "Extract only information supported by the OCR text. "
            "Do not hallucinate missing values. "
            "Preserve the source text evidence for every extracted field. "
            "Use ISO 8601 date format when normalizing dates. "
            "Use confidence='low' when OCR text is noisy or ambiguous."
        )

    def _extraction_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["document_type", "fields"],
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "Examples: receipt, citizen_id, invoice, transcript, unknown.",
                },
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "value", "source_text", "confidence", "needs_review"],
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": ["string", "number", "boolean", "null"]},
                            "source_text": {"type": "string"},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "needs_review": {"type": "boolean"},
                        },
                    },
                },
            },
        }

    def _to_extraction_result(self, payload: Dict[str, Any]) -> ExtractionResult:
        fields: Dict[str, ExtractedField] = {}
        for item in payload.get("fields", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            fields[name] = ExtractedField(
                value=item.get("value"),
                source_text=str(item.get("source_text", "")),
                confidence=FieldConfidence(item.get("confidence", "low")),
                needs_review=bool(item.get("needs_review", False)),
            )
        return ExtractionResult(document_type=str(payload.get("document_type", "unknown")), fields=fields)

