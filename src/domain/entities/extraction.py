from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class FieldConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationStatus(str, Enum):
    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class ExtractedField:
    value: Any
    source_text: str
    confidence: FieldConfidence
    needs_review: bool = False


@dataclass(frozen=True)
class ExtractionResult:
    document_type: str
    fields: Dict[str, ExtractedField]


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    errors: List[str] = field(default_factory=list)
    needs_review: List[str] = field(default_factory=list)
