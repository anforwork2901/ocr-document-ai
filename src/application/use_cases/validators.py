from src.domain.entities.extraction import ExtractionResult, ValidationResult, ValidationStatus


class ExtractionValidator:
    def validate(self, result: ExtractionResult) -> ValidationResult:
        errors = []
        needs_review = []

        for field_name, field in result.fields.items():
            if field.value in (None, ""):
                errors.append(f"{field_name} is empty")
            if field.needs_review or field.confidence in {"low", "medium"}:
                needs_review.append(field_name)

        if errors:
            status = ValidationStatus.INVALID
        elif needs_review:
            status = ValidationStatus.PARTIALLY_VALID
        else:
            status = ValidationStatus.VALID

        return ValidationResult(status=status, errors=errors, needs_review=needs_review)
