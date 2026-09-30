from src.domain.repositories.document_repository import DocumentRepository
from src.shared.exceptions import DocumentNotFoundError


class GetDocumentUseCase:
    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def execute(self, document_id: str) -> dict:
        result = self._repository.get_processing_result(document_id)
        if result is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        return result

