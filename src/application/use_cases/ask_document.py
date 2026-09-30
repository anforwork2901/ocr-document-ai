from src.application.dto.document_dto import AskDocumentResponse
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.services.llm_engine import LlmEngine
from src.domain.services.vector_store import VectorStore
from src.shared.exceptions import DocumentNotFoundError


class AskDocumentUseCase:
    def __init__(
        self,
        repository: DocumentRepository,
        vector_store: VectorStore,
        llm_engine: LlmEngine,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._llm_engine = llm_engine

    def execute(self, document_id: str, question: str) -> AskDocumentResponse:
        document = self._repository.get_processing_result(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")

        sources = self._vector_store.search(document_id=document_id, query=question)
        answer = self._llm_engine.answer(question=question, context=sources)
        return AskDocumentResponse(document_id=document_id, question=question, answer=answer, sources=sources)

