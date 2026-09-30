from fastapi import FastAPI, File, HTTPException, UploadFile

from src.presentation.api.dependencies import get_container
from src.presentation.api.schemas import QuestionRequest
from src.shared.config import get_settings
from src.shared.exceptions import DocumentNotFoundError, InvalidFileError, ProcessingDependencyError

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents")
async def process_document(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    use_case = get_container()["process_document"]
    try:
        result = use_case.execute(
            file_name=file.filename or "uploaded_document",
            content_type=file.content_type or "application/octet-stream",
            content=content,
        )
    except InvalidFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProcessingDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _model_to_dict(result)


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> dict:
    use_case = get_container()["get_document"]
    try:
        return use_case.execute(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/documents/{document_id}/questions")
def ask_document(document_id: str, request: QuestionRequest) -> dict:
    use_case = get_container()["ask_document"]
    try:
        result = use_case.execute(document_id=document_id, question=request.question)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProcessingDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _model_to_dict(result)


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
