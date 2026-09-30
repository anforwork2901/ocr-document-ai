from functools import lru_cache

from src.application.use_cases.ask_document import AskDocumentUseCase
from src.application.use_cases.get_document import GetDocumentUseCase
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.application.use_cases.validators import ExtractionValidator
from src.infrastructure.database.sqlite_document_repository import SQLiteDocumentRepository
from src.infrastructure.image_processing.opencv_image_processor import OpenCvImageProcessor
from src.infrastructure.llm.openai_llm_engine import OpenAiLlmEngine
from src.infrastructure.llm.rule_based_llm_engine import RuleBasedLlmEngine
from src.infrastructure.ocr.tesseract_ocr_engine import TesseractOcrEngine
from src.infrastructure.ocr.vietocr_engine import VietOcrEngine
from src.infrastructure.storage.local_file_storage import LocalFileStorage
from src.infrastructure.text_detection.opencv_text_detector import OpenCvTextDetector
from src.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore
from src.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from src.shared.config import get_settings


@lru_cache
def get_container() -> dict:
    settings = get_settings()
    repository = SQLiteDocumentRepository(settings.database_path)
    vector_store = _build_vector_store(settings)
    llm_engine = _build_llm_engine(settings)
    ocr_engine = _build_ocr_engine(settings)

    return {
        "process_document": ProcessDocumentUseCase(
            storage=LocalFileStorage(settings.storage_dir),
            image_processor=OpenCvImageProcessor(settings.storage_dir / "processed"),
            text_detector=OpenCvTextDetector(settings.storage_dir / "text_regions"),
            ocr_engine=ocr_engine,
            llm_engine=llm_engine,
            repository=repository,
            vector_store=vector_store,
            validator=ExtractionValidator(),
            allowed_extensions=settings.allowed_extensions,
        ),
        "ask_document": AskDocumentUseCase(
            repository=repository,
            vector_store=vector_store,
            llm_engine=llm_engine,
        ),
        "get_document": GetDocumentUseCase(repository=repository),
    }


def _build_ocr_engine(settings):
    if settings.ocr_engine.lower() == "vietocr":
        return VietOcrEngine(config_name=settings.vietocr_config, device=settings.vietocr_device)
    return TesseractOcrEngine()


def _build_llm_engine(settings):
    if settings.llm_engine.lower() == "openai":
        return OpenAiLlmEngine(api_key=settings.openai_api_key, model=settings.openai_model)
    return RuleBasedLlmEngine()


def _build_vector_store(settings):
    if settings.vector_store.lower() == "chroma":
        return ChromaVectorStore(
            persist_dir=settings.vector_store_dir,
            collection_name=settings.vector_collection_name,
        )
    return InMemoryVectorStore()
