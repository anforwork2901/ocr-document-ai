import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "Vietnamese Document Intelligence System"
    storage_dir: Path = Path("data/uploads")
    database_path: Path = Path("data/documents.db")
    ocr_engine: str = "tesseract"
    vietocr_config: str = "vgg_seq2seq"
    vietocr_device: str = "cpu"
    llm_engine: str = "rule_based"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    vector_store: str = "memory"
    vector_store_dir: Path = Path("data/chroma")
    vector_collection_name: str = "document_chunks"
    max_upload_size_mb: int = 10
    allowed_extensions: set = field(default_factory=lambda: {".jpg", ".jpeg", ".png", ".pdf", ".txt"})


@lru_cache
def get_settings() -> Settings:
    return Settings(
        storage_dir=Path(os.getenv("APP_STORAGE_DIR", "data/uploads")),
        database_path=Path(os.getenv("APP_DATABASE_PATH", "data/documents.db")),
        ocr_engine=os.getenv("APP_OCR_ENGINE", "tesseract"),
        vietocr_config=os.getenv("APP_VIETOCR_CONFIG", "vgg_seq2seq"),
        vietocr_device=os.getenv("APP_VIETOCR_DEVICE", "cpu"),
        llm_engine=os.getenv("APP_LLM_ENGINE", "rule_based"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        vector_store=os.getenv("APP_VECTOR_STORE", "memory"),
        vector_store_dir=Path(os.getenv("APP_VECTOR_STORE_DIR", "data/chroma")),
        vector_collection_name=os.getenv("APP_VECTOR_COLLECTION", "document_chunks"),
    )
