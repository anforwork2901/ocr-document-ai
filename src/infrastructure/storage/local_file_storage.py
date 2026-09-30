from pathlib import Path
from uuid import uuid4

from src.domain.services.file_storage import FileStorage


class LocalFileStorage(FileStorage):
    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, file_name: str, content: bytes) -> Path:
        suffix = Path(file_name).suffix.lower()
        path = self._storage_dir / f"{uuid4()}{suffix}"
        path.write_bytes(content)
        return path

