from abc import ABC, abstractmethod
from pathlib import Path


class FileStorage(ABC):
    @abstractmethod
    def save(self, file_name: str, content: bytes) -> Path:
        raise NotImplementedError

