from abc import ABC, abstractmethod
from typing import List


class VectorStore(ABC):
    @abstractmethod
    def index(self, document_id: str, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, document_id: str, query: str, limit: int = 3) -> List[str]:
        raise NotImplementedError
