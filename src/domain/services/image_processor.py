from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.image_processing import ImageProcessingResult


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, file_path: Path) -> ImageProcessingResult:
        raise NotImplementedError

