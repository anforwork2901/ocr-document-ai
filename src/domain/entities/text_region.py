from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class TextRegion:
    crop_path: Path
    bbox: List[int]
    confidence: float = 1.0


@dataclass(frozen=True)
class TextDetectionResult:
    regions: List[TextRegion]

    @property
    def count(self) -> int:
        return len(self.regions)

