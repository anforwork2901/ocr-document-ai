from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    bbox: Optional[List[float]] = None


@dataclass(frozen=True)
class OcrResult:
    lines: List[OcrLine]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def average_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)
