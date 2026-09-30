from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ImageQualityReport:
    quality_score: float
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    recommendation: str = "processable"


@dataclass(frozen=True)
class ImageProcessingResult:
    original_path: Path
    processed_path: Path
    quality_report: ImageQualityReport
    was_processed: bool

