from pathlib import Path

from src.domain.entities.image_processing import ImageProcessingResult, ImageQualityReport
from src.domain.services.image_processor import ImageProcessor
from src.shared.exceptions import ProcessingDependencyError


class OpenCvImageProcessor(ImageProcessor):
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, file_path: Path) -> ImageProcessingResult:
        if file_path.suffix.lower() == ".txt":
            return ImageProcessingResult(
                original_path=file_path,
                processed_path=file_path,
                quality_report=ImageQualityReport(quality_score=1.0, recommendation="text_input_skipped"),
                was_processed=False,
            )

        try:
            import cv2
        except ImportError as exc:
            raise ProcessingDependencyError(
                "OpenCV is required for image preprocessing. Install opencv-python-headless."
            ) from exc

        image = cv2.imread(str(file_path))
        if image is None:
            return ImageProcessingResult(
                original_path=file_path,
                processed_path=file_path,
                quality_report=ImageQualityReport(
                    quality_score=0.0,
                    issues=["unreadable_image"],
                    recommendation="needs_better_scan",
                ),
                was_processed=False,
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        issues = self._detect_issues(blur_score, brightness, contrast)
        quality_score = self._score_quality(blur_score, brightness, contrast)
        recommendation = (
            "needs_better_scan"
            if quality_score < 0.45
            else "processable_with_warning"
            if issues
            else "processable"
        )

        if not self._needs_preprocessing(issues):
            return ImageProcessingResult(
                original_path=file_path,
                processed_path=file_path,
                quality_report=ImageQualityReport(
                    quality_score=quality_score,
                    blur_score=blur_score,
                    brightness=brightness,
                    contrast=contrast,
                    issues=issues,
                    recommendation=recommendation,
                ),
                was_processed=False,
            )

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        enhanced = cv2.equalizeHist(denoised)
        processed_path = self._output_dir / f"{file_path.stem}_processed.png"
        cv2.imwrite(str(processed_path), enhanced)

        return ImageProcessingResult(
            original_path=file_path,
            processed_path=processed_path,
            quality_report=ImageQualityReport(
                quality_score=quality_score,
                blur_score=blur_score,
                brightness=brightness,
                contrast=contrast,
                issues=issues,
                recommendation=recommendation,
            ),
            was_processed=True,
        )

    def _detect_issues(self, blur_score: float, brightness: float, contrast: float) -> list:
        issues = []
        if blur_score < 80:
            issues.append("blurry_image")
        if brightness < 70:
            issues.append("too_dark")
        if brightness > 210:
            issues.append("too_bright")
        if contrast < 35:
            issues.append("low_contrast")
        return issues

    def _needs_preprocessing(self, issues: list) -> bool:
        if "too_bright" in issues:
            return False
        return any(issue in issues for issue in ["blurry_image", "too_dark", "low_contrast"])

    def _score_quality(self, blur_score: float, brightness: float, contrast: float) -> float:
        blur_component = min(blur_score / 300, 1.0)
        brightness_component = 1.0 - min(abs(brightness - 128) / 128, 1.0)
        contrast_component = min(contrast / 80, 1.0)
        return round((blur_component * 0.45) + (brightness_component * 0.25) + (contrast_component * 0.30), 4)
