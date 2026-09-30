from pathlib import Path
from typing import List

from src.domain.entities.text_region import TextDetectionResult, TextRegion
from src.domain.services.text_detector import TextDetector
from src.shared.exceptions import ProcessingDependencyError


class OpenCvTextDetector(TextDetector):
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def detect(self, file_path: Path) -> TextDetectionResult:
        if file_path.suffix.lower() == ".txt":
            return TextDetectionResult(regions=[])

        try:
            import cv2
        except ImportError as exc:
            raise ProcessingDependencyError("OpenCV is required for text region detection.") from exc

        image = cv2.imread(str(file_path))
        if image is None:
            return TextDetectionResult(regions=[])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )
        projection_boxes = self._detect_boxes_by_projection(binary, image.shape[1], image.shape[0], cv2)
        if projection_boxes:
            regions = self._write_crops(image, projection_boxes, file_path.stem, cv2)
            return TextDetectionResult(regions=regions)

        kernel_width = max(25, image.shape[1] // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
        connected = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = self._filter_and_sort_boxes(contours, image.shape[1], image.shape[0], cv2)
        regions = self._write_crops(image, boxes, file_path.stem, cv2)
        return TextDetectionResult(regions=regions)

    def _detect_boxes_by_projection(self, binary, image_width: int, image_height: int, cv2) -> List[List[int]]:
        row_counts = [cv2.countNonZero(binary[row : row + 1, :]) for row in range(image_height)]
        row_threshold = max(18, int(image_width * 0.025))
        groups = self._group_active_indices(row_counts, row_threshold, max_gap=4)

        boxes = []
        for y1, y2 in groups:
            if y2 - y1 < max(8, int(image_height * 0.01)):
                continue
            if y2 - y1 > int(image_height * 0.18):
                continue

            segment = binary[y1:y2, :]
            col_counts = [cv2.countNonZero(segment[:, col : col + 1]) for col in range(image_width)]
            col_groups = self._group_active_indices(col_counts, threshold=1, max_gap=12)
            col_groups = [group for group in col_groups if group[1] - group[0] > 4]
            if not col_groups:
                continue

            x1 = min(group[0] for group in col_groups)
            x2 = max(group[1] for group in col_groups)
            if x2 - x1 < max(20, int(image_width * 0.08)):
                continue

            pad_x = 8
            pad_y = 6
            boxes.append(
                [
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(image_width, x2 + pad_x),
                    min(image_height, y2 + pad_y),
                ]
            )

        return sorted(boxes, key=lambda box: (box[1], box[0]))

    def _group_active_indices(self, counts: List[int], threshold: int, max_gap: int) -> List[List[int]]:
        groups = []
        start = None
        last_active = None

        for index, count in enumerate(counts):
            if count > threshold:
                if start is None:
                    start = index
                last_active = index
            elif start is not None and last_active is not None and index - last_active > max_gap:
                groups.append([start, last_active + 1])
                start = None
                last_active = None

        if start is not None and last_active is not None:
            groups.append([start, last_active + 1])

        return groups

    def _filter_and_sort_boxes(self, contours, image_width: int, image_height: int, cv2) -> List[List[int]]:
        boxes = []
        min_width = max(20, int(image_width * 0.08))
        min_height = max(8, int(image_height * 0.008))
        max_height = int(image_height * 0.25)

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width < min_width or height < min_height or height > max_height:
                continue
            pad_x = max(4, int(width * 0.02))
            pad_y = max(3, int(height * 0.25))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image_width, x + width + pad_x)
            y2 = min(image_height, y + height + pad_y)
            boxes.append([x1, y1, x2, y2])

        merged = self._merge_overlapping_boxes(boxes)
        return sorted(merged, key=lambda box: (box[1], box[0]))

    def _merge_overlapping_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
        merged = [boxes[0]]
        for box in boxes[1:]:
            previous = merged[-1]
            vertically_close = box[1] <= previous[3] + 6
            if vertically_close:
                previous[0] = min(previous[0], box[0])
                previous[1] = min(previous[1], box[1])
                previous[2] = max(previous[2], box[2])
                previous[3] = max(previous[3], box[3])
            else:
                merged.append(box)
        return merged

    def _write_crops(self, image, boxes: List[List[int]], stem: str, cv2) -> List[TextRegion]:
        regions = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_path = self._output_dir / f"{stem}_line_{index + 1:03d}.png"
            cv2.imwrite(str(crop_path), crop)
            regions.append(TextRegion(crop_path=crop_path, bbox=box, confidence=1.0))
        return regions
