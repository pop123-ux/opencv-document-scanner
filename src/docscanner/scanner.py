from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2 as cv
import numpy as np

EnhancementMode = Literal["color", "gray", "bw"]


@dataclass(frozen=True)
class DocumentScanResult:
    """Result returned by a scan operation."""

    warped: np.ndarray
    corners: np.ndarray
    enhanced: np.ndarray


class DocumentScanner:
    """Detects document boundaries and creates clean perspective scans."""

    def __init__(
        self,
        max_dimension: int = 1600,
        canny_thresholds: tuple[int, int] = (50, 150),
    ) -> None:
        self.max_dimension = max_dimension
        self.canny_thresholds = canny_thresholds

    def scan(
        self,
        image: np.ndarray,
        mode: EnhancementMode = "bw",
    ) -> DocumentScanResult:
        """Return a perspective-corrected document scan from a BGR image."""

        if image is None or image.size == 0:
            raise ValueError("image is empty")

        ratio = image.shape[0] / float(self._resize_height(image.shape))
        resized = self._resize_for_detection(image)
        corners = self._find_document_corners(resized) * ratio
        warped = four_point_transform(image, corners)
        enhanced = enhance_scan(warped, mode)
        return DocumentScanResult(warped=warped, corners=corners, enhanced=enhanced)

    def scan_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mode: EnhancementMode = "bw",
    ) -> DocumentScanResult:
        """Scan an image file and save the enhanced result."""

        input_path = Path(input_path)
        output_path = Path(output_path)
        image = cv.imread(str(input_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        result = self.scan(image, mode=mode)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv.imwrite(str(output_path), result.enhanced):
            raise OSError(f"Could not write output image: {output_path}")
        return result

    def _resize_height(self, shape: tuple[int, ...]) -> int:
        height, width = shape[:2]
        if max(height, width) <= self.max_dimension:
            return height
        scale = self.max_dimension / float(max(height, width))
        return max(1, int(height * scale))

    def _resize_for_detection(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if max(height, width) <= self.max_dimension:
            return image.copy()
        scale = self.max_dimension / float(max(height, width))
        return cv.resize(image, (int(width * scale), int(height * scale)), interpolation=cv.INTER_AREA)

    def _find_document_corners(self, image: np.ndarray) -> np.ndarray:
        edged = preprocess_for_edges(image, self.canny_thresholds)
        contours, _ = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:8]

        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv.contourArea(approx) > 1000:
                return order_points(approx.reshape(4, 2).astype("float32"))

        height, width = image.shape[:2]
        return np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )


def preprocess_for_edges(
    image: np.ndarray,
    canny_thresholds: tuple[int, int] = (50, 150),
) -> np.ndarray:
    """Return a clean edge mask for document contour detection."""

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, *canny_thresholds)
    kernel = np.ones((3, 3), np.uint8)
    return cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel, iterations=2)


def order_points(points: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""

    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a perspective transform using four ordered corner points."""

    rect = order_points(points.astype("float32"))
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(1, int(max(height_a, height_b)))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv.getPerspectiveTransform(rect, destination)
    return cv.warpPerspective(image, matrix, (max_width, max_height))


def enhance_scan(image: np.ndarray, mode: EnhancementMode = "bw") -> np.ndarray:
    """Improve readability after perspective correction."""

    if mode == "color":
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        boosted = clahe.apply(l_channel)
        return cv.cvtColor(cv.merge((boosted, a_channel, b_channel)), cv.COLOR_LAB2BGR)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if mode == "gray":
        return cv.equalizeHist(gray)
    if mode == "bw":
        return cv.adaptiveThreshold(
            gray,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            21,
            10,
        )
    raise ValueError(f"Unsupported enhancement mode: {mode}")


def image_files(paths: Iterable[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of image paths."""

    supported = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(p for p in path.rglob("*") if p.suffix.lower() in supported)
        elif path.suffix.lower() in supported:
            found.append(path)
    return sorted(found)
