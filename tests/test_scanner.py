from pathlib import Path

import cv2 as cv
import numpy as np

from docscanner.scanner import DocumentScanner, enhance_scan, order_points


def synthetic_document() -> np.ndarray:
    image = np.full((600, 800, 3), 35, dtype=np.uint8)
    document = np.array([[180, 90], [620, 130], [570, 500], [130, 450]], dtype=np.int32)
    cv.fillConvexPoly(image, document, (245, 245, 235))
    cv.putText(image, "OpenCV Scanner", (210, 210), cv.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2)
    cv.line(image, (210, 260), (535, 290), (80, 80, 80), 3)
    cv.line(image, (200, 320), (520, 350), (80, 80, 80), 3)
    return image


def test_order_points_returns_stable_corner_order() -> None:
    points = np.array([[10, 10], [100, 10], [100, 80], [10, 80]], dtype="float32")
    shuffled = points[[2, 0, 3, 1]]

    ordered = order_points(shuffled)

    np.testing.assert_array_equal(ordered, points)


def test_scanner_detects_and_warps_synthetic_document() -> None:
    scanner = DocumentScanner()

    result = scanner.scan(synthetic_document(), mode="bw")

    assert result.enhanced.ndim == 2
    assert result.enhanced.shape[0] > 300
    assert result.enhanced.shape[1] > 300
    assert result.corners.shape == (4, 2)


def test_scan_file_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    cv.imwrite(str(input_path), synthetic_document())

    DocumentScanner().scan_file(input_path, output_path, mode="gray")

    assert output_path.exists()
    written = cv.imread(str(output_path), cv.IMREAD_GRAYSCALE)
    assert written is not None
    assert written.size > 0


def test_color_enhancement_keeps_three_channels() -> None:
    enhanced = enhance_scan(synthetic_document(), mode="color")

    assert enhanced.ndim == 3
    assert enhanced.shape[2] == 3
