"""Compatibility helpers for simple OpenCV document-scanner scripts."""

from docscanner.scanner import enhance_scan, four_point_transform, order_points, preprocess_for_edges

__all__ = ["enhance_scan", "four_point_transform", "order_points", "preprocess_for_edges"]
