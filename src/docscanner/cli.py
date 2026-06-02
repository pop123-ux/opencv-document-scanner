from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from .scanner import DocumentScanner, EnhancementMode, image_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docscanner",
        description="Create perspective-corrected document scans from photos.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or folders to scan.")
    parser.add_argument("-o", "--output", type=Path, default=Path("scans"), help="Output file or directory.")
    parser.add_argument(
        "--mode",
        choices=["bw", "gray", "color"],
        default="bw",
        help="Enhancement mode for the final scan.",
    )
    parser.add_argument("--max-dimension", type=int, default=1600, help="Largest dimension used for contour detection.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = image_files(args.inputs)
    if not inputs:
        raise SystemExit("No supported input images found.")

    scanner = DocumentScanner(max_dimension=args.max_dimension)
    single_output_file = len(inputs) == 1 and args.output.suffix

    for input_path in inputs:
        output_path = args.output if single_output_file else args.output / f"{input_path.stem}_scan.png"
        scanner.scan_file(input_path, output_path, mode=cast(EnhancementMode, args.mode))
        print(f"{input_path} -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
