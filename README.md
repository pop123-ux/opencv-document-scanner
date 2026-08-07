# OpenCV Document Scanner

[![CI](https://github.com/pop123-ux/opencv-document-scanner/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/opencv-document-scanner/actions/workflows/ci.yml)

A clean, portfolio-ready document scanner built with Python and OpenCV. It detects a document in a photo, corrects the perspective, and exports a readable scan in black-and-white, grayscale, or enhanced color.

## Features

- Automatic document contour detection
- Four-point perspective correction
- Black-and-white adaptive thresholding for crisp scans
- Grayscale and color enhancement modes
- Single-image and batch folder processing
- Small reusable Python package plus CLI

## Demo

```bash
docscanner examples/sample-document.png --mode bw -o scans/sample_scan.png
```

Input photos can be taken from a phone at an angle. The scanner will find the strongest rectangular page-like contour and flatten it.

## Installation

```bash
git clone https://github.com/pop123-ux/opencv-document-scanner.git
cd opencv-document-scanner
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Usage

Scan one image:

```bash
docscanner path\to\photo.jpg -o scans\photo_scan.png
```

Scan every supported image in a folder:

```bash
docscanner path\to\photos -o scans --mode color
```

Available modes:

- `bw`: high-contrast black-and-white scan, best for receipts and printed pages
- `gray`: equalized grayscale scan
- `color`: perspective-corrected scan with contrast enhancement

You can also run the compatibility script directly:

```bash
python doc_scanner.py path\to\photo.jpg -o scans\photo_scan.png
```

## Project Structure

```text
.
|-- src/docscanner/
|   |-- cli.py
|   `-- scanner.py
|-- tests/
|-- examples/
|-- doc_scanner.py
`-- utils.py
```

## How It Works

1. Resize large images for faster contour detection.
2. Convert to grayscale, blur, run Canny edge detection, and close gaps.
3. Find the largest four-point contour.
4. Order corners consistently and apply a perspective transform.
5. Enhance the flattened scan for readability.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
