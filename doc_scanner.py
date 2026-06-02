"""Backward-compatible entry point for the document scanner."""

from docscanner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
