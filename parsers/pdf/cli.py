"""Command line ingestion of local SkyTeam timetable PDFs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from itertools import chain
from pathlib import Path

from parsers.pdf.output import write_parquet_dataset
from parsers.pdf.reader import ParseOptions, PdfParseError, iter_pdf_batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDF file or directory of PDF files")
    parser.add_argument(
        "output", type=Path, help="New output directory (must not exist)"
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-pages", type=int, default=100000)
    parser.add_argument("--max-file-bytes", type=int, default=1024**3)
    parser.add_argument("--max-page-chars", type=int, default=200000)
    parser.add_argument("--valid-from", type=date.fromisoformat)
    parser.add_argument("--valid-to", type=date.fromisoformat)
    args = parser.parse_args()
    if args.input.is_dir():
        paths = sorted(
            path
            for path in args.input.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
    else:
        paths = [args.input] if args.input.is_file() else []
    if not paths:
        parser.error("No local PDF files found")
    try:
        options = ParseOptions(
            batch_size=args.batch_size,
            max_pages=args.max_pages,
            max_file_bytes=args.max_file_bytes,
            max_page_chars=args.max_page_chars,
            valid_from=args.valid_from,
            valid_to=args.valid_to,
        )
        batches = chain.from_iterable(iter_pdf_batches(path, options) for path in paths)
        summary = write_parquet_dataset(batches, args.output)
    except PdfParseError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            f"Ingestion failed ({type(exc).__name__}); check input and options.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(asdict(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
