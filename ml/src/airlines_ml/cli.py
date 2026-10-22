"""Один запуск: Parquet с пассажирами → spies.parquet."""

import argparse
import sys
import time
from pathlib import Path

import polars as pl

from airlines_ml.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    folder = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Поиск аномальных пассажирских историй"
    )
    parser.add_argument("input", nargs="?", type=Path, default=folder / "data")
    parser.add_argument(
        "output", nargs="?", type=Path, default=folder / "output" / "spies.parquet"
    )
    parser.add_argument("--airport-reference", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--n-neighbors", type=int, default=20)
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument(
        "--threshold-method", choices=("elbow", "percentile"), default=None
    )
    threshold.add_argument("--review-percentile", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    arguments = parser.parse_args(argv)
    started = time.monotonic()
    try:
        output = run_pipeline(
            arguments.input,
            arguments.output,
            device=arguments.device,
            n_neighbors=arguments.n_neighbors,
            review_percentile=arguments.review_percentile,
            threshold_method=arguments.threshold_method,
            batch_size=arguments.batch_size,
            airport_reference=arguments.airport_reference,
        )
    except (OSError, ValueError, RuntimeError, pl.exceptions.PolarsError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    print(f"Сохранено: {output} ({time.monotonic() - started:.1f} с)")
    return 0
