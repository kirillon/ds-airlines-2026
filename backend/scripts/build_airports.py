"""Build the local map reference from an unmodified OurAirports CSV snapshot."""

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = {
    "ident": "string",
    "name": "string",
    "iata_code": "string",
    "icao_code": "string",
    "iso_country": "string",
    "latitude_deg": "float64",
    "longitude_deg": "float64",
}


def build_reference(source: Path, output: Path) -> None:
    table = pd.read_csv(source, usecols=list(COLUMNS), dtype=COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    build_reference(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
