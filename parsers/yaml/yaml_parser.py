from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE = Path(__file__).with_name("SkyTeam-Exchange.yaml")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "yaml"

FLIGHTS_SCHEMA = pa.schema(
    [
        pa.field("flight_index", pa.int64(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("flight", pa.string(), nullable=False),
        pa.field("departure_airport", pa.string(), nullable=False),
        pa.field("arrival_airport", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
    ]
)

PASSENGERS_SCHEMA = pa.schema(
    [
        pa.field("flight_index", pa.int64(), nullable=False),
        pa.field("passenger_index", pa.int64(), nullable=False),
        pa.field("card_number", pa.string(), nullable=False),
        pa.field("travel_class", pa.string(), nullable=False),
        pa.field("fare", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ParseCounts:
    dates: int
    flights: int
    passengers: int


class BatchParquetWriter:
    def __init__(self, output_path: Path, schema: pa.Schema, batch_size: int) -> None:
        self.output_path = output_path
        self.temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        self.schema = schema
        self.batch_size = batch_size
        self.rows: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path.unlink(missing_ok=True)
        self.writer = pq.ParquetWriter(
            self.temporary_path,
            self.schema,
            compression="zstd",
            use_dictionary=True,
        )

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        if self.writer is None:
            raise RuntimeError("Parquet writer is not open")
        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            raise RuntimeError("Parquet writer is not open")
        self.writer.close()
        self.writer = None

    def commit(self) -> None:
        self.temporary_path.replace(self.output_path)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.temporary_path.unlink(missing_ok=True)


def parse_mapping_key(line: str, line_number: int) -> str:
    if not line.endswith(":"):
        raise ValueError(f"Expected mapping key at YAML line {line_number}")
    key = line[:-1].strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1]
    if not key:
        raise ValueError(f"Empty mapping key at YAML line {line_number}")
    return key


def parse_iso_date(value: str, line_number: int) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid ISO date {value!r} at YAML line {line_number}"
        ) from error


def parse_scalar_field(line: str, line_number: int) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Expected key/value field at YAML line {line_number}")
    key, value = line.split(":", 1)
    value = value.strip()
    if not value:
        raise ValueError(f"Empty value for {key!r} at YAML line {line_number}")
    return key.strip(), value


def parse_passenger(line: str, line_number: int) -> tuple[str, str, str]:
    separator = ": {"
    if separator not in line or not line.endswith("}"):
        raise ValueError(f"Invalid FF entry at YAML line {line_number}")
    card_number, payload = line.split(separator, 1)
    fields: dict[str, str] = {}
    for item in payload[:-1].split(", "):
        key, value = parse_scalar_field(item, line_number)
        fields[key] = value
    if set(fields) != {"CLASS", "FARE"}:
        raise ValueError(
            f"Expected CLASS and FARE in FF entry at YAML line {line_number}"
        )
    card_number = card_number.strip()
    if not card_number:
        raise ValueError(f"Empty card number at YAML line {line_number}")
    if not fields["FARE"].startswith(fields["CLASS"]):
        raise ValueError(
            f"FARE does not match CLASS in FF entry at YAML line {line_number}"
        )
    return card_number, fields["CLASS"], fields["FARE"]


def parse_yaml(source_path: Path, output_dir: Path, batch_size: int) -> ParseCounts:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not source_path.is_file():
        raise FileNotFoundError(f"YAML source file not found: {source_path}")

    writers = {
        "flights": BatchParquetWriter(
            output_dir / "yaml_flights.parquet", FLIGHTS_SCHEMA, batch_size
        ),
        "passengers": BatchParquetWriter(
            output_dir / "yaml_passengers.parquet", PASSENGERS_SCHEMA, batch_size
        ),
    }
    for writer in writers.values():
        writer.open()

    current_date: date | None = None
    current_flight: str | None = None
    current_flight_index: int | None = None
    current_fields: dict[str, str] = {}
    current_passengers: list[tuple[str, str, str]] = []
    date_count = 0
    flight_count = 0
    passenger_count = 0

    def flush_flight(line_number: int) -> None:
        nonlocal flight_count, passenger_count
        if current_flight is None:
            return
        if current_date is None or current_flight_index is None:
            raise ValueError(f"Flight without date before YAML line {line_number}")
        required_fields = {"FROM", "STATUS", "TO"}
        if set(current_fields) != required_fields:
            missing = sorted(required_fields - set(current_fields))
            extra = sorted(set(current_fields) - required_fields)
            raise ValueError(
                f"Invalid flight fields before YAML line {line_number}: "
                f"missing={missing}, extra={extra}"
            )
        if not current_passengers:
            raise ValueError(
                f"Flight {current_flight!r} has no FF entries before "
                f"YAML line {line_number}"
            )

        writers["flights"].append(
            {
                "flight_index": current_flight_index,
                "date": current_date,
                "flight": current_flight,
                "departure_airport": current_fields["FROM"],
                "arrival_airport": current_fields["TO"],
                "status": current_fields["STATUS"],
            }
        )
        for passenger_index, (card_number, travel_class, fare) in enumerate(
            current_passengers
        ):
            writers["passengers"].append(
                {
                    "flight_index": current_flight_index,
                    "passenger_index": passenger_index,
                    "card_number": card_number,
                    "travel_class": travel_class,
                    "fare": fare,
                }
            )
        flight_count += 1
        passenger_count += len(current_passengers)

    last_line_number = 0
    try:
        with source_path.open("r", encoding="utf-8", newline="") as source:
            for line_number, raw_line in enumerate(source, start=1):
                last_line_number = line_number
                line = raw_line.rstrip("\r\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if "\t" in line[: len(line) - len(line.lstrip())]:
                    raise ValueError(
                        f"Tab indentation is not supported at YAML line {line_number}"
                    )

                indent = len(line) - len(line.lstrip(" "))
                stripped = line.strip()

                if indent == 0:
                    flush_flight(line_number)
                    current_date = parse_iso_date(
                        parse_mapping_key(stripped, line_number), line_number
                    )
                    current_flight = None
                    current_flight_index = None
                    current_fields.clear()
                    current_passengers.clear()
                    date_count += 1
                elif indent == 2:
                    flush_flight(line_number)
                    if current_date is None:
                        raise ValueError(
                            f"Flight before date at YAML line {line_number}"
                        )
                    current_flight = parse_mapping_key(stripped, line_number)
                    current_flight_index = flight_count
                    current_fields.clear()
                    current_passengers.clear()
                elif indent == 4:
                    if current_flight is None:
                        raise ValueError(
                            f"Flight field before flight at YAML line {line_number}"
                        )
                    if stripped == "FF:":
                        continue
                    key, value = parse_scalar_field(stripped, line_number)
                    if key in current_fields:
                        raise ValueError(
                            f"Duplicate field {key!r} at YAML line {line_number}"
                        )
                    current_fields[key] = value
                elif indent == 6:
                    if current_flight is None:
                        raise ValueError(
                            f"FF entry before flight at YAML line {line_number}"
                        )
                    current_passengers.append(parse_passenger(stripped, line_number))
                else:
                    raise ValueError(
                        f"Unsupported indentation {indent} at YAML line {line_number}"
                    )

        flush_flight(last_line_number + 1)
        for writer in writers.values():
            writer.close()
        for writer in writers.values():
            writer.commit()
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    return ParseCounts(date_count, flight_count, passenger_count)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse SkyTeam Exchange YAML into normalized Parquet tables."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"YAML source path (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows buffered before each Parquet write (default: 50000)",
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    counts = parse_yaml(arguments.source, arguments.output_dir, arguments.batch_size)
    print(
        "YAML parsing completed: "
        f"{counts.dates} dates, {counts.flights} flights, "
        f"{counts.passengers} passenger records -> {arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
