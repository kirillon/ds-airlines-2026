from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from defusedxml.ElementTree import iterparse

DEFAULT_SOURCE = Path(__file__).with_name("PointzAggregator-AirlinesData.xml")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "xml"

USERS_SCHEMA = pa.schema(
    [
        pa.field("profile_index", pa.int64(), nullable=False),
        pa.field("uid", pa.string(), nullable=False),
        pa.field("first_name", pa.string(), nullable=False),
        pa.field("last_name", pa.string(), nullable=False),
    ]
)

CARDS_SCHEMA = pa.schema(
    [
        pa.field("profile_index", pa.int64(), nullable=False),
        pa.field("card_index", pa.int64(), nullable=False),
        pa.field("card_number", pa.string(), nullable=False),
        pa.field("bonus_program", pa.string(), nullable=False),
    ]
)

FLIGHTS_SCHEMA = pa.schema(
    [
        pa.field("profile_index", pa.int64(), nullable=False),
        pa.field("card_index", pa.int64(), nullable=False),
        pa.field("flight_index", pa.int64(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("flight", pa.string(), nullable=False),
        pa.field("departure_airport", pa.string(), nullable=False),
        pa.field("arrival_airport", pa.string(), nullable=False),
        pa.field("fare", pa.string(), nullable=False),
        pa.field("travel_class", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ParseCounts:
    profiles: int
    cards: int
    flights: int


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


def require_attribute(element: Any, attribute: str, context: str) -> str:
    value = element.attrib.get(attribute)
    if value is None or not value.strip():
        raise ValueError(f"Missing attribute {attribute!r} in {context}")
    return value.strip()


def require_child(element: Any, tag: str, context: str) -> Any:
    child = element.find(tag)
    if child is None:
        raise ValueError(f"Missing element {tag!r} in {context}")
    return child


def require_text(element: Any, tag: str, context: str) -> str:
    child = require_child(element, tag, context)
    value = child.text
    if value is None or not value.strip():
        raise ValueError(f"Missing text in element {tag!r} in {context}")
    return value.strip()


def parse_iso_date(value: str, context: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid ISO date {value!r} in {context}") from error


def iter_users(source_path: Path) -> Iterator[Any]:
    for _, element in iterparse(source_path, events=("end",)):
        if element.tag == "user":
            yield element
            element.clear()


def parse_xml(source_path: Path, output_dir: Path, batch_size: int) -> ParseCounts:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not source_path.is_file():
        raise FileNotFoundError(f"XML source file not found: {source_path}")

    writers = {
        "profiles": BatchParquetWriter(
            output_dir / "xml_profiles.parquet", USERS_SCHEMA, batch_size
        ),
        "cards": BatchParquetWriter(
            output_dir / "xml_cards.parquet", CARDS_SCHEMA, batch_size
        ),
        "flights": BatchParquetWriter(
            output_dir / "xml_flights.parquet", FLIGHTS_SCHEMA, batch_size
        ),
    }
    for writer in writers.values():
        writer.open()

    profile_count = 0
    card_count = 0
    flight_count = 0

    try:
        for profile_index, user in enumerate(iter_users(source_path)):
            uid = require_attribute(user, "uid", f"profile {profile_index}")
            name = require_child(user, "name", f"profile {profile_index}")
            first_name = require_attribute(
                name, "first", f"profile {profile_index} name"
            )
            last_name = require_attribute(name, "last", f"profile {profile_index} name")
            writers["profiles"].append(
                {
                    "profile_index": profile_index,
                    "uid": uid,
                    "first_name": first_name,
                    "last_name": last_name,
                }
            )
            profile_count += 1

            cards = require_child(user, "cards", f"profile {profile_index}")
            for card_index, card_element in enumerate(cards.findall("card")):
                card_context = f"profile {profile_index}, card {card_index}"
                card_number = require_attribute(card_element, "number", card_context)
                bonus_program = require_text(
                    card_element, "bonusprogramm", card_context
                )
                writers["cards"].append(
                    {
                        "profile_index": profile_index,
                        "card_index": card_index,
                        "card_number": card_number,
                        "bonus_program": bonus_program,
                    }
                )
                card_count += 1

                activities = require_child(card_element, "activities", card_context)
                for flight_index, activity in enumerate(activities.findall("activity")):
                    flight_context = f"{card_context}, flight {flight_index}"
                    flight_date = parse_iso_date(
                        require_text(activity, "Date", flight_context),
                        flight_context,
                    )
                    fare = require_text(activity, "Fare", flight_context)
                    writers["flights"].append(
                        {
                            "profile_index": profile_index,
                            "card_index": card_index,
                            "flight_index": flight_index,
                            "date": flight_date,
                            "flight": require_text(activity, "Code", flight_context),
                            "departure_airport": require_text(
                                activity, "Departure", flight_context
                            ),
                            "arrival_airport": require_text(
                                activity, "Arrival", flight_context
                            ),
                            "fare": fare,
                            "travel_class": fare[0],
                        }
                    )
                    flight_count += 1

        for writer in writers.values():
            writer.close()
        for writer in writers.values():
            writer.commit()
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    return ParseCounts(profile_count, card_count, flight_count)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse PointzAggregator XML into normalized Parquet tables."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"XML source path (default: {DEFAULT_SOURCE})",
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
    counts = parse_xml(arguments.source, arguments.output_dir, arguments.batch_size)
    print(
        "XML parsing completed: "
        f"{counts.profiles} profiles, {counts.cards} cards, "
        f"{counts.flights} flights -> {arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
