"""Write validated schedule batches to typed Parquet."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable

    from models.contracts.schedules import ScheduleRecord

SCHEDULE_SCHEMA = pa.schema(
    [
        pa.field("origin", pa.string(), nullable=False),
        pa.field("destination", pa.string(), nullable=False),
        pa.field("valid_from", pa.date32(), nullable=False),
        pa.field("valid_to", pa.date32(), nullable=False),
        pa.field("weekdays", pa.uint8(), nullable=False),
        pa.field("departure_minute", pa.int16(), nullable=False),
        pa.field("arrival_minute", pa.int16(), nullable=False),
        pa.field("arrival_day_offset", pa.int8(), nullable=False),
        pa.field("flight_number", pa.string(), nullable=False),
        pa.field("aircraft", pa.string(), nullable=False),
        pa.field("duration_minutes", pa.int16(), nullable=False),
        pa.field("codeshare", pa.bool_(), nullable=False),
    ],
    metadata={
        b"schema_version": b"2",
        b"weekdays": b"bit0=Sunday;bit6=Saturday",
        b"time_basis": b"local_unspecified_timezone",
    },
)


@dataclass(frozen=True)
class OutputSummary:
    records: int
    elapsed_seconds: float


def write_parquet_dataset(
    batches: Iterable[tuple[ScheduleRecord, ...]], destination: Path
) -> OutputSummary:
    """Publish completed output atomically without replacing existing data."""
    started = time.monotonic()
    if destination.exists():
        raise FileExistsError("Output directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".airlines-", dir=destination.parent))
    completed = False
    records = 0
    try:
        with pq.ParquetWriter(
            staging / "schedules.parquet", SCHEDULE_SCHEMA, compression="zstd"
        ) as schedules:
            for batch in batches:
                if batch:
                    schedules.write_table(
                        pa.Table.from_pylist(
                            [record.model_dump() for record in batch],
                            schema=SCHEDULE_SCHEMA,
                        )
                    )
                    records += len(batch)
        summary = OutputSummary(records, time.monotonic() - started)
        (staging / "manifest.json").write_text(
            json.dumps({"schema_version": 2, **asdict(summary)}, indent=2) + "\n"
        )
        if destination.exists():
            raise FileExistsError("Output directory already exists")
        os.rename(staging, destination)
        completed = True
        return summary
    finally:
        if not completed:
            shutil.rmtree(staging)
