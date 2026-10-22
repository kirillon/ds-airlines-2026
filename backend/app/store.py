from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.airports import load_airports
from backend.app.validation import REQUIRED_COLUMNS, InvalidDataError, load_table

ID_COLUMNS = ("Идентификатор пассажира", "passenger_id", "PassengerDocument")
AIRPORT_FIELDS = (
    "departure_airport",
    "arrival_airport",
    "origin",
    "destination",
    "Origin",
    "Destination",
    "Откуда",
    "Куда",
    "DepartureAirport",
    "ArrivalAirport",
    "From",
    "Dest",
)
TABLE_NAMES = frozenset(REQUIRED_COLUMNS)
RELATED_TABLE_NAMES = TABLE_NAMES - {"csv", "schedules"}


def _validate_source_pointer(pointer: Any) -> str:
    if not isinstance(pointer, str) or not 1 <= len(pointer) <= 512:
        raise InvalidDataError("Invalid source pointer")
    parts = pointer.split(":", 2)
    if len(parts) != 3:
        raise InvalidDataError("Invalid source pointer")
    source, kind, value = parts
    prefix = f"{source}:{kind}"
    if prefix in {"json:profile", "xml:profile", "xlsx:record"}:
        valid = value.isascii() and value.isdigit() and int(value) < 2**63
    elif prefix == "tab:document-country":
        country, separator, document = value.partition(":")
        valid = bool(country.strip() and separator and document.strip())
    else:
        valid = prefix in {"yaml:card", "tab:document"} and bool(value.strip())
    if not valid:
        raise InvalidDataError("Invalid source pointer")
    return pointer


def _sanitize_value(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _sanitize_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_sanitize_value(item) for item in value]
    return value


class ParquetStore:
    def __init__(
        self,
        folder: Path,
        airport_reference: Path,
        max_file_bytes: int = 128 * 1024 * 1024,
    ):
        self._tables: dict[str, pd.DataFrame] = {}
        if folder.is_symlink():
            raise InvalidDataError("Data paths must not be symbolic links")
        for name in sorted(TABLE_NAMES):
            path = folder / f"{name}.parquet"
            if path.is_symlink():
                raise InvalidDataError("Data paths must not be symbolic links")
            if path.is_file():
                self._tables[name] = load_table(path, max_file_bytes)
        self._airports = load_airports(airport_reference)

    def list_tables(self) -> list[str]:
        return sorted(self._tables)

    def _require_columns(self, table: str, *columns: str) -> pd.DataFrame:
        frame = self._tables[table]
        if not set(columns).issubset(frame.columns):
            raise InvalidDataError("Missing required columns")
        return frame

    def _find_id_column(self, table: str) -> str:
        for name in ID_COLUMNS:
            if name in self._tables[table].columns:
                return name
        raise InvalidDataError("Missing passenger identifier")

    def _build_rows(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [self._enrich_row(row) for row in frame.to_dict(orient="records")]

    def list_spies(self, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 1000:
            raise ValueError("Invalid pagination")
        if "spies" not in self._tables:
            raise InvalidDataError("Missing analytical results")
        self._find_id_column("spies")
        frame = self._tables["spies"]
        return {
            "total": len(frame),
            "offset": offset,
            "limit": limit,
            "items": self._build_rows(frame.iloc[offset : offset + limit]),
        }

    def get_spy(self, passenger_id: str) -> dict[str, Any]:
        if "spies" not in self._tables:
            raise KeyError(passenger_id)
        frame = self._tables["spies"]
        matches = frame.loc[frame[self._find_id_column("spies")].eq(passenger_id)]
        if matches.empty:
            raise KeyError(passenger_id)
        if len(matches) != 1:
            raise InvalidDataError("Duplicate analytical identifier")
        return self._build_rows(matches)[0]

    def _find_sources(self, passenger_id: str) -> tuple[list[str], str]:
        representative = _validate_source_pointer(passenger_id.removeprefix("person:"))
        if "passenger_sources" not in self._tables:
            return [representative], "representative_only"
        frame = self._require_columns(
            "passenger_sources", "passenger_id", "source_passenger_id", "source"
        )
        rows = frame.loc[frame["passenger_id"].eq(passenger_id)]
        if len(rows) > 100_000:
            raise InvalidDataError("Too many source pointers")
        pointers = {
            _validate_source_pointer(pointer) for pointer in rows["source_passenger_id"]
        }
        if (
            not rows["source_passenger_id"]
            .str.split(":")
            .str[0]
            .eq(rows["source"])
            .fillna(False)
            .all()
        ):
            raise InvalidDataError("Invalid source lineage")
        if representative not in pointers:
            raise InvalidDataError("Incomplete source lineage")
        conflicts = frame["source_passenger_id"].isin(pointers) & (
            frame["passenger_id"].isna() | frame["passenger_id"].ne(passenger_id)
        )
        if conflicts.any():
            raise InvalidDataError("Ambiguous source lineage")
        return sorted(pointers), "lineage"

    def list_related(self, passenger_id: str, table: str) -> dict[str, Any]:
        if table not in RELATED_TABLE_NAMES:
            raise KeyError(table)
        self.get_spy(passenger_id)
        pointers, linkage = self._find_sources(passenger_id)
        available = table in self._tables
        items = (
            self._build_rows(self._find_related(passenger_id, pointers, table))
            if available
            else []
        )
        return {
            "total": len(items),
            "items": items,
            "available": available,
            "table": table,
            "linkage": linkage,
        }

    def _find_related(
        self,
        passenger_id: str,
        pointers: list[str],
        table: str,
    ) -> pd.DataFrame:
        frame = self._tables[table]
        if table in {"spies", "passenger_sources"}:
            column = self._find_id_column(table)
            return frame.loc[frame[column].eq(passenger_id)]
        if table.endswith("_before") or table.endswith("_after"):
            ids = pointers if table.endswith("_before") else [passenger_id]
            return frame.loc[frame[self._find_id_column(table)].isin(ids)]
        if table in {
            "profiles",
            "cards",
            "flights",
            "xml_profiles",
            "xml_cards",
            "xml_flights",
        }:
            frame = self._require_columns(table, "profile_index")
            prefix = "xml:profile:" if table.startswith("xml_") else "json:profile:"
            indices = [int(p[len(prefix) :]) for p in pointers if p.startswith(prefix)]
            return frame.loc[frame["profile_index"].isin(indices)]
        if table == "xlsx":
            positions = {
                int(p[len("xlsx:record:") :])
                for p in pointers
                if p.startswith("xlsx:record:")
            }
            return frame.iloc[
                sorted(index for index in positions if index < len(frame))
            ]
        if table == "tab":
            return self._find_tab_rows(pointers)
        if table in {"yaml_passengers", "yaml_flights"}:
            cards = [
                p[len("yaml:card:") :] for p in pointers if p.startswith("yaml:card:")
            ]
            if not cards or "yaml_passengers" not in self._tables:
                return frame.iloc[:0]
            passengers = self._require_columns("yaml_passengers", "card_number")
            normalized = (
                passengers["card_number"]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", "", regex=True)
                .str.upper()
            )
            matches = passengers.loc[normalized.isin(cards)]
            if table == "yaml_passengers":
                return matches
            self._require_columns("yaml_passengers", "flight_index")
            frame = self._require_columns(table, "flight_index")
            return frame.loc[frame["flight_index"].isin(matches["flight_index"])]
        return frame.iloc[:0]

    def _find_tab_rows(self, pointers: list[str]) -> pd.DataFrame:
        frame = self._require_columns("tab", "TravelDoc")
        documents = frame["TravelDoc"].astype("string").str.strip()
        countries: pd.Series | None = None
        for name in ("issuing_country", "document_country"):
            if name in frame.columns:
                values = frame[name].astype("string").str.strip().str.upper()
                countries = values if countries is None else countries.fillna(values)
        matches = pd.Series(False, index=frame.index)
        for pointer in pointers:
            if pointer.startswith("tab:document:"):
                matches |= documents.eq(pointer.removeprefix("tab:document:"))
            elif countries is not None and pointer.startswith("tab:document-country:"):
                country, number = pointer.removeprefix("tab:document-country:").split(
                    ":", 1
                )
                matches |= countries.eq(country) & documents.eq(number)
        return frame.loc[matches]

    def _enrich_row(self, row: dict[str, Any]) -> dict[str, Any]:
        for field in AIRPORT_FIELDS:
            if field in row:
                value = row[field]
                code = value.strip().upper() if isinstance(value, str) else ""
                row[f"{field}_info"] = self._airports.get(code)
        return _sanitize_value(row)
