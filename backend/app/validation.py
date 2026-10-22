from pathlib import Path

import pandas as pd


class InvalidDataError(ValueError):
    pass


REQUIRED_COLUMNS = {
    "spies": {"Идентификатор пассажира", "ФИО", "Оценка LOF", "Количество вылетов"},
    "passenger_sources": {"source_passenger_id", "passenger_id", "source"},
    "profiles": {"profile_index"},
    "cards": {"profile_index", "card_index"},
    "flights": {"profile_index", "flight_index"},
    "xml_profiles": {"profile_index"},
    "xml_cards": {"profile_index", "card_index"},
    "xml_flights": {"profile_index", "card_index", "flight_index"},
    "yaml_flights": {"flight_index"},
    "yaml_passengers": {"flight_index", "passenger_index", "card_number"},
    "tab": {"TravelDoc"},
    "xlsx": {"passenger_name"},
    "csv": {"PassengerDocument"},
    "schedules": {"origin", "destination"},
    **{
        name: {"Идентификатор пассажира", "ФИО"}
        for name in (
            "flights_before",
            "flights_after",
            "passengers_before",
            "passengers_after",
        )
    },
}


def load_table(path: Path, max_bytes: int) -> pd.DataFrame:
    if path.stat().st_size > max_bytes:
        raise InvalidDataError("File exceeds size limit")
    table = pd.read_parquet(path, dtype_backend="numpy_nullable")
    if (
        len(table) > 5_000_000
        or len(table.columns) > 128
        or not table.columns.is_unique
    ):
        raise InvalidDataError("Parquet exceeds table limits")
    if table.memory_usage(index=True, deep=True).sum() > 2 * 1024**3:
        raise InvalidDataError("Table exceeds memory limit")
    return table
