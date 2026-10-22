"""Read source-local passenger histories without changing source datasets."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from airlines_ml.cleaning import clean_text
from airlines_ml.geography import normalize_country


@dataclass
class LoadedSource:
    name: str
    frame: pl.DataFrame
    identities: pl.DataFrame
    input_files: list[Path]


def _read_table(path: Path, required: list[str]) -> pl.DataFrame:
    return pl.scan_parquet(path).select(required).collect()


def _validate_schema(frame: pl.DataFrame, expected: pl.Schema, table: str) -> None:
    for column, dtype in expected.items():
        if frame.schema[column] != dtype:
            raise ValueError(f"{table}: {column} must have dtype {dtype}")


def _validate_key(frame: pl.DataFrame, keys: list[str], table: str) -> None:
    indices = [key for key in keys if key.endswith("_index")]
    _validate_schema(frame, pl.Schema({key: pl.Int64 for key in indices}), table)
    if indices and frame.select(pl.any_horizontal(pl.col(indices) < 0).any()).item():
        raise ValueError(f"{table}: negative primary key index")
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError(f"{table}: null primary key")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{table}: duplicate primary key {keys}")


def _validate_foreign_key(
    child: pl.DataFrame, parent: pl.DataFrame, keys: list[str], table: str
) -> None:
    if child.join(parent.select(keys), on=keys, how="anti").height:
        raise ValueError(f"{table}: orphan foreign key {keys}")


def _build_identity(column: str, prefix: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String).str.strip_chars()
    valid = value.is_not_null() & ~value.str.to_lowercase().is_in(
        ["", "not presented", "null", "none", "unknown", "nan", "n/a", "na", "-"]
    )
    return (
        pl.when(valid)
        .then(pl.lit(prefix) + value)
        .otherwise(None)
        .alias("PassengerDocument")
    )


def build_tab_identity(frame: pl.DataFrame) -> pl.Expr:
    document = (
        pl.when(clean_text(pl.col("TravelDoc")).is_not_null())
        .then(pl.lit("tab:document:") + pl.col("TravelDoc").str.strip_chars())
        .alias("PassengerDocument")
    )
    countries = [
        pl.coalesce(normalize_country(pl.col(name)), clean_text(pl.col(name)))
        for name in ["issuing_country", "document_country"]
        if name in frame.columns
    ]
    if not countries:
        return document
    country = pl.coalesce(countries)
    return (
        pl.when(document.is_not_null() & country.is_not_null())
        .then(
            pl.lit("tab:document-country:")
            + country
            + pl.lit(":")
            + pl.col("TravelDoc").str.strip_chars()
        )
        .otherwise(document)
        .alias("PassengerDocument")
    )


def _load_tab(path: Path) -> LoadedSource:
    available = pl.scan_parquet(path).collect_schema().names()
    frame = _read_table(
        path,
        [
            "TravelDoc",
            "DepartDate",
            "Flight",
            "Dest",
            *[
                name
                for name in ["issuing_country", "document_country"]
                if name in available
            ],
        ],
    )
    _validate_schema(
        frame,
        pl.Schema(
            {
                "TravelDoc": pl.String,
                "DepartDate": pl.Date,
                "Flight": pl.String,
                "Dest": pl.String,
            }
        ),
        "tab",
    )
    expressions = [
        build_tab_identity(frame),
        pl.col("DepartDate").alias("FlightDate"),
        pl.col("Flight").cast(pl.String).alias("FlightNumber"),
        pl.col("Dest").cast(pl.String).alias("Destination"),
    ]
    canonical = frame.select(expressions)
    return LoadedSource(
        "tab",
        canonical,
        canonical.select("PassengerDocument").drop_nulls().unique(),
        [path],
    )


def _load_relational(folder: Path, name: str) -> LoadedSource:
    prefix = "xml_" if name == "xml" else ""
    paths = {
        part: folder / f"{prefix}{part}.parquet"
        for part in ["profiles", "cards", "flights"]
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            f"{name}: incomplete relational source; missing tables {missing}"
        )
    profiles = _read_table(paths["profiles"], ["profile_index"])
    cards = _read_table(paths["cards"], ["profile_index", "card_index"])
    keys = (
        ["profile_index", "card_index", "flight_index"]
        if name == "xml"
        else ["profile_index", "flight_index"]
    )
    flights = _read_table(
        paths["flights"], [*keys, "date", "flight", "arrival_airport"]
    )
    _validate_schema(
        flights,
        pl.Schema({"date": pl.Date, "flight": pl.String, "arrival_airport": pl.String}),
        name,
    )
    _validate_key(profiles, ["profile_index"], f"{name} profiles")
    _validate_key(cards, ["profile_index", "card_index"], f"{name} cards")
    _validate_key(flights, keys, f"{name} flights")
    _validate_foreign_key(cards, profiles, ["profile_index"], f"{name} cards")
    _validate_foreign_key(flights, profiles, ["profile_index"], f"{name} flights")
    if name == "xml":
        _validate_foreign_key(
            flights, cards, ["profile_index", "card_index"], "xml flights"
        )
    identity = _build_identity("profile_index", f"{name}:profile:")
    canonical = flights.select(
        identity,
        pl.col("date").alias("FlightDate"),
        pl.col("flight").cast(pl.String).alias("FlightNumber"),
        pl.col("arrival_airport").cast(pl.String).alias("Destination"),
    )
    identities = profiles.select(identity)
    return LoadedSource(name, canonical, identities, list(paths.values()))


def _build_proxy_identity(column: str, prefix: str, *, card: bool = False) -> pl.Expr:
    value = pl.col(column).cast(pl.String).str.strip_chars()
    valid = value.is_not_null() & ~value.str.to_lowercase().is_in(
        [
            "",
            "not presented",
            "null",
            "none",
            "unknown",
            "nan",
            "n/a",
            "na",
            "-",
            "неизвестно",
        ]
    )
    if card:
        value = value.str.to_uppercase().str.replace_all("\\s+", "")
    return (
        pl.when(valid)
        .then(pl.lit(prefix) + value)
        .otherwise(None)
        .alias("PassengerDocument")
    )


def _load_yaml(folder: Path) -> LoadedSource:
    paths = [folder / "yaml_flights.parquet", folder / "yaml_passengers.parquet"]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            f"yaml: incomplete relational source; missing tables {missing}"
        )
    flights = _read_table(
        paths[0], ["flight_index", "date", "flight", "arrival_airport"]
    )
    passengers = _read_table(
        paths[1], ["flight_index", "passenger_index", "card_number"]
    )
    _validate_schema(
        flights,
        pl.Schema({"date": pl.Date, "flight": pl.String, "arrival_airport": pl.String}),
        "yaml flights",
    )
    _validate_schema(
        passengers, pl.Schema({"card_number": pl.String}), "yaml passengers"
    )
    _validate_key(flights, ["flight_index"], "yaml flights")
    _validate_key(flights, ["date", "flight"], "yaml flights date/flight")
    _validate_key(passengers, ["flight_index", "passenger_index"], "yaml passengers")
    _validate_foreign_key(passengers, flights, ["flight_index"], "yaml passengers")
    identity = _build_proxy_identity("card_number", "yaml:card:", card=True)
    canonical = passengers.join(flights, on="flight_index", how="left").select(
        identity,
        pl.col("date").alias("FlightDate"),
        pl.col("flight").cast(pl.String).alias("FlightNumber"),
        pl.col("arrival_airport").cast(pl.String).alias("Destination"),
    )
    identities = passengers.select(identity).drop_nulls().unique()
    return LoadedSource("yaml", canonical, identities, paths)


def _load_xlsx(path: Path) -> LoadedSource:
    frame = _read_table(
        path, ["passenger_name", "departure_date", "flight", "destination"]
    )
    _validate_schema(
        frame,
        pl.Schema(
            {
                "passenger_name": pl.String,
                "departure_date": pl.Date,
                "flight": pl.String,
                "destination": pl.String,
            }
        ),
        "xlsx",
    )
    canonical = frame.select(
        _build_proxy_identity("passenger_name", "xlsx:name:"),
        pl.col("departure_date").alias("FlightDate"),
        pl.col("flight").cast(pl.String).alias("FlightNumber"),
        pl.col("destination").cast(pl.String).alias("Destination"),
    )
    return LoadedSource(
        "xlsx",
        canonical,
        canonical.select("PassengerDocument").drop_nulls().unique(),
        [path],
    )


def load_sources(input_path: str | Path) -> list[LoadedSource]:
    """Load independent cohorts; never resolve identities between source families."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    folder = path if path.is_dir() else path.parent
    if path.is_file():
        if path.name in {"yaml_flights.parquet", "yaml_passengers.parquet"}:
            return [_load_yaml(folder)]
        if path.name == "xlsx.parquet":
            return [_load_xlsx(path)]
        if path.name == "tab.parquet":
            return [_load_tab(path)]
        if path.name in {"profiles.parquet", "cards.parquet", "flights.parquet"}:
            return [_load_relational(folder, "json")]
        if path.name in {
            "xml_profiles.parquet",
            "xml_cards.parquet",
            "xml_flights.parquet",
        }:
            return [_load_relational(folder, "xml")]
        if path.name == "schedules.parquet":
            raise ValueError("Schedule-only input has no passenger histories")
        raise ValueError(f"Unsupported passenger filename: {path.name}")
    result = []
    if (folder / "tab.parquet").exists():
        result.append(_load_tab(folder / "tab.parquet"))
    for name, prefix in [("json", ""), ("xml", "xml_")]:
        if any(
            (folder / f"{prefix}{part}.parquet").exists()
            for part in ["profiles", "cards", "flights"]
        ):
            result.append(_load_relational(folder, name))
    if any(
        (folder / name).exists()
        for name in ["yaml_flights.parquet", "yaml_passengers.parquet"]
    ):
        result.append(_load_yaml(folder))
    if (folder / "xlsx.parquet").exists():
        result.append(_load_xlsx(folder / "xlsx.parquet"))
    if not result:
        raise ValueError(
            "No supported passenger source found; schedules do not contain passengers"
        )
    return result
