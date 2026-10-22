"""Project raw sources to a common schema before ML cleaning or identity merging."""

from pathlib import Path

import polars as pl

from airlines_ml.inputs import LoadedSource


def _read_columns(path: Path, columns: list[str]) -> pl.DataFrame:
    scan = pl.scan_parquet(path)
    available = scan.collect_schema().names()
    return scan.select([column for column in columns if column in available]).collect()


def _select_optional(frame: pl.DataFrame, name: str) -> pl.Expr:
    return pl.col(name) if name in frame.columns else pl.lit(None, dtype=pl.String)


def _build_source_exports(loaded: LoadedSource) -> tuple[pl.DataFrame, pl.DataFrame]:
    folder = loaded.input_files[0].parent
    identities = loaded.identities
    if loaded.name in {"json", "xml"}:
        prefix = "xml_" if loaded.name == "xml" else ""
        profiles = _read_columns(
            folder / f"{prefix}profiles.parquet",
            ["profile_index", "last_name", "first_name", "middle_name"],
        )
        names = profiles.select(
            (
                pl.lit(f"{loaded.name}:profile:")
                + pl.col("profile_index").cast(pl.String)
            ).alias("PassengerDocument"),
            pl.concat_str(
                [
                    _select_optional(profiles, column)
                    for column in ["last_name", "first_name", "middle_name"]
                ],
                separator=" ",
                ignore_nulls=True,
            ).alias("ФИО"),
        )
        events = _read_columns(
            folder / f"{prefix}flights.parquet", ["departure_airport"]
        )
        flights = loaded.frame.with_columns(
            _select_optional(events, "departure_airport").alias("Origin")
            if "departure_airport" not in events.columns
            else events.get_column("departure_airport").alias("Origin")
        ).join(names, on="PassengerDocument", how="left", maintain_order="left")
    elif loaded.name in {"tab", "xlsx"}:
        name_column, origin_column = (
            ("PaxName", "From")
            if loaded.name == "tab"
            else ("passenger_name", "origin")
        )
        raw = _read_columns(loaded.input_files[0], [name_column, origin_column])
        flights = loaded.frame.with_columns(
            raw.get_column(name_column).alias("ФИО")
            if name_column in raw.columns
            else pl.lit(None, dtype=pl.String).alias("ФИО"),
            raw.get_column(origin_column).alias("Origin")
            if origin_column in raw.columns
            else pl.lit(None, dtype=pl.String).alias("Origin"),
        )
        if loaded.name == "xlsx":
            flights = flights.with_columns(
                (pl.lit("xlsx:record:") + pl.int_range(pl.len()).cast(pl.String)).alias(
                    "PassengerDocument"
                )
            )
            identities = flights.select("PassengerDocument")
        names = flights.select("PassengerDocument", "ФИО")
    else:
        raw_flights = _read_columns(
            folder / "yaml_flights.parquet",
            ["date", "flight", "arrival_airport", "departure_airport"],
        )
        origins = raw_flights.select(
            pl.col("date").alias("FlightDate"),
            pl.col("flight").alias("FlightNumber"),
            pl.col("arrival_airport").alias("Destination"),
            _select_optional(raw_flights, "departure_airport").alias("Origin"),
        )
        flights = loaded.frame.join(
            origins,
            on=["FlightDate", "FlightNumber", "Destination"],
            how="left",
            validate="m:1",
            nulls_equal=True,
            maintain_order="left",
        ).with_columns(pl.lit(None, dtype=pl.String).alias("ФИО"))
        names = loaded.identities.with_columns(
            pl.lit(None, dtype=pl.String).alias("ФИО")
        )
    names = names.group_by("PassengerDocument", maintain_order=True).agg(
        pl.col("ФИО")
        .filter(pl.col("ФИО").is_not_null() & (pl.col("ФИО").str.strip_chars() != ""))
        .first()
    )
    passengers = (
        identities.join(names, on="PassengerDocument", how="left")
        .rename({"PassengerDocument": "passenger_id"})
        .select("passenger_id", "ФИО")
    )
    return flights.select(
        "PassengerDocument",
        "ФИО",
        "FlightDate",
        "FlightNumber",
        "Origin",
        "Destination",
    ), passengers


def build_before_exports(
    sources: list[LoadedSource],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep all source flight rows and source identities, including unnamed profiles.

    This is a unified projection of source tables, not a byte-for-byte raw backup.
    Original field values are preserved; relational names are assembled in FIO order.
    """
    pairs = [_build_source_exports(source) for source in sources]
    return tuple(pl.concat(frames, how="vertical_relaxed") for frames in zip(*pairs))
