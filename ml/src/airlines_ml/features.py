"""Eight behavioral passenger aggregates from normalized travel records."""

import polars as pl

from airlines_ml.cleaning import clean_text
from airlines_ml.inputs import LoadedSource

FEATURE_NAMES = [
    "flight_count",
    "destination_count",
    "destination_rarity",
    "median_departure_gap",
    "gap_burstiness",
    "same_day_departure_share",
    "weekend_share",
    "marketed_carrier_entropy",
]


def clean_departures(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.select(
        pl.col("PassengerDocument")
        .str.strip_chars()
        .replace("", None)
        .alias("passenger_id"),
        pl.col("FlightDate").alias("date"),
        clean_text(pl.col("FlightNumber")).alias("flight"),
        clean_text(pl.col("Destination")).alias("destination"),
        clean_text(pl.col("Origin")).alias("origin")
        if "Origin" in frame.columns
        else pl.lit(None, dtype=pl.String).alias("origin"),
    )
    return frame.filter(
        pl.col("passenger_id").is_not_null()
        & pl.col("date").is_not_null()
        & pl.col("flight").str.contains(r"^[A-Z0-9]{2}\d{1,4}[A-Z]?$")
    ).unique()


def prepare_source(loaded: LoadedSource) -> pl.DataFrame:
    frame = loaded.frame.select(
        pl.col("PassengerDocument")
        .str.strip_chars()
        .replace("", None)
        .alias("passenger_id")
    )
    passengers = (
        pl.concat(
            [
                frame.select("passenger_id"),
                loaded.identities.select(
                    pl.col("PassengerDocument")
                    .str.strip_chars()
                    .replace("", None)
                    .alias("passenger_id")
                ),
            ]
        )
        .drop_nulls()
        .unique()
    )
    clean = clean_departures(loaded.frame)
    population = clean.group_by("passenger_id").agg(
        pl.len().cast(pl.Int64).alias("departure_count"),
        pl.col("date").n_unique().alias("active_dates"),
        pl.col("destination").drop_nulls().n_unique().alias("destinations"),
        (pl.col("date").dt.weekday() >= 6).mean().alias("weekend_share"),
    )
    eligible = population.filter(pl.col("departure_count") >= 5).select("passenger_id")
    known = clean.filter(pl.col("destination").is_not_null())
    prevalence = (
        known.join(eligible, on="passenger_id", how="semi")
        .select("passenger_id", "destination")
        .unique()
        .group_by("destination")
        .len(name="passengers")
        .with_columns(
            ((pl.col("passengers") + 1) / (eligible.height + 2)).alias("probability")
        )
    )
    rarity = (
        known.join(prevalence, on="destination", how="left")
        .group_by("passenger_id")
        .agg(
            (-pl.col("probability").fill_null(1 / (eligible.height + 2)).log())
            .mean()
            .alias("destination_rarity")
        )
    )
    gaps = (
        clean.select("passenger_id", "date")
        .unique()
        .sort(["passenger_id", "date"])
        .with_columns(
            pl.col("date")
            .diff()
            .over("passenger_id")
            .dt.total_days()
            .cast(pl.Float64)
            .alias("gap")
        )
        .group_by("passenger_id")
        .agg(
            pl.col("gap").median().log1p().alias("median_departure_gap"),
            pl.col("gap").mean().alias("mean"),
            pl.col("gap").std(ddof=0).alias("std"),
        )
        .with_columns(
            ((pl.col("std") - pl.col("mean")) / (pl.col("std") + pl.col("mean"))).alias(
                "gap_burstiness"
            )
        )
        .select("passenger_id", "median_departure_gap", "gap_burstiness")
    )
    carriers = (
        clean.with_columns(pl.col("flight").str.slice(0, 2).alias("carrier"))
        .group_by("passenger_id", "carrier")
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("passenger_id")).alias(
                "probability"
            )
        )
        .group_by("passenger_id")
        .agg(
            (-pl.col("probability") * pl.col("probability").log())
            .sum()
            .alias("marketed_carrier_entropy")
        )
    )
    return (
        passengers.join(population, on="passenger_id", how="left")
        .join(rarity, on="passenger_id", how="left")
        .join(gaps, on="passenger_id", how="left")
        .join(carriers, on="passenger_id", how="left")
        .with_columns(
            pl.col("departure_count").fill_null(0),
            pl.col("departure_count").log1p().alias("flight_count"),
            pl.when(pl.col("destinations") > 0)
            .then(pl.col("destinations").log1p())
            .otherwise(None)
            .alias("destination_count"),
            (
                (pl.col("departure_count") - pl.col("active_dates"))
                / pl.col("departure_count")
            ).alias("same_day_departure_share"),
        )
        .select("passenger_id", "departure_count", *FEATURE_NAMES)
        .sort("passenger_id")
    )
