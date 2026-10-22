"""Recover missing endpoints only from unique observed flights."""

import polars as pl


def recover_event_fields(frame: pl.DataFrame) -> pl.DataFrame:
    """Use original endpoints as anchors to avoid cascading inferred evidence."""
    observed = frame.with_row_index("_event")
    result = observed
    valid = observed.filter(
        pl.col("FlightDate").is_not_null()
        & pl.col("FlightNumber").str.contains(r"^[A-Z0-9]{2}\d{1,4}[A-Z]?$")
    )
    for field, anchor in [("Origin", "Destination"), ("Destination", "Origin")]:
        keys = ["FlightDate", "FlightNumber", anchor]
        lookup = (
            valid.drop_nulls([*keys, field])
            .filter(
                pl.col(field).str.contains(r"^[A-Z]{3}$")
                & pl.col(anchor).str.contains(r"^[A-Z]{3}$")
            )
            .group_by(keys)
            .agg(
                pl.col(field).n_unique().alias("_versions"),
                pl.col(field).first().alias("_candidate"),
            )
            .filter(pl.col("_versions") == 1)
            .drop("_versions")
        )
        candidates = (
            observed.select("_event", *keys)
            .join(lookup, on=keys, how="left")
            .select("_event", "_candidate")
        )
        result = (
            result.join(candidates, on="_event", how="left")
            .with_columns(pl.coalesce(field, "_candidate").alias(field))
            .drop("_candidate")
        )
    return result.drop("_event")
