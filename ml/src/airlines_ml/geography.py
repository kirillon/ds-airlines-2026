"""Resolve countries from an offline airport reference without guessing cities."""

import gettext
from functools import lru_cache
from pathlib import Path

import polars as pl
import pycountry
from babel import Locale
from babel.core import get_global

DEFAULT_AIRPORT_REFERENCE = (
    Path(__file__).resolve().parents[2] / "resources/airports.parquet"
)


def _normalize_country_key(value: pl.Expr) -> pl.Expr:
    return (
        value.cast(pl.String)
        .str.normalize("NFKD")
        .str.replace_all(r"\p{M}", "")
        .str.replace_all("['’]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.to_uppercase()
    )


@lru_cache(maxsize=1)
def _load_country_codes() -> dict[str, str]:
    reference = pl.read_parquet(
        DEFAULT_AIRPORT_REFERENCE.with_name("countries.parquet")
    )
    aliases = [(name, code) for code, name in reference.iter_rows()]
    translate = gettext.translation(
        "iso3166-1", pycountry.LOCALES_DIR, languages=["ru"]
    ).gettext
    for country in pycountry.countries:
        aliases.extend(
            [(country.alpha_2, country.alpha_2), (country.alpha_3, country.alpha_2)]
        )
        for field in ("name", "official_name", "common_name"):
            if name := getattr(country, field, None):
                aliases.extend(
                    [(name, country.alpha_2), (translate(name), country.alpha_2)]
                )
    codes = set(reference["code"])
    for language in ("en", "ru"):
        aliases.extend(
            (name, code)
            for code, name in Locale(language).territories.items()
            if code in codes
        )
    aliases.extend(
        (alias, targets[0])
        for alias, targets in get_global("territory_aliases").items()
        if len(targets) == 1 and targets[0] in codes
    )
    return dict(
        pl.DataFrame(aliases, schema=["name", "code"], orient="row")
        .with_columns(_normalize_country_key(pl.col("name")))
        .group_by("name")
        .agg(pl.col("code").unique())
        .filter(pl.col("code").list.len() == 1)
        .select("name", pl.col("code").list.first())
        .iter_rows()
    )


def normalize_country(value: pl.Expr) -> pl.Expr:
    """Match exact ISO/CLDR English and Russian names; unknown labels stay null."""
    return _normalize_country_key(value).replace_strict(
        _load_country_codes(), default=None, return_dtype=pl.String
    )


def _normalize_location(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.to_uppercase()
        .replace("", None)
    )


def enrich_countries(
    events: pl.DataFrame,
    airport_reference: str | Path | pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Match exact IATA, ICAO, ident or airport name to ISO country.

    Reference columns are ``iso_country`` and any of ``name``, ``iata_code``,
    ``icao_code``, ``ident``. Keys resolving to different countries are rejected.
    Existing recognized countries take precedence over airport-derived values.
    Country names and codes share one ISO vocabulary; truncated names stay unknown.
    """
    reference_input = (
        DEFAULT_AIRPORT_REFERENCE if airport_reference is None else airport_reference
    )
    reference: pl.DataFrame
    if isinstance(reference_input, str | Path):
        reference_path = Path(reference_input)
        reference = pl.read_parquet(reference_path)
    else:
        reference = reference_input
    result = events
    for field in ("OriginCountry", "DestinationCountry"):
        result = result.with_columns(
            normalize_country(pl.col(field)).alias(field)
            if field in result.columns
            else pl.lit(None, dtype=pl.String).alias(field)
        )
    aliases = [
        column
        for column in ("name", "iata_code", "icao_code", "ident")
        if column in reference.columns
    ]
    if "iso_country" not in reference.columns or not aliases:
        raise ValueError("Airport reference requires iso_country and an airport key")
    lookup = (
        reference.select(
            normalize_country(pl.col("iso_country")).alias("_country"),
            *[_normalize_location(column).alias(column) for column in aliases],
        )
        .unpivot(index="_country", on=aliases, value_name="_airport")
        .drop_nulls(["_airport", "_country"])
        .filter(pl.col("_country").str.contains(r"^[A-Z]{2}$"))
        .group_by("_airport")
        .agg(
            pl.col("_country").n_unique().alias("_countries"),
            pl.col("_country").first(),
        )
        .filter(pl.col("_countries") == 1)
        .select("_airport", "_country")
    )
    for endpoint in ("Origin", "Destination"):
        if endpoint not in result.columns:
            continue
        result = (
            result.with_columns(_normalize_location(endpoint).alias("_airport"))
            .join(lookup, on="_airport", how="left", maintain_order="left")
            .with_columns(
                pl.coalesce(f"{endpoint}Country", "_country").alias(
                    f"{endpoint}Country"
                )
            )
            .drop("_airport", "_country")
        )
    return result


def compute_country_features(events: pl.DataFrame) -> pl.DataFrame:
    """Log1p distinct known countries across both endpoints; unknown stays null."""
    if "passenger_id" not in events.columns:
        events = events.rename({"PassengerDocument": "passenger_id"})
    countries = (
        events.select(
            "passenger_id",
            normalize_country(pl.col("OriginCountry")).alias("OriginCountry"),
            normalize_country(pl.col("DestinationCountry")).alias("DestinationCountry"),
        )
        .unpivot(
            index="passenger_id",
            on=["OriginCountry", "DestinationCountry"],
            value_name="country",
        )
        .drop_nulls("country")
        .group_by("passenger_id")
        .agg(pl.col("country").n_unique().log1p().alias("visited_country_diversity"))
    )
    return (
        events.select("passenger_id")
        .unique()
        .join(countries, on="passenger_id", how="left")
    )
