"""Пассажирские признаки, LOF и выгрузки до/после очистки."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from unidecode import unidecode

from airlines_ml.entities import resolve_passengers
from airlines_ml.exports import build_before_exports
from airlines_ml.features import clean_departures, prepare_source
from airlines_ml.geography import compute_country_features, enrich_countries
from airlines_ml.inputs import load_sources
from airlines_ml.lof import fit_lof
from airlines_ml.preprocessing import fit_preprocessor
from airlines_ml.thresholds import (
    find_elbow_threshold,
    write_score_plots,
    write_trimmed_histogram,
)

AUDIT_FILES = (
    "flights_before.parquet",
    "flights_after.parquet",
    "passengers_before.parquet",
    "passengers_after.parquet",
)
PASSENGER_SOURCES_FILE = "passenger_sources.parquet"
FLIGHT_COLUMNS = {
    "PassengerDocument": "Идентификатор пассажира",
    "FlightDate": "Дата вылета",
    "FlightNumber": "Рейс",
    "Origin": "Откуда",
    "Destination": "Куда",
}


FEATURE_LABELS = {
    "document_conflict_count": "Количество конфликтующих документов",
    "visited_country_diversity": "Количество посещённых стран (логарифм)",
    "flight_count": "Активность полётов (логарифм)",
    "destination_count": "Количество направлений (логарифм)",
    "destination_rarity": "Редкость направлений",
    "median_departure_gap": "Медианный интервал между днями вылета (логарифм)",
    "gap_burstiness": "Неравномерность интервалов",
    "same_day_departure_share": "Доля повторных вылетов за день",
    "weekend_share": "Доля вылетов в выходные",
    "marketed_carrier_entropy": "Разнообразие авиакомпаний",
}


def build_spies(
    passengers: pl.DataFrame,
    passenger_names: pl.DataFrame,
    matrix: np.ndarray,
    feature_names: list[str],
    scores: np.ndarray,
    neighbors: np.ndarray,
    review_percentile: float = 99,
    *,
    score_threshold: float | None = None,
) -> pl.DataFrame:
    """Показывает отличия от соседей, не приписывая признакам причинность."""
    cutoff = (
        np.percentile(scores, review_percentile)
        if score_threshold is None
        else score_threshold
    )
    selected = np.flatnonzero(scores >= cutoff)
    differences = matrix[selected] - np.median(matrix[neighbors[selected]], axis=1)
    explanations = []
    for row in differences:
        largest = np.argsort(-np.abs(row), kind="stable")[:3]
        values = [
            f"{FEATURE_LABELS[feature_names[index]]}: {row[index]:+.3f}"
            for index in largest
        ]
        explanations.append(
            "Отличия от медианы соседей (в единицах МКР): " + "; ".join(values)
        )
    return (
        passengers[selected]
        .select("passenger_id", "departure_count", *feature_names)
        .join(
            passenger_names.select("passenger_id", "ФИО"),
            on="passenger_id",
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.Series("Оценка LOF", scores[selected]),
            pl.Series("Пояснение", explanations, dtype=pl.String),
            pl.col("ФИО")
            .map_elements(unidecode, return_dtype=pl.String)
            .str.to_uppercase()
            .str.replace_all(r"\s+", " ")
            .str.strip_chars(),
        )
        .rename(
            {
                "passenger_id": "Идентификатор пассажира",
                "departure_count": "Количество вылетов",
                **{name: FEATURE_LABELS[name] for name in feature_names},
            }
        )
        .select(
            "Идентификатор пассажира",
            "ФИО",
            "Количество вылетов",
            "Оценка LOF",
            *[FEATURE_LABELS[name] for name in feature_names],
            "Пояснение",
        )
        .sort(["Оценка LOF", "Идентификатор пассажира"], descending=[True, False])
    )


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
    n_neighbors: int = 20,
    review_percentile: float | None = None,
    threshold_method: str | None = None,
    batch_size: int = 1024,
    airport_reference: str | Path | None = None,
) -> Path:
    input_path = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if output.suffix != ".parquet":
        output = output / "spies.parquet"
    data_folder = input_path if input_path.is_dir() else input_path.parent
    if output.is_relative_to(data_folder):
        raise ValueError("Результат нельзя сохранять поверх исходных данных")
    if review_percentile is not None and not 0 < review_percentile < 100:
        raise ValueError("Процентиль должен находиться между 0 и 100")
    if output.name in (*AUDIT_FILES, PASSENGER_SOURCES_FILE):
        raise ValueError("Имя результата совпадает с именем выгрузки до/после очистки")
    method = threshold_method or (
        "percentile" if review_percentile is not None else "elbow"
    )
    if method not in {"elbow", "percentile"}:
        raise ValueError("Метод порога должен быть elbow или percentile")
    sources = load_sources(input_path)
    flights_before, passengers_before = build_before_exports(sources)
    resolved = resolve_passengers(sources)
    named = resolved.passengers.filter(
        pl.col("ФИО").is_not_null() & (pl.col("ФИО").str.strip_chars() != "")
    )
    excluded = resolved.passengers.height - named.height
    print(f"Исключено историй без ФИО после объединения: {excluded}")
    resolved.source.frame = resolved.source.frame.join(
        named.select(pl.col("passenger_id").alias("PassengerDocument")),
        on="PassengerDocument",
        how="semi",
    )
    resolved.source.identities = named.select(
        pl.col("passenger_id").alias("PassengerDocument")
    )
    flights_after = (
        clean_departures(resolved.source.frame)
        .rename(
            {
                "passenger_id": "PassengerDocument",
                "date": "FlightDate",
                "flight": "FlightNumber",
                "origin": "Origin",
                "destination": "Destination",
            }
        )
        .join(
            named.select(pl.col("passenger_id").alias("PassengerDocument"), "ФИО"),
            on="PassengerDocument",
            how="left",
            validate="m:1",
        )
        .select(
            "PassengerDocument",
            "ФИО",
            "FlightDate",
            "FlightNumber",
            "Origin",
            "Destination",
        )
    )
    enriched = enrich_countries(resolved.source.frame, airport_reference)
    countries = compute_country_features(
        enriched.filter(
            pl.col("FlightDate").is_not_null()
            & pl.col("FlightNumber").str.contains(r"^[A-Z0-9]{2}\d{1,4}[A-Z]?$")
        )
    )
    passengers = (
        prepare_source(resolved.source)
        .filter(pl.col("departure_count") >= 3)
        .join(countries, on="passenger_id", how="left", validate="1:1")
        .join(
            named.select("passenger_id", "document_conflict_count"),
            on="passenger_id",
            how="left",
            validate="1:1",
        )
    )
    print(f"Историй с ФИО, прошедших отбор по вылетам: {passengers.height}")
    print(
        "Историй с конфликтующими документами: "
        f"{passengers.filter(pl.col('document_conflict_count') > 0).height}"
    )
    matrix, feature_names = fit_preprocessor(passengers)
    print("Признаки LOF: " + ", ".join(feature_names))
    scores, neighbors = fit_lof(
        matrix, device=device, n_neighbors=n_neighbors, batch_size=batch_size
    )
    percentile = review_percentile if review_percentile is not None else 99.0
    fallback_reason = None
    if method == "elbow":
        try:
            cutoff = find_elbow_threshold(scores)
        except ValueError as exc:
            fallback_reason = str(exc)
            method = "percentile_fallback"
            cutoff = float(np.percentile(scores, percentile))
            if np.ptp(scores) == 0:
                cutoff = float(np.nextafter(scores[0], np.inf))
                method = "no_separation"
    else:
        cutoff = float(np.percentile(scores, percentile))
    contamination = float(np.mean(scores >= cutoff))
    print(f"Порог LOF: {cutoff:.6f}; метод: {method}; доля отбора: {contamination:.4%}")
    if fallback_reason:
        print(f"Локоть не найден: {fallback_reason}")
    spies = build_spies(
        passengers,
        resolved.passengers,
        matrix,
        feature_names,
        scores,
        neighbors,
        percentile,
        score_threshold=cutoff,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        output.name: spies,
        PASSENGER_SOURCES_FILE: resolved.passenger_sources,
        "flights_before.parquet": flights_before.rename(FLIGHT_COLUMNS),
        "flights_after.parquet": flights_after.rename(FLIGHT_COLUMNS),
        "passengers_before.parquet": passengers_before.rename(
            {"passenger_id": "Идентификатор пассажира"}
        ),
        "passengers_after.parquet": named.select("passenger_id", "ФИО").rename(
            {"passenger_id": "Идентификатор пассажира"}
        ),
    }
    for filename, table in tables.items():
        target = output.parent / filename
        with tempfile.NamedTemporaryFile(
            dir=output.parent, suffix=".parquet", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            table.write_parquet(temporary_path, compression="zstd")
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)
    analysis_folder = output.parent / "lof_analysis"
    analysis_folder.mkdir(parents=True, exist_ok=True)
    passengers.select(
        pl.col("passenger_id").alias("Идентификатор пассажира")
    ).with_columns(
        pl.Series("lof_score", scores),
        pl.Series("negative_outlier_factor_", -scores),
    ).write_parquet(analysis_folder / "lof_scores.parquet", compression="zstd")
    sensitivity = []
    for grid in (501, 1001, 2001):
        for value in (1.0, 2.0, 5.0):
            try:
                candidate = find_elbow_threshold(
                    scores, sensitivity=value, grid_size=grid
                )
                sensitivity.append(
                    {
                        "grid_size": grid,
                        "sensitivity": value,
                        "threshold": candidate,
                        "contamination": float(np.mean(scores >= candidate)),
                    }
                )
            except ValueError:
                sensitivity.append(
                    {
                        "grid_size": grid,
                        "sensitivity": value,
                        "threshold": None,
                        "contamination": None,
                    }
                )
    pl.DataFrame(sensitivity).write_parquet(
        analysis_folder / "elbow_sensitivity.parquet"
    )
    pl.DataFrame(
        {
            "method": [method],
            "positive_lof_threshold": [cutoff],
            "negative_outlier_factor_threshold": [-cutoff],
            "contamination": [contamination],
            "eligible_passengers": [len(scores)],
            "selected_passengers": [spies.height],
            "n_neighbors": [n_neighbors],
            "device": [device],
            "fallback_reason": [fallback_reason],
            "previous_99_percentile_threshold": [float(np.percentile(scores, 99))],
        }
    ).write_parquet(analysis_folder / "threshold.parquet")
    write_score_plots(scores, cutoff, analysis_folder)
    if np.count_nonzero(scores < cutoff) >= 2:
        write_trimmed_histogram(scores, cutoff, analysis_folder)
    return output
