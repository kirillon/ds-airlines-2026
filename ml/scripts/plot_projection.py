"""Проекция всех обучающих историй; существующие оценки LOF не меняются."""

import time
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from airlines_ml.entities import resolve_passengers
from airlines_ml.features import prepare_source
from airlines_ml.geography import compute_country_features, enrich_countries
from airlines_ml.inputs import load_sources
from airlines_ml.preprocessing import fit_preprocessor


def main() -> None:
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    folder = root / "output" / "lof_analysis"
    resolved = resolve_passengers(load_sources(root / "data"))
    named = resolved.passengers.filter(
        pl.col("ФИО").is_not_null() & (pl.col("ФИО").str.strip_chars() != "")
    )
    resolved.source.frame = resolved.source.frame.join(
        named.select(pl.col("passenger_id").alias("PassengerDocument")),
        on="PassengerDocument",
        how="semi",
    )
    resolved.source.identities = named.select(
        pl.col("passenger_id").alias("PassengerDocument")
    )
    enriched = enrich_countries(resolved.source.frame, None)
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
    matrix, names = fit_preprocessor(passengers)
    scores = pl.read_parquet(folder / "lof_scores.parquet").rename(
        {"Идентификатор пассажира": "passenger_id"}
    )
    aligned = passengers.select("passenger_id").join(
        scores, on="passenger_id", how="left", validate="1:1", maintain_order="left"
    )
    if aligned.height != scores.height or aligned["lof_score"].null_count():
        raise ValueError("Состав историй не совпадает с сохранёнными оценками LOF")
    pl.DataFrame(matrix, schema=names).with_columns(
        passengers["passenger_id"]
    ).write_parquet(folder / "training_features.parquet")
    print(f"Проекция всей матрицы {matrix.shape}; признаки: {names}", flush=True)
    projection = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=42,
        method="barnes_hut",
        n_jobs=4,
        verbose=1,
    )
    coordinates = projection.fit_transform(matrix)
    threshold = pl.read_parquet(folder / "threshold.parquet")["positive_lof_threshold"][
        0
    ]
    values = aligned["lof_score"].to_numpy()
    flagged = values >= threshold
    aligned.with_columns(
        pl.Series("tsne_1", coordinates[:, 0]),
        pl.Series("tsne_2", coordinates[:, 1]),
        pl.Series("above_threshold", flagged),
    ).write_parquet(folder / "tsne_projection.parquet")
    figure = Figure(figsize=(16, 7), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    left.scatter(
        *coordinates[~flagged].T,
        s=2,
        alpha=0.25,
        c="#477ba9",
        rasterized=True,
        label=f"Ниже порога: {(~flagged).sum():,}",
    )
    left.scatter(
        *coordinates[flagged].T,
        s=3,
        alpha=0.6,
        c="#d44b35",
        rasterized=True,
        label=f"Выше порога: {flagged.sum():,}",
    )
    left.legend(markerscale=4, loc="upper right")
    left.set_title(f"Текущий порог LOF ≥ {threshold:.3f}")
    upper = np.quantile(values, 0.99)
    scatter = right.scatter(
        *coordinates.T,
        c=values,
        s=2,
        alpha=0.65,
        cmap="viridis",
        vmin=values.min(),
        vmax=upper,
        rasterized=True,
    )
    figure.colorbar(
        scatter, ax=right, label="LOF (цвет ограничен 99-м процентилем)", extend="max"
    )
    right.set_title("Непрерывная оценка аномальности")
    for axis in (left, right):
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"t-SNE: все {len(matrix):,} историй, {len(names)} признаков LOF\n"
        "Медианное заполнение и масштабирование по МКР · perplexity=30 · seed=42",
        fontsize=15,
    )
    figure.savefig(folder / "tsne_projection.png", dpi=180)
    elapsed = time.monotonic() - started
    pl.DataFrame(
        {
            "method": ["t-SNE Barnes-Hut"],
            "sklearn_version": [sklearn.__version__],
            "init": ["pca"],
            "learning_rate": ["auto"],
            "max_iter": [1000],
            "angle": [0.5],
            "rows": [len(matrix)],
            "features": [names],
            "perplexity": [30],
            "seed": [42],
            "iterations": [projection.n_iter_],
            "kl_divergence": [projection.kl_divergence_],
            "runtime_seconds": [elapsed],
            "threshold": [threshold],
            "flagged": [int(flagged.sum())],
        }
    ).write_parquet(folder / "tsne_parameters.parquet")
    print(f"Готово: {elapsed:.1f} с; {flagged.sum()} выше порога", flush=True)


if __name__ == "__main__":
    main()
