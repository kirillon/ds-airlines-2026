from pathlib import Path

import numpy as np
from kneed import KneeLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.stats import norm


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError(
            "Elbow detection requires at least three one-dimensional scores"
        )
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("LOF scores must be finite and positive")
    return values


def find_elbow_threshold(
    scores: np.ndarray, *, sensitivity: float = 1.0, grid_size: int = 1001
) -> float:
    values = _validate_scores(scores)
    if not np.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("Elbow sensitivity must be finite and positive")
    if grid_size < 3:
        raise ValueError("Elbow quantile grid must contain at least three points")
    probabilities = np.linspace(0, 1, min(grid_size, values.size))
    quantiles = np.quantile(values, probabilities)
    spread = quantiles[-1] - quantiles[0]
    if spread <= 0:
        raise ValueError(
            "The LOF distribution is constant; an elbow cannot be identified"
        )
    normalized = (quantiles - quantiles[0]) / spread
    if np.max(probabilities - normalized) <= 1e-6:
        raise ValueError("The LOF quantile curve has no convex elbow")
    locator = KneeLocator(
        probabilities,
        quantiles,
        curve="convex",
        direction="increasing",
        S=sensitivity,
        online=False,
    )
    if locator.knee is None or not 0 < locator.knee < 1:
        raise ValueError("No interior elbow was identified in the LOF quantile curve")
    return float(np.interp(locator.knee, probabilities, quantiles))


def write_score_plots(
    scores: np.ndarray, threshold: float, folder: Path
) -> tuple[Path, Path]:
    values = _validate_scores(scores)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The LOF threshold must be finite and positive")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    histogram_path = folder / "lof_histogram.png"
    quantiles_path = folder / "lof_quantiles.png"
    negative_scores = -values
    flagged_percent = 100 * np.mean(values >= threshold)
    threshold_label = f"Порог: −{threshold:.3f}; выделено {flagged_percent:.2f}%"

    figure = Figure(figsize=(10, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.hist(negative_scores, bins=50, color="#4477AA", edgecolor="white")
    axis.axvline(-threshold, color="#CC3311", linestyle="--", label=threshold_label)
    axis.set(
        title="Распределение отрицательных оценок LOF",
        xlabel="negative_outlier_factor_ (−LOF): левее — аномальнее",
        ylabel="Количество пассажиров",
    )
    axis.legend()
    figure.savefig(histogram_path, dpi=180)
    figure.clear()

    probabilities = np.linspace(0, 1, min(1001, values.size))
    figure = Figure(figsize=(13, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].plot(100 * probabilities, np.quantile(negative_scores, probabilities))
    axes[0].axvline(flagged_percent, color="#CC3311", linestyle=":")
    axes[0].set(
        title="Квантили negative_outlier_factor_",
        xlabel="Квантиль, % (аномальные значения слева)",
        ylabel="negative_outlier_factor_ (−LOF)",
    )
    axes[1].plot(100 * probabilities, -np.quantile(values, probabilities))
    axes[1].axvline(100 - flagged_percent, color="#CC3311", linestyle=":")
    axes[1].set(
        title="Отрицательные оценки по возрастанию аномальности",
        xlabel="Квантиль положительного LOF, % (аномалии справа)",
        ylabel="−Q(LOF): это не Q(−LOF)",
    )
    for axis in axes:
        axis.axhline(-threshold, color="#CC3311", linestyle="--", label=threshold_label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(quantiles_path, dpi=180)
    figure.clear()
    return histogram_path, quantiles_path


def write_trimmed_histogram(scores: np.ndarray, threshold: float, folder: Path) -> Path:
    values = _validate_scores(scores)
    retained = -values[values < threshold]
    if retained.size < 2:
        raise ValueError("Need at least two retained scores for a trimmed histogram")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "lof_histogram_trimmed.png"
    figure = Figure(figsize=(10, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    _, edges, _ = axis.hist(
        retained,
        bins=50,
        color="#4477AA",
        edgecolor="white",
        label="Оценки после отсечения хвоста",
    )
    mean, std = float(retained.mean()), float(retained.std())
    if std > 0:
        x = np.linspace(edges[0], edges[-1], 500)
        axis.plot(
            x,
            norm.pdf(x, loc=mean, scale=std) * len(retained) * (edges[1] - edges[0]),
            color="#228833",
            linewidth=2,
            label=f"Нормальная кривая для сравнения: μ={mean:.3f}, σ={std:.3f}",
        )
    axis.axvline(
        -threshold,
        color="#CC3311",
        linestyle="--",
        label=f"Граница отсечения: −{threshold:.3f}",
    )
    axis.set(
        title=(
            "Распределение LOF без аномального хвоста\n"
            f"Оставлено {len(retained):,} оценок ({len(retained) / len(values):.2%}); "
            f"удалено {len(values) - len(retained):,}"
        ),
        xlabel="negative_outlier_factor_ (−LOF)",
        ylabel="Количество пассажиров",
    )
    axis.legend(fontsize=9)
    figure.savefig(path, dpi=180)
    figure.clear()
    return path
