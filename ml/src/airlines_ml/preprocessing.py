"""Reference-median imputation, Spearman pruning and median/IQR scaling."""

import numpy as np
import polars as pl
from scipy.stats import rankdata

from airlines_ml.features import FEATURE_NAMES


def fit_preprocessor(frame: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    if frame.is_empty():
        raise ValueError("No eligible passengers available for preprocessing")
    columns = []
    ranks: list[np.ndarray] = []
    names = []
    for name in [
        *FEATURE_NAMES,
        "visited_country_diversity",
        "document_conflict_count",
    ]:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy().astype(np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        median = np.median(values[finite])
        values = np.where(finite, values, median)
        if np.ptp(values) == 0:
            continue
        ranked = rankdata(values)
        if any(abs(np.corrcoef(ranked, other)[0, 1]) > 0.95 for other in ranks):
            continue
        iqr = np.quantile(values, 0.75) - np.quantile(values, 0.25)
        columns.append((values - median) / (iqr if iqr > 0 else 1.0))
        ranks.append(ranked)
        names.append(name)
    if not columns:
        raise ValueError(
            "No informative features: all reference features are missing or constant"
        )
    return np.column_stack(columns), names
