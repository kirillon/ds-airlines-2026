"""Retrospective fixed-k LOF on dimensionless, preprocessed reference vectors."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class GPUUnavailableError(RuntimeError):
    """Requested CUDA computation cannot run; CPU is never selected silently."""


def fit_lof(
    matrix: np.ndarray,
    *,
    device: str = "cpu",
    n_neighbors: int = 20,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit and score the same reference population, without novelty prediction.

    Scores are density ratios, not probabilities. Fixed-k ties can select different
    equally distant neighbors across devices, so score parity assumes untied data.
    Neighbor storage is O(n*k); exact search may use backend-managed workspaces.
    """
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    values = np.asarray(matrix, dtype=np.float64, order="C")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("matrix must be two-dimensional with at least one feature")
    if not 1 <= n_neighbors < len(values) or batch_size < 1:
        raise ValueError("Нужны k >= 1, размер пакета >= 1 и не менее k+1 пассажиров")
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values")
    if device == "gpu":
        return _fit_gpu(values, n_neighbors, batch_size)
    estimator = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="brute", metric="euclidean", n_jobs=1
    ).fit(values)
    distances, indices = estimator.kneighbors(n_neighbors=n_neighbors)
    k_distances = distances[:, -1]
    reachability = np.maximum(distances, k_distances[indices])
    densities = 1.0 / (reachability.mean(axis=1) + 1e-10)
    return densities[indices].mean(axis=1) / densities, indices


def _fit_gpu(
    values: np.ndarray, k: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    # Optional imports keep CPU execution usable on systems without CUDA libraries.
    try:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors
    except (ImportError, OSError) as exc:
        raise GPUUnavailableError(
            "GPU mode requires compatible CuPy, cuML and NVIDIA CUDA libraries"
        ) from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise GPUUnavailableError("GPU mode requires an available NVIDIA CUDA GPU")
        reference = cp.asarray(values, dtype=cp.float64)
        estimator = NearestNeighbors(
            n_neighbors=k + 1,
            algorithm="brute",
            metric="euclidean",
            output_type="cupy",
        )
        estimator.fit(reference)
        size = values.shape[0]
        indices = cp.empty((size, k), dtype=cp.int64)
        distances = cp.empty((size, k), dtype=cp.float64)
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            batch_distances, batch_indices = estimator.kneighbors(
                reference[start:stop], n_neighbors=k + 1
            )
            rows = cp.arange(start, stop)[:, None]
            # A duplicate vector can displace self from the returned tie set.
            positions = cp.broadcast_to(cp.arange(k + 1), batch_indices.shape)
            positions = cp.where(batch_indices != rows, positions, k + 1)
            selected = cp.sort(positions, axis=1)[:, :k]
            indices[start:stop] = cp.take_along_axis(batch_indices, selected, axis=1)
            distances[start:stop] = cp.take_along_axis(
                batch_distances, selected, axis=1
            )
        k_distances = distances[:, -1]
        densities = cp.empty(size, dtype=cp.float64)
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            reachability = cp.maximum(
                distances[start:stop], k_distances[indices[start:stop]]
            )
            densities[start:stop] = 1.0 / (reachability.mean(axis=1) + 1e-10)
        scores = cp.empty(size, dtype=cp.float64)
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            scores[start:stop] = (
                densities[indices[start:stop]].mean(axis=1) / densities[start:stop]
            )
        cp.cuda.get_current_stream().synchronize()
        return cp.asnumpy(scores), cp.asnumpy(indices)
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise GPUUnavailableError(f"CUDA execution unavailable: {exc}") from exc
