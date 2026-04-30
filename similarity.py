import numpy as np
import networkx as nx


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def _cosine_safe(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _dtw_distance(a: np.ndarray, b: np.ndarray, window: int = 0) -> float:
    """
    DTW distance between 1-D sequences a and b.

    window : Sakoe-Chiba band half-width.
             0 = unconstrained full DTW (O(T^2) — slow for long series).
             Recommend 10-50 for T > 200.
    """
    n, m = len(a), len(b)
    w = window if window > 0 else max(n, m)
    w = max(w, abs(n - m))  # band must be wide enough to reach the endpoint

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(np.sqrt(dtw[n, m]))


def _dtw_safe(a: np.ndarray, b: np.ndarray, window: int = 0) -> float:
    """DTW distance converted to similarity in [0, 1] via 1 / (1 + dist/T)."""
    dist = _dtw_distance(a, b, window)
    return 1.0 / (1.0 + dist / max(len(a), 1))


def _xcorr_safe(a: np.ndarray, b: np.ndarray, max_lag: int = 0) -> float:
    """
    Peak normalized cross-correlation between 1-D sequences a and b.

    Searches over lags in [-max_lag, +max_lag] (0 = all lags).
    Returns similarity in [0, 1] via (peak_ncc + 1) / 2.
    """
    n = len(a)
    a_z = a - a.mean()
    b_z = b - b.mean()
    denom = a_z.std() * b_z.std() * n
    if denom == 0:
        return 0.5  # no variation — treat as neutral similarity
    corr = np.correlate(a_z, b_z, mode='full') / denom
    center = n - 1
    if max_lag > 0:
        lo = max(0, center - max_lag)
        hi = min(len(corr) - 1, center + max_lag)
        corr = corr[lo:hi + 1]
    peak = float(np.max(corr))
    return (peak + 1.0) / 2.0


def precompute_feature_similarity(
    G: nx.Graph,
    features: np.ndarray,
    method: str = 'auto',
    dtw_window: int = 0,
    xcorr_max_lag: int = 0,
) -> dict:
    """
    Precomputes pairwise similarity s(u,v) for every edge in G, normalised to [0,1].

    Feature layouts
        (T, N, F)  temporal — default method: pearson
        (N, F)     static   — default method: cosine

    method
        'auto'    — pearson for temporal, cosine for static (original behaviour)
        'pearson' — mean Pearson correlation across F features  (temporal only)
        'cosine'  — cosine similarity over the feature vector   (static only)
        'dtw'     — mean DTW-based similarity across F features (temporal only)
        'xcorr'   — peak normalized cross-correlation          (temporal only)

    dtw_window
        Sakoe-Chiba half-width for DTW; 0 = unconstrained full DTW.
    xcorr_max_lag
        Maximum lag (in timesteps) for cross-correlation search; 0 = all lags.
    """
    sim_dict = {}

    if features.ndim == 3:
        T, N, F = features.shape
        eff_method = 'pearson' if method == 'auto' else method

        if eff_method not in ('pearson', 'dtw', 'xcorr'):
            raise ValueError(
                f"method='{method}' is not supported for temporal (T,N,F) features. "
                "Choose 'pearson', 'dtw', 'xcorr', or 'auto'."
            )

        for u, v in G.edges():
            if eff_method == 'pearson':
                r = float(np.mean([
                    _pearson_safe(features[:, u, f], features[:, v, f])
                    for f in range(F)
                ]))
                s_uv = (r + 1.0) / 2.0
            elif eff_method == 'xcorr':
                s_uv = float(np.mean([
                    _xcorr_safe(features[:, u, f], features[:, v, f], xcorr_max_lag)
                    for f in range(F)
                ]))
            else:  # dtw
                s_uv = float(np.mean([
                    _dtw_safe(features[:, u, f], features[:, v, f], dtw_window)
                    for f in range(F)
                ]))
            sim_dict[(u, v)] = s_uv
            sim_dict[(v, u)] = s_uv

    elif features.ndim == 2:
        N, F = features.shape
        eff_method = 'cosine' if method == 'auto' else method

        if eff_method not in ('cosine',):
            raise ValueError(
                f"method='{method}' is not supported for static (N,F) features. "
                "DTW and xcorr require temporal (T,N,F) features with a meaningful time axis. "
                "Choose 'cosine' or 'auto'."
            )

        for u, v in G.edges():
            s_uv = _cosine_safe(features[u], features[v])
            sim_dict[(u, v)] = s_uv
            sim_dict[(v, u)] = s_uv

    else:
        raise ValueError(f"Unsupported feature shape: {features.shape}")

    return sim_dict
