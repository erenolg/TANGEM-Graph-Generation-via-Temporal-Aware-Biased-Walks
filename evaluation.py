import logging
import numpy as np
import networkx as nx
import os
import secrets
import subprocess as sp
import sys
from string import ascii_uppercase, digits

import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# MMD evaluation
# ---------------------------------------------------------------------------

def mmd_evaluation(
    original_graph: nx.Graph | list[nx.Graph],
    generated_graphs: list[nx.Graph],
) -> dict[str, float]:
    """
    Compute all MMD metrics between original and generated graphs.

    Prints results and returns them as a dict for downstream logging.
    """
    if not isinstance(original_graph, list):
        original_graph = [original_graph]

    degree, clustering, spectral, orbit, motif = mmd(original_graph, generated_graphs)
    print(f"Degree:     {degree}")
    print(f"Clustering: {clustering}")
    print(f"Spectral:   {spectral}")
    print(f"Orbit:      {orbit}")
    print(f"Motif:      {motif}")

    return {
        "mmd_degree":     degree,
        "mmd_clustering": clustering,
        "mmd_spectral":   spectral,
        "mmd_orbit":      orbit,
        "mmd_motif":      motif,
    }


def mmd(
    graph_gt: list[nx.Graph],
    graph_pred: list[nx.Graph],
    plot: bool = False,
) -> tuple[float, float, float, float, float]:
    mmd_degree     = degree_stats(graph_gt, graph_pred, plot=plot)
    mmd_clustering = clustering_stats(graph_gt, graph_pred, plot=plot)
    mmd_spectral   = spectral_stats(graph_gt, graph_pred, plot=plot)

    # motif_stats and orbit_stats_all require the ORCA binary.
    # If it is missing (not yet compiled), degrade gracefully instead of crashing.
    try:
        mmd_motif = motif_stats(graph_gt, graph_pred, plot=plot)
        mmd_orbit = orbit_stats_all(graph_gt, graph_pred, plot=plot)
    except Exception as exc:
        logger.warning(
            "Motif and orbit MMD scores skipped (%s: %s). "
            "To enable them, compile orca/orca%s.",
            type(exc).__name__, exc,
            ".exe" if sys.platform == "win32" else "",
        )
        mmd_motif = float("nan")
        mmd_orbit = float("nan")

    return mmd_degree, mmd_clustering, mmd_spectral, mmd_orbit, mmd_motif


# ---------------------------------------------------------------------------
# Per-statistic MMD functions
# ---------------------------------------------------------------------------

def degree_stats(
    graph_ref_list: list[nx.Graph],
    graph_pred_list: list[nx.Graph],
    plot: bool = False,
) -> float:
    sample_ref = [np.array(nx.degree_histogram(G)) for G in graph_ref_list]
    sample_pred = [np.array(nx.degree_histogram(G)) for G in graph_pred_list]
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, plot=plot)


def clustering_stats(
    graph_ref_list: list[nx.Graph],
    graph_pred_list: list[nx.Graph],
    plot: bool = False,
    bins: int = 100,
) -> float:
    def _hist(G: nx.Graph) -> np.ndarray:
        coeff_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(coeff_list, bins=bins, range=(0.0, 1.0), density=False)
        return hist

    sample_ref = [_hist(G) for G in graph_ref_list]
    sample_pred = [_hist(G) for G in graph_pred_list]
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10, plot=plot)


motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts:'


def motif_stats(
    graph_ref_list: list[nx.Graph],
    graph_pred_list: list[nx.Graph],
    plot: bool = False,
    motif_type: str = '3path',
    ground_truth_match: int | None = None,
) -> float:
    indices = motif_to_indices[motif_type]
    graph_pred_list = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    def _counts(graphs: list[nx.Graph]) -> list[np.ndarray]:
        result = []
        for G in graphs:
            orbit_counts = orca(G)
            motif_counts = np.sum(orbit_counts[:, indices], axis=1)
            result.append(np.array([np.sum(motif_counts) / G.number_of_nodes()]))
        return result

    total_counts_ref = np.array(_counts(graph_ref_list))
    total_counts_pred = np.array(_counts(graph_pred_list))
    return compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, plot=plot)


def orbit_stats_all(
    graph_ref_list: list[nx.Graph],
    graph_pred_list: list[nx.Graph],
    plot: bool = False,
) -> float:
    graph_pred_list = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    def _orbit_counts(G: nx.Graph) -> np.ndarray:
        orbit_counts = orca(G)
        return np.sum(orbit_counts, axis=0) / G.number_of_nodes()

    total_counts_ref = np.array([_orbit_counts(G) for G in graph_ref_list])
    total_counts_pred = np.array([_orbit_counts(G) for G in graph_pred_list])
    return compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, sigma=30.0, plot=plot)


def spectral_stats(
    graph_ref_list: list[nx.Graph],
    graph_pred_list: list[nx.Graph],
    plot: bool = False,
) -> float:
    sample_ref = [_spectral_pmf(G) for G in graph_ref_list]
    sample_pred = [_spectral_pmf(G) for G in graph_pred_list]
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, is_hist=False, plot=plot)


def _spectral_pmf(G: nx.Graph) -> np.ndarray:
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    return spectral_pmf / spectral_pmf.sum()


# ---------------------------------------------------------------------------
# MMD kernel / distance utilities
# ---------------------------------------------------------------------------

def compute_mmd(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    kernel,
    is_hist: bool = True,
    plot: bool = False,
    *args,
    **kwargs,
) -> float:
    if is_hist:
        samples1 = [s / np.sum(s) for s in samples1]
        samples2 = [s / np.sum(s) for s in samples2]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        axes[0].hist(samples1, bins=30, alpha=0.7, label='Original')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Original Histogram')
        axes[0].legend()
        axes[1].hist(samples2, bins=30, alpha=0.7, label='Generated')
        axes[1].set_xlabel('Value')
        axes[1].set_title('Generated Histogram')
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )


def disc(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    kernel,
    *args,
    **kwargs,
) -> float:
    total = 0.0
    for s1 in samples1:
        for s2 in samples2:
            total += kernel(s1, s2, *args, **kwargs)
    return total / (len(samples1) * len(samples2))


def gaussian_tv(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    support_size = max(len(x), len(y))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if len(x) < support_size:
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < support_size:
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


# ---------------------------------------------------------------------------
# ORCA orbit counting
# ---------------------------------------------------------------------------

def edge_list_reindexed(G: nx.Graph) -> list[tuple[int, int]]:
    """Return the edge list of G with nodes re-indexed to 0…N-1."""
    id2idx = {str(u): i for i, u in enumerate(G.nodes())}
    return [(id2idx[str(u)], id2idx[str(v)]) for u, v in G.edges()]


def _orca_bin_path() -> str:
    """Return the platform-appropriate path to the ORCA binary."""
    name = "orca.exe" if sys.platform == "win32" else "orca"
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "orca", name)


def orca(graph: nx.Graph) -> np.ndarray:
    """Run the ORCA orbit-counting tool on `graph` and return the orbit count matrix."""
    orca_bin = _orca_bin_path()
    if not os.path.exists(orca_bin):
        raise FileNotFoundError(
            f"ORCA binary not found at '{orca_bin}'. "
            "Compile the C++ source in the orca/ directory:\n"
            "  Linux/Mac : g++ -O2 -o orca/orca orca/orca.cpp\n"
            "  Windows   : g++ -O2 -o orca/orca.exe orca/orca.cpp  (MinGW)\n"
            "              cl /O2 /Fe:orca\\orca.exe orca\\orca.cpp  (MSVC)"
        )

    tmp_fname = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for _ in range(8))}.txt',
    )

    with open(tmp_fname, 'w') as f:
        f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for u, v in edge_list_reindexed(graph):
            f.write(f"{u} {v}\n")

    output = sp.check_output([orca_bin, 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([
        list(map(int, row.strip().split()))
        for row in output.strip('\n').split('\n')
    ])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts
