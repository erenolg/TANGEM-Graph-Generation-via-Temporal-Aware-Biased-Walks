"""
The script for semantic-filter effect (Research Question-3)

  1. Loads CiteSeer (LCC only).
  2. Iterates over noise_ratios, injecting synthetic inter-class edges each time.
  3. For each noisy graph, runs two walk strategies:
       • Structural    – standard Node2Vec (p/q bias only)
       • Feature-Guided – Node2Vec × cosine-similarity bias
"""

import argparse
from dataclasses import dataclass, field
from typing import List

import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Walk hyperparameters
    walk_length: int = 10
    walks_per_node: int = 10
    p: float = 0.5
    q: float = 1.0
    sim_weight: float = 0.5          # Feature similarity influence [0.0 – 1.0]
    feature_smoothing_steps: int = 1 # 0 = raw (~h_feat 0.17), 1 = ~0.5, 2 = ~0.7

    noise_ratios: List[float] = field(
        default_factory=lambda: [0.0, 0.03, 0.06, 0.09, 0.12]
    )
    seed: int = 42
    data_root: str = "./data"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_citeseer(root: str = "./data"):
    """
    Load CiteSeer via torch_geometric, extract the LCC, and return a remapped
    NetworkX graph together with node features and labels aligned to new indices.
    """
    dataset = Planetoid(root=root, name="CiteSeer")
    data = dataset[0]

    G_full = to_networkx(data, to_undirected=True)
    features_all = data.x.numpy()
    labels_all   = data.y.numpy()

    lcc_nodes = max(nx.connected_components(G_full), key=len)
    G_lcc     = G_full.subgraph(lcc_nodes).copy()

    node_list = sorted(lcc_nodes)
    node_map  = {old: new for new, old in enumerate(node_list)}
    G         = nx.relabel_nodes(G_lcc, node_map)

    features = features_all[node_list]
    labels   = labels_all[node_list]

    print(
        f"[Data] CiteSeer LCC: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, "
        f"{len(set(labels))} classes"
    )
    return G, features, labels


def inject_noise(G: nx.Graph, labels: np.ndarray, noise_ratio: float, seed: int = 42):
    """
    Add inter-class edges equal to `noise_ratio * |E_original|`.
    Returns (G_noisy, noisy_edges) where noisy_edges is a frozenset of
    (min_u, max_v) tuples.
    """
    if noise_ratio == 0.0:
        return G.copy(), frozenset()

    rng      = np.random.RandomState(seed)
    G_noisy  = G.copy()
    nodes    = np.array(list(G.nodes()))
    n_target = int(G.number_of_edges() * noise_ratio)

    noisy_edges: set = set()
    attempts        = 0
    max_attempts    = n_target * 500

    while len(noisy_edges) < n_target and attempts < max_attempts:
        u, v = rng.choice(nodes, size=2, replace=False)
        attempts += 1
        if labels[u] != labels[v] and not G_noisy.has_edge(u, v):
            G_noisy.add_edge(u, v)
            noisy_edges.add((min(u, v), max(u, v)))

    actual = len(noisy_edges)
    print(
        f"[Noise] ratio={noise_ratio:.0%}: "
        f"added {actual}/{n_target} noisy edges "
        f"(graph now has {G_noisy.number_of_edges()} edges)"
    )
    return G_noisy, frozenset(noisy_edges)


def smooth_features(G: nx.Graph, features: np.ndarray, steps: int = 1) -> np.ndarray:
    """
    Graph-based feature smoothing: X' = D̃^{-1} Ã X, applied `steps` times.
    Features are L2-normalised after each step.
    """
    if steps == 0:
        return features

    nodes = sorted(G.nodes())
    X = features.copy().astype(float)

    for _ in range(steps):
        X_new = X.copy()
        for v in nodes:
            nbrs = list(G.neighbors(v))
            if nbrs:
                X_new[v] = np.mean(X[nbrs + [v]], axis=0)
        X = normalize(X_new, norm="l2")

    return X


def compute_feature_homophily(G: nx.Graph, features: np.ndarray) -> float:
    """Average cosine similarity between feature vectors of connected node pairs."""
    n_edges = G.number_of_edges()
    if n_edges == 0:
        return 0.0
    feat_norm = normalize(features, norm="l2")
    sims = [float(np.dot(feat_norm[u], feat_norm[v])) for u, v in G.edges()]
    return float(np.mean(sims))


def compute_homophily(G: nx.Graph, labels: np.ndarray) -> float:
    """Edge homophily: fraction of edges whose endpoints share the same class label."""
    n_edges = G.number_of_edges()
    if n_edges == 0:
        return 0.0
    same = sum(1 for u, v in G.edges() if labels[u] == labels[v])
    return same / n_edges


# ── Walker ────────────────────────────────────────────────────────────────────

class RandomWalker:
    """
    Biased random walker with optional feature-guided transitions.

    Standard mode  (use_feature_bias=False):
        w(t, u, x) = α_pq(t, u, x)   [Node2Vec return/in-out bias only]

    Feature-guided mode (use_feature_bias=True):
        w(t, u, x) = α_pq(t, u, x) · ((1 − λ) + λ · sim(u, x))
        sim(u, x)  = (cosine(feat_u, feat_x) + 1) / 2  ∈ [0, 1]
    """

    def __init__(self, G, features: np.ndarray, config, use_feature_bias: bool = False):
        self.G                = G
        self.config           = config
        self.use_feature_bias = use_feature_bias

        self._feat_norm = normalize(features, norm="l2")
        self._adj       = {n: list(G.neighbors(n)) for n in G.nodes()}
        self._adj_set   = {n: set(G.neighbors(n))  for n in G.nodes()}

        self._sim: dict = {}
        if use_feature_bias:
            self._precompute_similarities()

    def _precompute_similarities(self) -> None:
        for u, v in self.G.edges():
            cos = float(np.dot(self._feat_norm[u], self._feat_norm[v]))
            sim = (cos + 1.0) / 2.0
            self._sim[(u, v)] = sim
            self._sim[(v, u)] = sim

    def _transition_weights(self, prev: int, curr: int) -> np.ndarray:
        nbrs     = self._adj[curr]
        p, q     = self.config.p, self.config.q
        prev_set = self._adj_set[prev]

        weights = np.empty(len(nbrs), dtype=np.float64)
        for i, nbr in enumerate(nbrs):
            if nbr == prev:
                bias = 1.0 / p
            elif nbr in prev_set:
                bias = 1.0
            else:
                bias = 1.0 / q

            if self.use_feature_bias:
                lam  = self.config.sim_weight
                sim  = self._sim.get((curr, nbr), 0.5)
                bias *= (1.0 - lam) + lam * sim

            weights[i] = bias

        return weights

    def _walk(self, start: int, rng: np.random.RandomState) -> list:
        walk = [start]

        while len(walk) < self.config.walk_length:
            curr = walk[-1]
            nbrs = self._adj[curr]
            if not nbrs:
                break

            if len(walk) == 1:
                nxt = rng.choice(nbrs)
            else:
                w = self._transition_weights(walk[-2], curr)
                s = w.sum()
                p = w / s if s > 0 else np.ones(len(nbrs)) / len(nbrs)
                nxt = rng.choice(nbrs, p=p)

            walk.append(nxt)

        return walk

    def generate_walks(self) -> list:
        """Generate `walks_per_node` walks from every node, shuffled each pass."""
        rng   = np.random.RandomState(self.config.seed)
        nodes = list(self.G.nodes())
        walks = []

        for _ in range(self.config.walks_per_node):
            rng.shuffle(nodes)
            for n in nodes:
                walks.append(self._walk(n, rng))

        return walks


# ── Evaluate ──────────────────────────────────────────────────────────────────

def safe_div(numerator: int, denominator: int, default: float = 0.0) -> float:
    return numerator / denominator if denominator > 0 else default


def compute_walk_metrics(
    walks: list,
    labels: np.ndarray,
    noisy_edges: frozenset,
    n_total_nodes: int,
) -> dict:
    same_label_steps   = 0
    total_label_steps  = 0
    noisy_traversals   = 0
    total_traversals   = 0
    per_walk_coverage  = []

    for walk in walks:
        if not walk:
            continue

        start_label = labels[walk[0]]
        per_walk_coverage.append(safe_div(len(set(walk)), len(walk)))

        same_label_steps  += sum(1 for n in walk if labels[n] == start_label)
        total_label_steps += len(walk)

        for i in range(len(walk) - 1):
            u, v  = walk[i], walk[i + 1]
            edge  = (min(u, v), max(u, v))
            if edge in noisy_edges:
                noisy_traversals += 1
            total_traversals += 1

    return {
        "label_purity":         safe_div(same_label_steps,  total_label_steps),
        "noisy_edge_adherence": safe_div(noisy_traversals,  total_traversals),
        "coverage":             float(np.mean(per_walk_coverage)) if per_walk_coverage else 0.0,
    }


# ── Constants ─────────────────────────────────────────────────────────────────

CASES = [
    ("Structural",     False),
    ("Feature-Guided", True),
]

# ── Core experiment loop ──────────────────────────────────────────────────────

def run_experiment(config: Config) -> None:
    G_clean, features, labels = load_citeseer(config.data_root)
    n_nodes = G_clean.number_of_nodes()

    if config.feature_smoothing_steps > 0:
        features = smooth_features(G_clean, features, steps=config.feature_smoothing_steps)
        h_raw = compute_feature_homophily(G_clean, features)
        print(f"[Features] smoothing steps={config.feature_smoothing_steps}  →  feat_homophily={h_raw:.4f}")

    for noise_ratio in config.noise_ratios:
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  Noise ratio: {noise_ratio:.0%}")
        print(sep)

        G_noisy, noisy_edges = inject_noise(G_clean, labels, noise_ratio, seed=config.seed)
        label_h = compute_homophily(G_noisy, labels)
        feat_h  = compute_feature_homophily(G_noisy, features)
        print(f"  Label homophily: {label_h:.4f}   Feature homophily: {feat_h:.4f}")

        for case_name, use_feature_bias in CASES:
            print(f"\n[{case_name}] generating walks …")
            walker = RandomWalker(G_noisy, features, config, use_feature_bias=use_feature_bias)
            walks  = walker.generate_walks()

            walk_m = compute_walk_metrics(walks, labels, noisy_edges, n_nodes)
            print(
                f"  label_purity          = {walk_m['label_purity']:.4f}\n"
                f"  noisy_edge_adherence  = {walk_m['noisy_edge_adherence']:.4f}\n"
                f"  coverage              = {walk_m['coverage']:.4f}"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-Guided Random Walk experiment on CiteSeer"
    )
    parser.add_argument("--walk_length",              type=int,   default=None)
    parser.add_argument("--walks_per_node",           type=int,   default=None)
    parser.add_argument("--p",                        type=float, default=None)
    parser.add_argument("--q",                        type=float, default=None)
    parser.add_argument("--sim_weight",               type=float, default=None,
                        help="Feature similarity influence [0.0–1.0]")
    parser.add_argument("--feature_smoothing_steps",  type=int,   default=None,
                        help="Graph feature smoothing rounds (0=raw, 2=default)")
    parser.add_argument("--seed",                     type=int,   default=None)
    parser.add_argument("--data_root",                type=str,   default=None)
    parser.add_argument(
        "--noise_ratios",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="Space-separated list, e.g. --noise_ratios 0.0 0.2 0.4",
    )
    return parser


if __name__ == "__main__":
    args   = _build_parser().parse_args()
    config = Config()

    for field_name, value in vars(args).items():
        if value is not None:
            setattr(config, field_name, value)

    print("\nConfig:")
    for k, v in vars(config).items():
        print(f"  {k} = {v}")
    print()

    run_experiment(config)
