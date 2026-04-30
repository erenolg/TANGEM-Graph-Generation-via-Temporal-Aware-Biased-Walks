import random
import logging
import numpy as np
import networkx as nx
from collections import deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk strategies
# ---------------------------------------------------------------------------

def random_walk(G: nx.Graph, source_node: int, walk_length: int) -> list[int]:
    """
    Simple random walk of length `walk_length` starting from `source_node`.
    If a dead-end is reached (no neighbours), the walk stays at the current node
    for the remaining steps so the output length is always `walk_length`.
    """
    walk = [source_node]
    while len(walk) < walk_length:
        current_node = walk[-1]
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            # Dead-end: pad with current node (series2graph skips consecutive duplicates)
            walk.extend([current_node] * (walk_length - len(walk)))
            break
        walk.append(random.choice(neighbors))
    return walk


def biased_random_walk(
    G: nx.Graph,
    source_node: int,
    walk_length: int,
    p: float,
    q: float,
) -> list[int]:
    """
    Node2vec-style biased random walk.

    Args:
        G: Input graph.
        source_node: Starting node id.
        walk_length: Length of the walk.
        p: Return parameter.
        q: In-out parameter.
    """
    walk = [source_node]
    while len(walk) < walk_length:
        current_node = walk[-1]
        neighbors = list(G.neighbors(current_node))

        if not neighbors:
            walk.extend([current_node] * (walk_length - len(walk)))
            break

        if len(walk) == 1:
            next_node = random.choice(neighbors)
        else:
            prev_node = walk[-2]
            probabilities = []
            for neighbor in neighbors:
                if neighbor == prev_node:
                    probabilities.append(1 / p)
                elif G.has_edge(prev_node, neighbor):
                    probabilities.append(1.0)
                else:
                    probabilities.append(1 / q)

            total = sum(probabilities)
            probabilities = [prob / total for prob in probabilities]
            next_node = random.choices(neighbors, weights=probabilities, k=1)[0]

        walk.append(next_node)
    return walk



# ---------------------------------------------------------------------------
# Walk generator (formerly generate_walks.py)
# ---------------------------------------------------------------------------

def generate_walks(
    G: nx.Graph,
    num_walks: int,
    walk_length: int,
    walk_type: str,
    **kwargs,
) -> dict[int, list[list[int]]]:
    """
    Generates walks for every (sampled) node in the graph.

    Args:
        G: Input graph.
        number_of_walks_per_node: How many walks to start from each node.
        walk_length: Length of each walk.
        walk_type: One of 'random', 'random_plus', 'brn', 'brn_plus'.
        similarity_matrix: N×N similarity matrix; required for *_plus walk types.
        ratio: Fraction of nodes to sample walks from (avoids overfitting).
        **kwargs: Extra parameters forwarded to the walk strategy (p, q, prob, degree).

    Returns:
        Dict mapping node id → list of walks.
    """
    walks: dict[int, list[list[int]]] = {}
    logger.debug("Walk type: %s | kwargs: %s", walk_type, kwargs)

    isolated = 0
    for node in G.nodes():
        if G.degree(node) == 0:
            # Isolated nodes have no neighbours — a walk cannot proceed from here.
            isolated += 1
            continue
        walks[node] = [
            _get_walk(G, node, walk_length, walk_type, **kwargs)
            for _ in range(num_walks)
        ]

    if isolated:
        logger.warning("Skipped %d isolated node(s) with degree 0.", isolated)

    return walks


def _get_walk(
    G: nx.Graph,
    node: int,
    walk_length: int,
    walk_type: str,
    **kwargs,
) -> list[int]:
    """Dispatches to the appropriate walk strategy."""
    if walk_type == "random":
        return random_walk(G, node, walk_length)

    if walk_type == "brn":
        p = kwargs.get("p", 1.0)
        q = kwargs.get("q", 1.0)
        return biased_random_walk(G, node, walk_length, p, q)


    raise ValueError(f"Invalid walk type '{walk_type}'. Choose from: random, brn")


# ---------------------------------------------------------------------------
# Feature-aware biased walk
# ---------------------------------------------------------------------------

_EPS = 1e-9


def feature_biased_random_walk(
    G: nx.Graph,
    source_node: int,
    walk_length: int,
    p: float,
    q: float,
    alpha: float,
    sim_dict: dict,
) -> list[int]:
    """
    Feature-aware Node2Vec walk.


    Args:
        G          : undirected graph
        source_node: starting node
        walk_length: number of steps
        p          : return parameter
        q          : in-out parameter
        alpha      : blend weight (1=pure Node2Vec, 0=pure feature similarity)
        sim_dict   : {(u,v): float ∈ [0,1]} from precompute_feature_similarity
    """
    walk = [source_node]

    while len(walk) < walk_length:
        u = walk[-1]
        neighbors = list(G.neighbors(u))

        if not neighbors:
            walk.extend([u] * (walk_length - len(walk)))
            break

        prev = walk[-2] if len(walk) > 1 else None
        weights = []

        for v in neighbors:
            if prev is None:
                n2v = 1.0
            elif v == prev:
                n2v = 1.0 / p
            elif G.has_edge(prev, v):
                n2v = 1.0
            else:
                n2v = 1.0 / q

            s_hat = sim_dict.get((u, v), 0.5)
            phi   = max(alpha + (1.0 - alpha) * s_hat, _EPS)
            weights.append(n2v * phi)

        walk.append(random.choices(neighbors, weights=weights, k=1)[0])

    return walk


def generate_feature_walks(
    G: nx.Graph,
    sim_dict: dict,
    num_walks: int,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
    alpha: float = 0.5,
) -> dict[int, list[list[int]]]:
    """
    Generates feature-biased walks for every non-isolated node.

    Returns:
        walks : {node_id: [walk_1, walk_2, ...]}
    """
    walks: dict[int, list[list[int]]] = {}
    for node in G.nodes():
        if G.degree(node) == 0:
            continue
        walks[node] = [
            feature_biased_random_walk(G, node, walk_length, p, q, alpha, sim_dict)
            for _ in range(num_walks)
        ]
    return walks


# ---------------------------------------------------------------------------
# Graph reconstruction from walk sequences
# ---------------------------------------------------------------------------

def seq2graph(seq: list[int]) -> nx.Graph:
    """Converts a walk sequence into a graph by adding edges between consecutive nodes."""
    G = nx.Graph()
    for u, v in zip(seq[:-1], seq[1:]):
        if u != v:
            G.add_edge(u, v)
    return G
