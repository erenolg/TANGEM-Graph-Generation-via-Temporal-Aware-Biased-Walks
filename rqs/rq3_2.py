import numpy as np
import networkx as nx
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────────────
NUM_COMMUNITIES = 4
COMMUNITY_SIZE  = 100
P_INTRA = 0.08
P_INTER = 0.02

P = 1.0
Q = 0.1
NUM_WALKS   = 20
WALK_LENGTH = 40

FEAT_DIM        = 16
FEAT_NOISE      = 0.3
FEAT_SEPARATION = 3.0
BIAS_STRENGTH   = 2.0

EMB_DIM = 64
WINDOW  = 5
EPOCHS  = 10

SEED = 42

# ── Graph generation ───────────────────────────────────────────────────────────
def generate_sbm(sizes, p_intra, p_inter, seed=42):
    K = len(sizes)
    probs = np.full((K, K), p_inter)
    np.fill_diagonal(probs, p_intra)
    G = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed)
    communities = {node: G.nodes[node]["block"] for node in G.nodes()}
    return G, communities

# ── Feature assignment ─────────────────────────────────────────────────────────
def _make_centroids(K, dim, separation, rng):
    centroids = rng.randn(K, dim)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids *= separation
    return centroids


def assign_homophilic_features(G, communities, dim=16, noise=0.3,
                                separation=3.0, seed=42):
    rng = np.random.RandomState(seed)
    K = len(set(communities.values()))
    centroids = _make_centroids(K, dim, separation, rng)
    return {node: centroids[communities[node]] + rng.randn(dim) * noise
            for node in G.nodes()}


def assign_heterophilic_features(G, communities, dim=16, noise=0.1,
                                  separation=3.0, seed=42):
    rng = np.random.RandomState(seed)
    K = len(set(communities.values()))
    comm_nodes = defaultdict(list)
    for node, c in communities.items():
        comm_nodes[c].append(node)

    colorings = {}
    max_colors = 0
    for c in range(K):
        sub = G.subgraph(comm_nodes[c])
        col = nx.coloring.greedy_color(sub, strategy="largest_first")
        colorings[c] = col
        max_colors = max(max_colors, max(col.values()) + 1)

    centroids = _make_centroids(max_colors, dim, separation, rng)
    features = {}
    for c in range(K):
        for node in comm_nodes[c]:
            features[node] = centroids[colorings[c][node]] + rng.randn(dim) * noise
    return features


def assign_random_features(G, dim=16, seed=42):
    rng = np.random.RandomState(seed)
    return {node: rng.randn(dim) for node in G.nodes()}

# ── Walk utilities ─────────────────────────────────────────────────────────────
def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)


def _feature_weight(feat_u, feat_v, bias_strength):
    sim = _cosine_sim(feat_u, feat_v)
    return max(0.01, (sim + 1.0) / 2.0) ** bias_strength


def biased_walk(G, features, start, walk_length,
                p=1.0, q=1.0, feature_bias=False, bias_strength=1.0):
    walk = [start]
    neighbors = list(G.neighbors(start))
    if not neighbors:
        return walk

    if feature_bias and features is not None:
        weights = np.array([_feature_weight(features[start], features[nbr], bias_strength)
                            for nbr in neighbors])
        weights /= weights.sum()
        nxt = neighbors[np.random.choice(len(neighbors), p=weights)]
    else:
        nxt = neighbors[np.random.randint(len(neighbors))]
    walk.append(nxt)

    for _ in range(walk_length - 2):
        cur, prev = walk[-1], walk[-2]
        neighbors = list(G.neighbors(cur))
        if not neighbors:
            break
        prev_nbrs = set(G.neighbors(prev))
        alpha = np.array([
            1.0 / p if nbr == prev else (1.0 if nbr in prev_nbrs else 1.0 / q)
            for nbr in neighbors
        ])
        weights = alpha.copy()
        if feature_bias and features is not None:
            feat_w = np.array([_feature_weight(features[cur], features[nbr], bias_strength)
                               for nbr in neighbors])
            weights *= feat_w
        weights /= weights.sum()
        walk.append(neighbors[np.random.choice(len(neighbors), p=weights)])

    return walk


def generate_walks(G, features, num_walks, walk_length,
                   p=1.0, q=1.0, feature_bias=False,
                   bias_strength=1.0, seed=42):
    np.random.seed(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            w = biased_walk(G, features, node, walk_length,
                            p=p, q=q, feature_bias=feature_bias,
                            bias_strength=bias_strength)
            walks.append([str(n) for n in w])
    return walks

# ── Embedding & evaluation ─────────────────────────────────────────────────────
def train_embeddings(walks, dim=64, window=5, epochs=10, seed=42):
    model = Word2Vec(sentences=walks, vector_size=dim, window=window,
                     min_count=0, sg=1, workers=4, epochs=epochs, seed=seed)
    return model


def evaluate(embeddings, labels, K):
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    pred = km.fit_predict(embeddings)
    return {
        "NMI":        normalized_mutual_info_score(labels, pred),
        "ARI":        adjusted_rand_score(labels, pred),
        "Silhouette": silhouette_score(embeddings, labels),
        "pred":       pred,
    }

# ── t-SNE plot & save ──────────────────────────────────────────────────────────
def plot_and_save_tsne(all_embeddings, cond_names, display_titles, labels, K, path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#999999",
    })

    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    tsne_coords = {}
    for i, (name, title) in enumerate(zip(cond_names, display_titles)):
        ax = axes_flat[i]
        emb = all_embeddings[name]
        emb_2d = TSNE(n_components=2, random_state=42, perplexity=30,
                      init="pca", learning_rate="auto").fit_transform(emb)
        tsne_coords[name] = emb_2d

        for c in range(K):
            mask = labels == c
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       s=40, color=colors[c % len(colors)],
                       alpha=0.7, label=f"Cluster {c+1}",
                       edgecolors="white", linewidths=0.3)

        ax.set_title(title, fontsize=20, weight="bold", pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[saved] {path}")
    return tsne_coords

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Graph
    sizes = [COMMUNITY_SIZE] * NUM_COMMUNITIES
    G, communities = generate_sbm(sizes, P_INTRA, P_INTER, seed=SEED)
    K = NUM_COMMUNITIES
    nodes_sorted = sorted(G.nodes())
    labels = np.array([communities[n] for n in nodes_sorted])

    print("=" * 60)
    print("Feature-Guided Biased Walks: Homophily vs Heterophily")
    print("=" * 60)
    print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}  |  Communities: {K}")
    print(f"Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.1f}")
    print()

    # Features
    feat_homo   = assign_homophilic_features(G, communities, dim=FEAT_DIM,
                    noise=FEAT_NOISE, separation=FEAT_SEPARATION, seed=SEED)
    feat_hetero = assign_heterophilic_features(G, communities, dim=FEAT_DIM,
                    noise=FEAT_NOISE, separation=FEAT_SEPARATION, seed=SEED)
    feat_rand   = assign_random_features(G, dim=FEAT_DIM, seed=SEED)
    print("Features assigned (homophilic / heterophilic / random).")

    # Walk conditions
    conditions = [
        ("Structural (no bias)",    None,        False),
        ("Feature-Biased (Homo)",   feat_homo,   True),
        ("Feature-Biased (Random)", feat_rand,   True),
        ("Feature-Biased (Hetero)", feat_hetero, True),
    ]
    display_titles = ["Pure Structure", "Homophilic", "Random", "Heterophilic"]

    # Walks + embeddings
    all_embeddings = {}
    for name, feats, use_bias in conditions:
        print(f"Generating walks: {name} ...", end=" ", flush=True)
        walks = generate_walks(G, feats, NUM_WALKS, WALK_LENGTH,
                               p=P, q=Q, feature_bias=use_bias,
                               bias_strength=BIAS_STRENGTH, seed=SEED)
        print(f"{len(walks)} walks  |  training ...", end=" ", flush=True)
        model = train_embeddings(walks, dim=EMB_DIM, window=WINDOW,
                                 epochs=EPOCHS, seed=SEED)
        emb = np.array([model.wv[str(n)] for n in nodes_sorted])
        all_embeddings[name] = emb
        print("done")

    # Evaluation
    print()
    print(f"{'Condition':<30} {'NMI':>8} {'ARI':>8} {'Silhouette':>10}")
    print("-" * 58)
    results = {}
    for name, emb in all_embeddings.items():
        res = evaluate(emb, labels, K)
        results[name] = res
        print(f"{name:<30} {res['NMI']:>8.4f} {res['ARI']:>8.4f} {res['Silhouette']:>10.4f}")

    # t-SNE
    print()
    cond_names = list(all_embeddings.keys())
    print("Computing t-SNE embeddings and saving ...")
    tsne_coords = plot_and_save_tsne(
        all_embeddings, cond_names, display_titles,
        labels, K, path="tsne_results.png"
    )


    print()
    print("Done.")


if __name__ == "__main__":
    main()
