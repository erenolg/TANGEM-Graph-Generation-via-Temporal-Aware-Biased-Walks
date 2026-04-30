import argparse
import json
import os
import random
import numpy as np
import networkx as nx
import torch

from data import read_graph, read_feats
from similarity import precompute_feature_similarity
from walks import generate_feature_walks
from model import TransformerGG

_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')


def _load_config(config_path: str, graph: str) -> dict:
    """Flatten config sections into a single dict, applying dataset overrides."""
    with open(config_path) as f:
        cfg = json.load(f)
    flat = {**cfg['walk'], **cfg['train'], **cfg['model'], **cfg['generation']}
    flat.update(cfg.get('datasets', {}).get(graph, {}))
    return flat


def parse_args():
    p = argparse.ArgumentParser(description="Train feature-biased TransformerGG on graph walks.")

    p.add_argument('--config', type=str, default=_DEFAULT_CONFIG,
                   help="Path to config JSON (default: config.json next to main.py)")
    p.add_argument('--graph',  type=str, default='ibb1', help="Graph name")

    # Walk
    p.add_argument('--alpha',       type=float, help="Feature-structure blend (1=Node2Vec, 0=feature only)")
    p.add_argument('--num_walks',   type=int,   help="Walks per node")
    p.add_argument('--walk_length', type=int,   help="Walk length")
    p.add_argument('--p',           type=float, help="Node2Vec return parameter")
    p.add_argument('--q',           type=float, help="Node2Vec in-out parameter")

    # Training
    p.add_argument('--train_split',  type=float, help="Train/val split ratio")
    p.add_argument('--lr',           type=float, help="Learning rate")
    p.add_argument('--min_lr',       type=float, help="Minimum LR for cosine schedule")
    p.add_argument('--max_iters',    type=int,   help="Maximum training iterations")
    p.add_argument('--batch_size',   type=int,   help="Batch size")
    p.add_argument('--log_every',    type=int,   help="Print loss every N iterations")
    p.add_argument('--eval_batches', type=int,   help="Batches averaged per loss estimate")
    p.add_argument('--patience',     type=int,   help="Early stopping patience (iterations)")
    p.add_argument('--grad_clip',    type=float, help="Gradient clipping norm")

    # Model
    p.add_argument('--embed_dim',   type=int,   help="Embedding dimension")
    p.add_argument('--num_heads',   type=int,   help="Attention heads")
    p.add_argument('--num_layers',  type=int,   help="Transformer layers")
    p.add_argument('--dropout',     type=float, help="Dropout rate")

    # Generation + evaluation
    p.add_argument('--eval_graphs',     type=int,   help="Number of graphs to generate for MMD evaluation (0 = skip)")
    p.add_argument('--sequence_length', type=int,   help="Tokens per generated sequence")
    p.add_argument('--num_sequences',   type=int,   help="Sequences merged into each generated graph")
    p.add_argument('--temperature',     type=float, help="Sampling temperature for generation")

    # Output
    p.add_argument('--save_model', type=str, default=None, help="Path to save trained model (.pt)")
    p.add_argument('--save_emb',   type=str, default=None, help="Path to save node embeddings (.npy)")

    # Similarity method
    p.add_argument('--similarity',  type=str, default='auto',
                   choices=['auto', 'pearson', 'cosine', 'dtw', 'xcorr'],
                   help="Feature similarity method ('auto' picks pearson/cosine by feature shape)")
    p.add_argument('--dtw_window', type=int, default=30,
                   help="Sakoe-Chiba band for DTW (0=full DTW; recommend 10-50 for T>200)")
    p.add_argument('--xcorr_max_lag', type=int, default=50,
                   help="Max lag for cross-correlation search (0=all lags; recommend 20-100 for T>200)")

    # Reproducibility
    p.add_argument('--seed', type=int, default=None, help="Random seed (omit for non-deterministic)")

    # First pass: get --config and --graph to load the right defaults
    known, _ = p.parse_known_args()
    cfg = _load_config(known.config, known.graph)
    p.set_defaults(**cfg)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_batch(x, y, batch_size, device):
    idx = torch.randint(len(x), (batch_size,))
    return x[idx].to(device), y[idx].to(device)


@torch.no_grad()
def estimate_loss(model, x, y, eval_batches, batch_size, device):
    model.eval()
    losses = [
        model(*get_batch(x, y, batch_size, device))[1].item()
        for _ in range(eval_batches)
    ]
    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading graph: {args.graph}")
    G     = read_graph(args.graph)
    feats = read_feats(args.graph)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | features {feats.shape}")

    extra_info = ""
    if args.similarity == 'dtw':
        extra_info = f", dtw_window={args.dtw_window}"
    elif args.similarity == 'xcorr':
        extra_info = f", xcorr_max_lag={args.xcorr_max_lag}"
    print(f"Precomputing feature similarity (method={args.similarity}{extra_info})...")
    if args.similarity == 'dtw' and args.dtw_window == 0:
        T = feats.shape[0] if feats.ndim == 3 else feats.shape[1]
        if T > 200:
            print(f"  Warning: full DTW on T={T} is slow. Consider --dtw_window 20-50.")
    if args.similarity == 'xcorr' and args.xcorr_max_lag == 0:
        T = feats.shape[0] if feats.ndim == 3 else feats.shape[1]
        if T > 200:
            print(f"  Warning: full xcorr on T={T} is O(T^2). Consider --xcorr_max_lag 50-100.")
    sim_dict = precompute_feature_similarity(
        G, feats,
        method=args.similarity,
        dtw_window=args.dtw_window,
        xcorr_max_lag=args.xcorr_max_lag,
    )

    # ── Walks ─────────────────────────────────────────────────────────────────
    print(f"Generating walks  (num_walks={args.num_walks}, walk_length={args.walk_length}, "
          f"p={args.p}, q={args.q}, α={args.alpha})")
    walks     = generate_feature_walks(G, sim_dict, args.num_walks, args.walk_length,
                                       p=args.p, q=args.q, alpha=args.alpha)
    all_walks = [w for node_walks in walks.values() for w in node_walks]
    print(f"  {len(all_walks)} total walks")

    # ── Dataset ───────────────────────────────────────────────────────────────
    block_size = args.walk_length - 1
    xs = torch.tensor([w[:-1] for w in all_walks], dtype=torch.long)
    ys = torch.tensor([w[1:]  for w in all_walks], dtype=torch.long)

    n     = len(xs)
    split = int(n * args.train_split)
    perm  = torch.randperm(n)
    x_train, y_train = xs[perm[:split]], ys[perm[:split]]
    x_val,   y_val   = xs[perm[split:]], ys[perm[split:]]
    print(f"Dataset: {len(x_train)} train / {len(x_val)} val  |  block_size={block_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransformerGG(
        vocab_size = G.number_of_nodes(),
        n_embd     = args.embed_dim,
        n_head     = args.num_heads,
        block_size = block_size,
        n_layer    = args.num_layers,
        dropout    = args.dropout,
        device     = device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  |  device: {device}")

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iters, eta_min=args.min_lr
    )

    best_val, no_improve = float('inf'), 0

    for it in range(1, args.max_iters + 1):
        xb, yb = get_batch(x_train, y_train, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if it % args.log_every == 0:
            t_loss = estimate_loss(model, x_train, y_train, args.eval_batches, args.batch_size, device)
            v_loss = estimate_loss(model, x_val,   y_val,   args.eval_batches, args.batch_size, device)
            print(f"iter {it:5d}  train {t_loss:.4f}  val {v_loss:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

            if v_loss < best_val:
                best_val, no_improve = v_loss, 0
            else:
                no_improve += args.log_every
                if no_improve >= args.patience:
                    print(f"Early stop at iter {it}")
                    break

    print(f"\nBest val loss: {best_val:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved  → {args.save_model}")

    if args.save_emb:
        emb = model.token_embedding_table.weight.detach().cpu().numpy()
        np.save(args.save_emb, emb)
        print(f"Embeddings saved  → {args.save_emb}  {emb.shape}")

    # ── Generation + MMD evaluation ───────────────────────────────────────────
    if args.eval_graphs > 0:
        from evaluation import mmd_evaluation

        print(f"\nGenerating {args.eval_graphs} graph(s) for MMD evaluation  "
              f"(sequences={args.num_sequences}, length={args.sequence_length}, "
              f"temperature={args.temperature})")

        model.eval()
        generated_graphs = []

        for g in range(args.eval_graphs):
            G_sample = nx.Graph()
            G_sample.add_nodes_from(G.nodes())

            with torch.no_grad():
                for _ in range(args.num_sequences):
                    seed_node = random.randint(0, G.number_of_nodes() - 1)
                    seed = torch.tensor([[seed_node]], dtype=torch.long, device=device)
                    out  = model.generate(seed,
                                          max_new_tokens=args.sequence_length - 1,
                                          temperature=args.temperature)
                    seq = out[0].cpu().tolist()
                    for u, v in zip(seq[:-1], seq[1:]):
                        if u != v:
                            G_sample.add_edge(u, v)

            G_sample.remove_nodes_from(list(nx.isolates(G_sample)))
            generated_graphs.append(G_sample)
            print(f"  graph {g+1:2d}  nodes: {G_sample.number_of_nodes()}  edges: {G_sample.number_of_edges()}")

        print(f"\nMMD scores  ({args.eval_graphs} generated graphs vs original)")
        print("─" * 45)
        mmd_evaluation(G, generated_graphs)


if __name__ == '__main__':
    main()
