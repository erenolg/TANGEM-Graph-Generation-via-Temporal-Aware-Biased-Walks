import argparse
import json
import os
import random
import numpy as np
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
    flat = {**cfg['walk'], **cfg['train'], **cfg['model']}
    flat.update(cfg.get('datasets', {}).get(graph, {}))
    return flat


def parse_args():
    p = argparse.ArgumentParser(description="Train feature-biased TransformerGG on graph walks.")

    p.add_argument('--config', type=str, default=_DEFAULT_CONFIG,
                   help="Path to config JSON (default: config.json next to train.py)")
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

    # Output
    p.add_argument('--save_model', type=str, default=None, help="Path to save trained model (.pt)")
    p.add_argument('--save_emb',   type=str, default=None, help="Path to save node embeddings (.npy)")

    # First pass: get --config and --graph so we can load the right defaults
    known, _ = p.parse_known_args()
    cfg = _load_config(known.config, known.graph)
    p.set_defaults(**cfg)

    return p.parse_args()


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


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading graph: {args.graph}")
    G     = read_graph(args.graph)
    feats = read_feats(args.graph)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | features {feats.shape}")

    print(f"Precomputing feature similarity...")
    sim_dict = precompute_feature_similarity(G, feats)

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


if __name__ == '__main__':
    main()
