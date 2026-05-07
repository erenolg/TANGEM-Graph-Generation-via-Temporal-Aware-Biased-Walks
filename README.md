# Feature-Aware Biased Random Walks on Graphs

Trains a decoder-only Transformer on feature-biased random walks sampled from a graph. The learned token embeddings become node representations. After training, the model can generate new graphs by autoregressively sampling walk sequences and converting them to edges.

## Method

### Walk sampling
Transition weights combine Node2Vec structural bias with node feature similarity:

```
π(u → v | t) = node2vec_weight(t, v)  ·  φ(u, v)

node2vec_weight:  1/p  if v == t  (return)
                  1    if (t,v) ∈ E  (BFS)
                  1/q  otherwise  (DFS/explore)

φ(u, v) = alpha + (1 - alpha) · s_hat(u, v)
```

`s_hat(u,v)` is node similarity normalized to [0, 1]:
- **Temporal graphs** `(T, N, F)`: mean Pearson correlation across F features, shifted `(r+1)/2`
- **Static graphs** `(N, F)`: cosine similarity over feature vectors

### Training
Walks are treated as token sequences. The Transformer is trained with a next-node prediction objective (cross-entropy). The `token_embedding_table` weights are the resulting node embeddings.

### Generation
The trained model autoregressively generates long sequences from random seed nodes. Consecutive node pairs in each sequence become edges. Multiple sequences are merged into a single generated graph.

## File Structure

```
TANGEM_scratch/
├── main.py          — CLI entry point: train + optional generate + MMD eval
├── config.json      — default hyperparameters and per-dataset overrides
├── data.py          — read_graph, read_feats
├── similarity.py    — precompute_feature_similarity (Pearson / cosine)
├── walks.py         — random_walk, biased_random_walk, feature_biased_random_walk, seq2graph
├── model.py         — TransformerGG (decoder-only Transformer)
├── evaluation.py    — MMD metrics (degree, clustering, spectral, orbit, motif)
└── 3biased_walk.ipynb — interactive notebook covering the full pipeline
```

## Datasets

| Name       | Nodes | Edges | Features          | Type     |
|------------|-------|-------|-------------------|----------|
| ibb1       | 294   | 362   | (1416, 294, 4)    | temporal |
| ibb2       | 256   | 441   | (1416, 256, 4)    | temporal |
| ibb_big    | 2451  | 3667  | (744, 2451, 4)    | temporal |
| pems04     | 237   | 280   | (1416, 237, 3)    | temporal |
| citeseer   | 2120  | 3679  | (2120, 3703)      | static   |

## Usage

### Training only
```bash
python main.py --graph ibb1
```

### Training + MMD evaluation
```bash
# generate 5 graphs after training and compute MMD scores
python main.py --graph ibb1 --eval_graphs 5
```


## Key Hyperparameters

| Parameter         | Default | Description |
|-------------------|---------|-------------|
| `alpha`           | 0.5     | Blend between structure and features. 1 = pure Node2Vec, 0 = pure feature similarity |
| `p`               | 1.0     | Node2Vec return parameter. High p discourages backtracking |
| `q`               | 0.1     | Node2Vec in-out parameter. Low q encourages DFS-like exploration |
| `walk_length`     | 60      | Steps per walk (= Transformer context length) |
| `num_walks`       | 60      | Walks sampled per node |
| `embed_dim`       | 64      | Node embedding dimension |
| `eval_graphs`     | 0       | Generated graphs for MMD evaluation (0 = skip) |
| `sequence_length` | 500     | Tokens per generated sequence |
| `num_sequences`   | 10      | Sequences merged into each generated graph |
| `temperature`     | 1.0     | Generation temperature. < 1 = conservative, > 1 = exploratory |

## Notes

- **Orbit and motif MMD** require the ORCA binary. Compile it with:
  ```bash
  g++ -O2 -o orca/orca orca/orca.cpp
  ```
  If missing, those two scores are skipped and reported as `nan`.

