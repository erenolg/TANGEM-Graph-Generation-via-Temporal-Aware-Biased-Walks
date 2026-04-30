import os
import numpy as np
import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))

_GRAPH_PATHS = {
    'ibb1':     'data/ibb1.txt',
    'ibb2':     'data/ibb2.txt',
    'ibb_big':  'data/ibb_big.txt',
    'pems04':   'data/pems04.txt',
    'citeseer': 'data/citeseer.edgelist',
}

_FEAT_PATHS = {
    'ibb1':     'data/ibb1_feats.npy',
    'ibb2':     'data/ibb2_feats.npy',
    'ibb_big':  'data/ibb_big_feats.npy',
    'pems04':   'data/pems04_feats.npy',
    'citeseer': 'data/citeseer_feats.npy',
}


def read_graph(graph_name: str) -> nx.Graph:
    if graph_name not in _GRAPH_PATHS:
        raise ValueError(f"Unknown graph '{graph_name}'. Available: {list(_GRAPH_PATHS)}")
    return nx.read_edgelist(os.path.join(_HERE, _GRAPH_PATHS[graph_name]), nodetype=int)


def read_feats(graph_name: str) -> np.ndarray:
    if graph_name not in _FEAT_PATHS:
        raise ValueError(f"Unknown graph '{graph_name}'. Available: {list(_FEAT_PATHS)}")
    return np.load(os.path.join(_HERE, _FEAT_PATHS[graph_name]))
