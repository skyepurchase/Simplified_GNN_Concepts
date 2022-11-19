import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeSampler

from torch import Tensor
from torch_sparse import SparseTensor
from .utils import normalize_adjacency, precompute_features


def get_loaders(name: str,
                data: Data,
                config: dict) -> tuple[DataLoader,DataLoader]:
    """
    """ #TODO: Add a docstring
    if name == "RandomNodeSampler":
        return RandomNodeSampler(data, shuffle=True, **config["train"]), RandomNodeSampler(data, shuffle=False, **config["val"])
    elif name == "SGC":
        adjacency: Tensor  = SparseTensor(row=data.edge_index[0], col=data.edge_index[1]).to_dense()
        norm_adj: Tensor = normalize_adjacency(adjacency)
        features: Tensor
        precompute_time: float
        features, precompute_time = precompute_features(data.x, norm_adj, config["degree"])
        print(f'PRECOMPUTE TIME: {precompute_time}')

        data.__setitem__("x", features)
        assert torch.all(features.eq(data.x))
        return RandomNodeSampler(data, shuffle=True, **config["train"]), RandomNodeSampler(data, shuffle=True, **config["val"])
    else:
        raise ValueError(f"Unsupported data loader {name}")

