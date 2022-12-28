import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import RandomNodeSampler, DataLoader
from torch.utils.data import DataLoader as Loader

from torch import Tensor
from torch_sparse import SparseTensor
from .utils import normalize_adjacency, precompute_features

def get_loaders(name: str,
                dataset: Dataset,
                config: dict) -> list[Loader]:
    """
    """ #TODO: Add a docstring
    if name == "RandomNodeSampler":
        print("DEPRICATED: check that ***node*** sampling is desired! Change to DataLoader")
        temp = dataset[0]
        if isinstance(temp, Data):
            data: Data = temp
        else:
            raise ValueError(f'Expected item at index zero to be type {Data} received type {type(temp)}')

        if "val" in config.keys():
            return [RandomNodeSampler(data, shuffle=True, **config["train"]), RandomNodeSampler(data, shuffle=False, num_workers=16, **config["val"]), RandomNodeSampler(data, shuffle=True, num_workers=16, **config["test"])]
        else:
            return [RandomNodeSampler(data, shuffle=True, num_workers=16, **config["train"]), RandomNodeSampler(data, shuffle=True, num_workers=16, **config["test"])]

    elif name == "SGC":
        temp = dataset[0]
        if isinstance(temp, Data):
            data: Data = temp
        else:
            raise ValueError(f'Expected dataset at index zero to be type {Data} received type {type(temp)}')

        adjacency: Tensor  = SparseTensor(row=data.edge_index[0], col=data.edge_index[1]).to_dense()
        norm_adj: Tensor = normalize_adjacency(adjacency)
        features: Tensor
        precompute_time: float
        features, precompute_time = precompute_features(data.x, norm_adj, config["degree"])
        print(f'PRECOMPUTE TIME: {precompute_time}')

        data.__setitem__("x", features)
        assert torch.all(features.eq(data.x))

        if "val" in config.keys():
            return [RandomNodeSampler(data, shuffle=True, num_workers=16, **config["train"]), RandomNodeSampler(data, shuffle=False, num_workers=16, **config["val"]), RandomNodeSampler(data, shuffle=True, num_workers=16, **config["test"])]
        else:
            return [RandomNodeSampler(data, shuffle=True, num_workers=16, **config["train"]), RandomNodeSampler(data, shuffle=True, num_workers=16, **config["test"])]

    else:
        raise ValueError(f"Unsupported data loader {name}")

