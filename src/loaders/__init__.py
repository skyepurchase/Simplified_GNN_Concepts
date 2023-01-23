import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import RandomNodeLoader, DataLoader
from torch.utils.data import DataLoader as Loader

from torch import Tensor
from torch_sparse import SparseTensor
from .utils import normalize_adjacency, precompute_features, save_activation


def save_precomputation(path: str):
    """Wrapper for save_activation"""
    save_activation(path)


def get_loaders(name: str,
                dataset: Dataset,
                config: dict) -> list[Loader]:
    """
    """ #TODO: Add a docstring
    if name == "RandomNodeSampler":
        raise DeprecationWarning("DEPRICATED: check that ***node*** sampling is desired! Change to DataLoader")
#         temp = dataset[0]
#         if isinstance(temp, Data):
#             data: Data = temp
#         else:
#             raise ValueError(f'Expected item at index zero to be type {Data} received type {type(temp)}')
# 
#         if "val" in config.keys():
#             return [RandomNodeLoader(data, shuffle=True, **config["train"]),
#                     RandomNodeLoader(data, shuffle=False, num_workers=16, **config["val"]),
#                     RandomNodeLoader(data, shuffle=True, num_workers=16, **config["test"])]
#         else:
#             return [RandomNodeLoader(data, shuffle=True, num_workers=16, **config["train"]),
#                     RandomNodeLoader(data, shuffle=True, num_workers=16, **config["test"])]

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
            return [RandomNodeLoader(data, shuffle=True, num_workers=16, **config["train"]),
                    RandomNodeLoader(data, shuffle=False, num_workers=16, **config["val"]),
                    RandomNodeLoader(data, shuffle=True, num_workers=16, **config["test"])]
        else:
            return [RandomNodeLoader(data, shuffle=True, num_workers=16, **config["train"]),
                    RandomNodeLoader(data, shuffle=True, num_workers=16, **config["test"])]

    elif name == "DataLoader":
        if "val" in config.keys():
            raise ValueError("ATTENTION: DataLoader should be for GCExplainer comparison. No known reason for validation set")
        else:
            train_loader = DataLoader(dataset, num_workers=16)
            test_loader = DataLoader(dataset, num_workers=16)

            return [train_loader, test_loader]

    elif name == "GraphLoader":
        graphs = dataset.shuffle()

        train_idx = int(len(graphs) * 0.8) # TODO: Potentially make this a user defined value
        train_set = graphs[:train_idx]
        test_set = graphs[train_idx:]

        if isinstance(train_set, Dataset) and isinstance(test_set, Dataset):
            if "val" in config.keys():
                raise ValueError(f"Validation set not available for {name}")
            else:
                train_loader = DataLoader(train_set, shuffle=True, num_workers=16, **config["train"])
                test_loader = DataLoader(test_set, shuffle=False, num_workers=16, **config["train"])
            return [train_loader, test_loader]
        else:
            raise TypeError(f"Train and Test set expected to be type {Dataset} received type {type(train_set)} and {type(test_set)}")

    else:
        raise ValueError(f"Unsupported data loader {name}")

