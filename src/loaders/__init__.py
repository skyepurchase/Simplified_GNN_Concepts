import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import RandomNodeLoader, DataLoader
from torch_sparse import SparseTensor

from .utils import normalize_adjacency, precompute_features, save_activation

# Typing
from torch.utils.data import DataLoader as Loader
from torch_geometric.data import Data, Dataset
from torch import Tensor
from typing import Any, Union, Tuple


def save_precomputation(path: str):
    """Wrapper for save_activation
    INPUT
        path    : The path to store the activations"""
    save_activation(path)


def get_data(dataset: Dataset) -> Data:
    """Extracting the first data element of a dataset confirming it is of type Data
    INPUT
        dataset : A dataset containing a single graph
    OUTPUT
        data    : The graph from the dataset"""
    temp = dataset[0]
    if isinstance(temp, Data):
        data: Data = temp
    else:
        raise ValueError(f'Expected dataset at index zero to be type {Data} received type {type(temp)}')

    return data


def get_graphs(dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """Split a dataset of graphs into train and test randomly
    INPUT
        dataset     : A dataset of multiple graphs
    OUTPUT
        train_set   : A dataset of 80% of the graphs
        test_set    : A dataset of the remaining graphs"""
    graphs = dataset.shuffle()

    if isinstance(graphs, Dataset):
        train_idx: int = int(len(graphs) * 0.8)
        train_set: Union[Dataset, Data] = graphs[:train_idx]
        test_set: Union[Dataset, Data] = graphs[train_idx:]
    else:
        raise TypeError(f"Expected graphs to be type {Dataset} instead received type {type(graphs)}")

    if isinstance(train_set, Dataset) and isinstance(test_set, Dataset):
        return train_set, test_set
    else:
        raise TypeError(f"Train and Test set expected to be type {Dataset} received type {type(train_set)} and {type(test_set)}")


def precompute_graph(graph: Data,
                     config: dict[str, Any]) -> Tuple[Tensor, float]:
    """A wrapper for the SGC precompute_features function
    INPUT 
        graph       : The graph which the precomputation is applied to 
        config      : A dictionary detailing how the SGC model is built
    OUTPUT
        features    : The resulting computed features
        time        : The time it took to compute the features"""
    adjacency: Tensor  = SparseTensor(row=graph.edge_index[0], col=graph.edge_index[1]).to_dense()
    norm_adj: Tensor = normalize_adjacency(adjacency)
    return precompute_features(graph.x, norm_adj, config["degree"])


def get_loaders(name: str,
                dataset: Dataset,
                config: dict) -> list[Loader]:
    """
    """ #TODO: Add a docstring
    if name == "RandomNodeSampler":
        raise DeprecationWarning("DEPRICATED: check that ***node*** sampling is desired! Change to DataLoader")

    elif name == "SGC":
        data: Data = get_data(dataset)

        features: Tensor
        precompute_time: float
        features, precompute_time = precompute_graph(data, config)
        print(f'PRECOMPUTE TIME: {precompute_time}')

        data.__setitem__("x", features)
        assert torch.all(features.eq(data.x))

        if "val" in config.keys():
            return [RandomNodeLoader(data, shuffle=True, num_workers=16, **config["train"]),
                    RandomNodeLoader(data, shuffle=False, num_workers=16, **config["val"]),
                    RandomNodeLoader(data, shuffle=False, num_workers=16, **config["test"])]
        else:
            return [RandomNodeLoader(data, shuffle=True, num_workers=16, **config["train"]),
                    RandomNodeLoader(data, shuffle=False, num_workers=16, **config["test"])]

    elif name == "GraphSGC":
        train_set: Dataset
        test_set: Dataset
        train_set, test_set = get_graphs(dataset)

        total_compute_time: float = 0.0
        graph: Union[Dataset, Data]
        for graph in train_set:
            assert isinstance(graph, Data)

            features: Tensor
            precompute_time: float
            features, precompute_time = precompute_graph(graph, config)

            total_compute_time += precompute_time

            graph.__setitem__("x", features)
            assert torch.all(features.eq(graph.x))

        for graph in test_set:
            assert isinstance(graph, Data)

            features: Tensor
            precompute_time: float
            features, precompute_time = precompute_graph(graph, config)

            total_compute_time += precompute_time

            graph.__setitem__("x", features)
            assert torch.all(features.eq(graph.x))
        print(f"TOTAL PRECOMPUTATION TIME: {total_compute_time}")

        if "val" in config.keys():
            return [DataLoader(train_set, shuffle=True, num_workers=16, **config["train"]),
                    DataLoader(train_set, shuffle=True, num_workers=16, **config["val"]),
                    DataLoader(train_set, shuffle=True, num_workers=16, **config["test"])]
        else:
            return [DataLoader(train_set, shuffle=True, num_workers=16, **config["train"]),
                    DataLoader(train_set, shuffle=True, num_workers=16, **config["test"])]

    elif name == "DataLoader":
        if "val" in config.keys():
            raise ValueError("ATTENTION: DataLoader expects GCExplainer model. No validation set supported")
        else:
            train_loader = DataLoader(dataset, num_workers=16)
            test_loader = DataLoader(dataset, num_workers=16)

            return [train_loader, test_loader]

    elif name == "GraphLoader":
        train_set: Dataset
        test_set: Dataset
        train_set, test_set = get_graphs(dataset)

        if "val" in config.keys():
            raise ValueError(f"Validation set not supported")
        else:
            train_loader = DataLoader(train_set, shuffle=True, num_workers=16, **config["train"])
            test_loader = DataLoader(test_set, shuffle=False, num_workers=16, **config["train"])
            # want to extract all graphs in a single batch for activations
            full_loader = DataLoader(dataset, shuffle=False, num_workers=16, batch_size=len(dataset))
        return [train_loader, test_loader, full_loader]

    else:
        raise ValueError(f"Unsupported data loader {name}")

