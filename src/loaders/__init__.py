from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeSampler


def get_loaders(name: str,
                data: Data,
                config: dict) -> tuple[DataLoader,DataLoader]:
    if name == "RandomNodeSampler":
        return RandomNodeSampler(data, shuffle=True, **config["train"]), RandomNodeSampler(data, shuffle=False, **config["val"])
    else:
        raise ValueError(f"Unsupported data loader {name}")

