from torch_geometric.data import InMemoryDataset
from .syngraphs import SynGraph 
from torch_geometric.datasets import Planetoid

datasets = {
    "BA-Shapes": SynGraph,
    "BA-Grid": SynGraph,
    "BA-Community": SynGraph,
    "Tree-Shapes": SynGraph,
    "Tree-Grid": SynGraph,
    "Cora": Planetoid,
}

def get_dataset(name: str, root: str) -> InMemoryDataset:
    if name == "Cora":
        return Planetoid(root, name)
    elif name == "BA-Shapes":
        return SynGraph(root)
    else:
        raise ValueError(f'Unsupported dataset {name}')

