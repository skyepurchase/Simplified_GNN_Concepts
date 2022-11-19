from torch_geometric.data import InMemoryDataset
from .syngraphs import SynGraph 
from torch_geometric.datasets import Planetoid, Reddit

datasets = {
    "BA-Shapes": SynGraph,
    "BA-Grid": SynGraph,
    "BA-Community": SynGraph,
    "Tree-Shapes": SynGraph,
    "Tree-Grid": SynGraph,
}

def get_dataset(name: str, root: str) -> InMemoryDataset:
    if name in ["Cora", "PubMed", "CiteSeer"]:
        return Planetoid(root, name)
    elif name == "Reddit":
        return Reddit(root)
    elif name == "BA-Shapes":
        return SynGraph(root)
    else:
        raise ValueError(f'Unsupported dataset {name}')

