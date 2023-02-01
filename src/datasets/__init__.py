from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import Constant
from .syngraphs import SynGraph 
from torch_geometric.datasets import Planetoid, Reddit, TUDataset


def get_dataset(name: str, root: str) -> InMemoryDataset:
    """
    """ #TODO: Add a Docstring
    if name in ["Cora", "PubMed", "CiteSeer"]:
        return Planetoid(root, name)
    elif name == "REDDIT-BINARY":
        return TUDataset(root, name, transform=Constant())
    elif name == "Mutagenicity":
        return TUDataset(root, name)
    elif name == "Reddit":
        return Reddit(root)
    elif name == "BA-Shapes":
        return SynGraph(root)
    elif name == "BA-Community":
        return SynGraph(root,
                        join=True)
    elif name == "BA-Grid":
        return SynGraph(root,
                        shape="grid")
    elif name == "Tree-Cycles":
        return SynGraph(root,
                        basis="Tree",
                        graph_size=8,
                        shape="cycle")
    elif name == "Tree-Grid":
        return SynGraph(root,
                        basis="Tree",
                        graph_size=8,
                        shape="grid")
    else:
        raise ValueError(f'Unsupported dataset {name}')

