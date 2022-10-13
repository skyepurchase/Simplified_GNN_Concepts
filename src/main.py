import torch

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt


def main():
    edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    graph = to_networkx(data)
    nx.draw(graph)
    plt.savefig('test.png')


if __name__=="__main__":
    main()

