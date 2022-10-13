import torch

from torch_geometric.data import Data


def main():
    edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data.num_nodes, data.num_edges, data.num_node_features, data.has_isolated_nodes())    

if __name__=="__main__":
    main()

