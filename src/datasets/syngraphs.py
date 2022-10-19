import torch
from torch_geometric.data import Data, InMemoryDataset

# typing
from typing import Callable, Optional
from torch.functional import Tensor
from torch_geometric.utils import barabasi_albert_graph


def house() -> tuple[Tensor, Tensor]:
    """

    Returns
        edge_index : CCO edge adjacency matrix of a house
        label      : label of the nodes
    """
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                               [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]])
    label = torch.tensor([1, 1, 2, 2, 3])
    
    return edge_index, label


class BAShapes(InMemoryDataset):
    """ 
    """ #TODO: add docstring
    def __init__(self,
                 root: str,
                 num_nodes: int = 300,
                 num_houses: int = 80,
                 connection_distribution: str = "random",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
       
        # Generate the base BA graph
        print("Generating Barabasi-Albert Graph")
        edge_index = barabasi_albert_graph(num_nodes, num_edges=5)
        if isinstance(edge_index, Tensor):
            edge_label = torch.zeros(edge_index.size(1), dtype=torch.int64)
            node_label = torch.zeros(num_nodes, dtype=torch.int64)
        else:
            raise ValueError(f'expected edge_index to be type {Tensor} got {type(edge_index)} instead.')
        
        # Select nodes to connect shapes to
        print("Selecting nodes to connect to")
        if connection_distribution == "random":
            connecting_nodes: Tensor = torch.randperm(num_nodes)[:num_houses]
        else:
            step = num_nodes // num_houses
            connecting_nodes = torch.arange(0, num_nodes, step)

        # Connecting houses to BA graph
        print("Connecting nodes")
        edge_indices = [edge_index]
        edge_labels = [edge_label]
        node_labels = [node_label]
        for i in range(num_houses):
            house_edge_index, house_label = house()
            base_house_node_id = num_nodes # First node in house is the last node so far

            edge_indices.append(house_edge_index + base_house_node_id) # To prevent overlap
            edge_indices.append(
                torch.tensor([[int(connecting_nodes[i]), base_house_node_id],
                              [base_house_node_id, int(connecting_nodes[i])]])
            ) # Adding connection between base house node and connecting node

            edge_labels.append(
                torch.ones(house_edge_index.size(1), dtype=torch.long) # Identifying edges in house strcuture
            )
            edge_labels.append(torch.zeros(2, dtype=torch.long)) # Connections between house and BA are not counted

            # Account for added nodes
            node_labels.append(house_label)
            num_nodes += 5

        print("Generating dataset")
        # Flatten indices
        edge_index = torch.cat(edge_indices, dim=1)
        edge_label = torch.cat(edge_labels, dim=0)
        node_label = torch.cat(node_labels, dim=0)

        x = torch.ones((num_nodes, 10), dtype=torch.float) # No feature data added
        #TODO: Ask about what expl_mask could be

        data = Data(x=x,
                    y=node_label,
                    edge_index=edge_index,
                    edge_label=edge_label)

        self.data, self.slices = self.collate([data])


if __name__=='__main__':
    dataset = BAShapes("data/BAShapes")
    data = dataset[0]

    print(len(dataset), dataset.num_classes, dataset.num_node_features)
    
    if isinstance(data, Data):
        print(data.num_nodes, data.num_edges)
