from collections import defaultdict
import torch
from torch_geometric.data import Data, InMemoryDataset

# typing
from typing import Callable, Optional
from torch.functional import Tensor
from torch_geometric.utils import barabasi_albert_graph


def house_shape():
    return (torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                          [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]]),
            torch.tensor([1, 1, 2, 2, 3]))


def house_size():
    return 5


# Dictionary of shapes to attach to basis graph
SHAPES: dict[str, tuple[Tensor, Tensor]] = defaultdict(house_shape)
SIZES: dict[str, int] = defaultdict(house_size)

class SynGraph(InMemoryDataset):
    """ 
    """ #TODO: add docstring
    def __init__(self,
                 root: str,
                 basis: str = "Barabasi-Albert",
                 num_nodes: int = 300,
                 shape: str = "house",
                 num_shapes: int = 80,
                 connection_distribution: str = "random",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.num_basis_nodes = num_nodes
        self.shape = shape
        self.num_shapes = num_shapes

        # Generate the base BA graph
        edge_index, edge_label, node_label = self._generate_basis(basis, num_nodes)
        
        # Select nodes to connect shapes to
        print("Selecting nodes to connect to")
        if connection_distribution == "random":
            connecting_nodes: Tensor = torch.randperm(num_nodes)[:num_shapes]
        else:
            step = num_nodes // num_shapes
            connecting_nodes = torch.arange(0, num_nodes, step)

        # Connecting shapes to basis graph
        edge_index, edge_label, node_label = self._attach_shapes(edge_index, edge_label, node_label, num_nodes, connecting_nodes)

        print("Generating dataset")
        x = torch.ones((num_nodes, 10), dtype=torch.float) # No feature data added
        #TODO: Ask about what expl_mask could be

        data = Data(x=x,
                    y=node_label,
                    edge_index=edge_index,
                    edge_label=edge_label)

        self.data, self.slices = self.collate([data])

    def _generate_basis(self,
                        basis: str,
                        num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        """ #TODO: Add docstring
        print(f"Generating {basis} Graph")
        if basis == "Barabasi-Albert":
            edge_index = barabasi_albert_graph(num_nodes, num_edges=5)
        else:
            raise ValueError(f'Implementation for Barabasi-Albert only, received {basis} instead.')
        
        if isinstance(edge_index, Tensor):
            edge_label = torch.zeros(edge_index.size(1), dtype=torch.int64)
            node_label = torch.zeros(num_nodes, dtype=torch.int64)
        else:
            raise ValueError(f'Expected edge_index to be type {Tensor} got {type(edge_index)} instead.')

        return edge_index, edge_label, node_label


    def _attach_shapes(self,
                       edge_index: Tensor,
                       edge_label: Tensor,
                       node_label: Tensor,
                       base_shape_node_id: int,
                       connecting_nodes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        """ #TODO: Add docstring
        print("Connecting nodes")
        edge_indices = [edge_index]
        edge_labels = [edge_label]
        node_labels = [node_label]

        shape_edge_index, shape_label = SHAPES[self.shape]
        shape_size = SIZES[self.shape]
        for i in range(self.num_shapes):
            edge_indices.append(shape_edge_index + base_shape_node_id) # To prevent overlap
            edge_indices.append(
                torch.tensor([[int(connecting_nodes[i]), base_shape_node_id],
                              [base_shape_node_id, int(connecting_nodes[i])]])
            ) # Adding connection between base shape node and connecting node

            edge_labels.append(
                torch.ones(shape_edge_index.size(1), dtype=torch.long) # Identifying edges in shape strcuture
            )
            edge_labels.append(torch.zeros(2, dtype=torch.long)) # Connections between shape and BA are not counted

            # Account for added nodes
            node_labels.append(shape_label)
            base_shape_node_id += shape_size 
        
        # Flatten indices
        edge_index = torch.cat(edge_indices, dim=1)
        edge_label = torch.cat(edge_labels, dim=0)
        node_label = torch.cat(node_labels, dim=0)

        return edge_index, edge_label, node_label


if __name__=='__main__':
    dataset = SynGraph("data/BAShapes")
    data = dataset[0]

    print(len(dataset), dataset.num_classes, dataset.num_node_features)
    
    if isinstance(data, Data):
        print(data.num_nodes, data.num_edges)
