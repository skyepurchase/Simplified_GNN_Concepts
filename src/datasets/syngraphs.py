import torch
import random
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import numpy as np

# typing
from numpy.typing import NDArray
from typing import Callable, Optional
from torch.functional import Tensor


def house_shape():
    return (torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                          [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]]),
            torch.tensor([1, 1, 2, 2, 3]))


def house_size():
    return 5


# Dictionary of shapes to attach to basis graph
SHAPES: dict[str, tuple[Tensor, Tensor]] = defaultdict(house_shape)
SHAPES['grid'] = (torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8],
                                [1, 3, 0, 4, 2, 1, 5, 0, 4, 6, 1, 3, 5, 7, 2, 4, 8, 3, 7, 4, 6, 8, 5, 7]]),
                  torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]))
SHAPES['cycle'] = (torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                 [1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 0, 4]]),
                   torch.tensor([1, 1, 1, 1, 1, 1]))

SIZES: dict[str, int] = defaultdict(house_size)
SIZES['grid'] = 9
SIZES['cycle'] = 6


def tree(height: int = 8) -> Tensor:
    """Builds a balanced binary-tree of height h
    INPUT:
    
    height      :    int height of the tree 
    
    OUTPUT:
    
    graph       :    a tree shape graph
    """
    graph = nx.balanced_tree(2, height)
    edge_index = from_networkx(graph).edge_index

    return edge_index


class SynGraph(InMemoryDataset):
    """Builds a Pytorch Geometric dataset based on those outlined in the GNNExplainer and GCExplainer paper
    INPUT:

    root
    basis
    join
    graph_size
    shape
    num_shapes
    transform
    pre_transform
    pre_filter
    """ #TODO: add docstring
    def __init__(self,
                 root: str,
                 basis: str = "Barabasi-Albert",
                 join: bool = False,
                 num_join_edges: int = 350,
                 graph_size: int = 300,
                 shape: str = "house",
                 num_shapes: int = 80,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.basis = basis
        self.shape = shape
        self.num_shapes = num_shapes
        self.graph_size = graph_size
        self.num_join_edges = num_join_edges
        
        if basis == "Barabasi-Albert":
            self.num_basis_nodes = graph_size
        elif basis == "Tree":
            self.num_basis_nodes = 2 ** (graph_size + 1) - 1
        else:
            raise ValueError(f'Implementation for {basis} does not exist, try Barabasi-Albert or Tree instead.')

        # Generate graph
        edge_index, node_label = self._gen_graph()
        if join:
            edge_index, node_label = self._join(edge_index,
                                                node_label)

        x = torch.ones((len(node_label), 10), dtype=torch.float) # No feature data added

        # Generating random split
        # TODO: Test that these masks work correctly
        train_mask: NDArray = np.ones(len(node_label), dtype=int)
        train_mask[:int(0.2 * len(node_label))] = 0
        np.random.shuffle(train_mask)
        test_mask: NDArray = 1 - train_mask

        data = Data(x=x,
                    y=node_label,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    edge_index=edge_index)

        self.data, self.slices = self.collate([data])

    def _gen_graph(self) -> tuple[Tensor, Tensor]:
        # Generate the base graph
        edge_index, node_label = self._generate_basis(self.basis, self.graph_size)
        
        # Select nodes to connect shapes to
        connecting_nodes: Tensor = torch.randperm(self.num_basis_nodes)[:self.num_shapes]

        # Connecting shapes to basis graph
        edge_index, node_label = self._attach_shapes(edge_index, node_label, self.num_basis_nodes, connecting_nodes)

        return edge_index, node_label

    def _generate_basis(self,
                        basis: str,
                        graph_size: int) -> tuple[Tensor, Tensor]:
        """
        """ #TODO: Add docstring
        if basis == "Barabasi-Albert":
            edge_index = barabasi_albert_graph(graph_size, num_edges=5)
        elif basis == "Tree":
            edge_index = tree(graph_size)
        else:
            raise ValueError(f'Implementation for {basis} does not exist, try Barabasi-Albert or Tree instead.')
        
        if isinstance(edge_index, Tensor):
            node_label = torch.zeros(len(edge_index.unique()), dtype=torch.int64)
        else:
            raise ValueError(f'Expected edge_index to be type {Tensor} got {type(edge_index)} instead.')

        return edge_index, node_label


    def _attach_shapes(self,
                       edge_index: Tensor,
                       node_label: Tensor,
                       base_shape_node_id: int,
                       connecting_nodes: Tensor) -> tuple[Tensor, Tensor]:
        """
        """ #TODO: Add docstring
        edge_indices = [edge_index]
        node_labels = [node_label]

        shape_edge_index, shape_label = SHAPES[self.shape]
        shape_size = SIZES[self.shape]
        for i in range(self.num_shapes):
            edge_indices.append(shape_edge_index + base_shape_node_id) # To prevent overlap
            edge_indices.append(
                torch.tensor([[int(connecting_nodes[i]), base_shape_node_id],
                              [base_shape_node_id, int(connecting_nodes[i])]])
            ) # Adding connection between base shape node and connecting node

            # Account for added nodes
            node_labels.append(shape_label)
            base_shape_node_id += shape_size 
        
        # Flatten indices
        edge_index = torch.cat(edge_indices, dim=1)
        node_label = torch.cat(node_labels, dim=0)

        return edge_index, node_label

    def _join(self,
              edge_index: Tensor,
              node_label: Tensor) -> tuple[Tensor, Tensor]:

        num_A_labels = torch.max(node_label)

        edge_index_B, node_label_B = self._gen_graph()

        num_A_nodes = len(node_label)
        num_B_nodes = len(node_label_B)

        edge_count = 0
        edge_indices = [edge_index, edge_index_B + num_A_nodes]
        while edge_count < self.num_join_edges:
            node_1 = random.randint(0, num_A_nodes - 1)
            node_2 = random.randint(num_A_nodes, num_B_nodes + num_A_nodes - 1)
            connections = torch.tensor([[node_1, node_2],
                                        [node_2, node_1]])
            edge_indices.append(connections)
            edge_count += 1

        return torch.cat(edge_indices, dim=1), torch.cat([node_label, node_label_B + num_A_labels + 1], dim=0)


if __name__=='__main__':
    dataset = SynGraph("data/BAGrid",
                       shape="grid")
    data = dataset[0]

    print(len(dataset), dataset.num_classes, dataset.num_node_features)
    
    if isinstance(data, Data):
        print(data.num_nodes, data.num_edges)
