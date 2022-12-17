import unittest
from numpy.typing import NDArray
import torch
import numpy as np
from torch import tensor
from torch.functional import Tensor
from torch_geometric.data import InMemoryDataset
from .. import syngraphs


class TestSyntheticGraphs(unittest.TestCase):
    def test_tree_height_one_has_three_nodes(self):
        """Test whether a balanced binary tree of height 1 has 3 nodes"""
        # Arrange

        # Act
        tree: Tensor = syngraphs.tree(1)
        num_nodes: int = len(tree.unique())
        
        # Assert
        self.assertEqual(num_nodes, 3)

    def test_tree_height_2_is_correct(self):
        """Test whether a balanced binary tree of height 2 has the expected tensor"""
        # Arrange
        expected: Tensor = tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6],
                                   [1, 2, 0, 3, 4, 0, 5, 6, 1, 1, 2, 2]])

        # Act
        tree: Tensor = syngraphs.tree(2)

        #Assert
        self.assertTrue(torch.all(expected.eq(tree)))

    def test_tree_default_is_height_8(self):
        """Test whether the default binary tree is a height 8"""
        # Arrange
        expected: Tensor = syngraphs.tree(8)

        # Act 
        tree: Tensor = syngraphs.tree()

        # Assert
        self.assertTrue(torch.all(expected.eq(tree)))

    def test_generate_basis_tree_is_a_tree(self):
        """Test that generate basis creates a tree with correct edges, nodes and labels"""
        # Arrange
        expected_edges: Tensor = syngraphs.tree(8)
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=False,
                                                          graph_size=8,
                                                          shape="house",
                                                          num_shapes=0)

        # Act
        edge_index, node_label = syngraph._generate_basis(basis="Tree",
                                                          graph_size=8)
        num_nodes = len(node_label)
        nodes = torch.arange(num_nodes)
        unique_nodes: Tensor = edge_index.unique()

        # Assert
        self.assertTrue(torch.all(expected_edges.eq(edge_index)))
        self.assertEqual(num_nodes, (2 ** 9) - 1)
        self.assertTrue(torch.all(unique_nodes.eq(nodes)))

    def test_attach_shapes_attach_one_house_to_root(self):
        """Test to check that houses are attached correctly"""
        # Arrange
        expected_edges: Tensor = tensor([[0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 0, 3],
                                         [1, 2, 0, 0, 4, 6, 7, 7, 5, 3, 4, 6, 5, 3, 3, 4, 3, 0]])
        expected_label: Tensor = tensor([0, 0, 0, 1, 1, 2, 2, 3])
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=False,
                                                          graph_size=1,
                                                          shape="house",
                                                          num_shapes=1)
        basis_edges, basis_labels = syngraph._generate_basis(basis="Tree",
                                                              graph_size=1)

        # Act
        edge_index, node_label = syngraph._attach_shapes(basis_edges,
                                                         basis_labels,
                                                         base_shape_node_id=3,
                                                         connecting_nodes=tensor([0]))
        num_nodes = len(node_label)
        unique_nodes = len(edge_index.unique())

        # Assert
        self.assertEqual(num_nodes, syngraphs.SIZES["house"] + 3)
        self.assertEqual(num_nodes, unique_nodes)
        self.assertTrue(torch.all(node_label.eq(expected_label)))
        self.assertTrue(torch.all(edge_index.eq(expected_edges)))
 
    def test_attach_shapes_attach_one_grid_to_root(self):
        """Test to check that grids are attached correctly"""
        # Arrange
        expected_edges: Tensor = tensor([[0, 0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 0, 3],
                                        [1, 2, 0, 0, 4, 6, 3, 7, 5, 4, 8, 3, 7, 9, 4, 6, 8, 10, 5, 7, 11, 6, 10, 7, 9, 11, 8, 10, 3, 0]])
        expected_label: Tensor = tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=False,
                                                          graph_size=1,
                                                          shape="grid",
                                                          num_shapes=1)
        basis_edges, basis_labels = syngraph._generate_basis(basis="Tree",
                                                              graph_size=1)

        # Act
        edge_index, node_label = syngraph._attach_shapes(basis_edges,
                                                         basis_labels,
                                                         base_shape_node_id=3,
                                                         connecting_nodes=tensor([0]))
        num_nodes = len(node_label)
        unique_nodes = len(edge_index.unique())

        # Assert
        self.assertEqual(num_nodes, syngraphs.SIZES["grid"] + 3)
        self.assertEqual(num_nodes, unique_nodes)
        self.assertTrue(torch.all(node_label.eq(expected_label)))
        self.assertTrue(torch.all(edge_index.eq(expected_edges)))       

    def test_attach_shapes_attach_one_cycle_to_root(self):
        """Test to check that cycles are attached correctly"""
        # Arrange
        expected_edges: Tensor = tensor([[0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 3],
                                         [1, 2, 0, 0, 4, 8, 3, 5, 4, 6, 5, 7, 6, 8, 3, 7, 3, 0]])
        expected_label: Tensor = tensor([0, 0, 0, 1, 1, 1, 1, 1, 1])
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=False,
                                                          graph_size=1,
                                                          shape="cycle",
                                                          num_shapes=1)
        basis_edges, basis_labels = syngraph._generate_basis(basis="Tree",
                                                              graph_size=1)

        # Act
        edge_index, node_label = syngraph._attach_shapes(basis_edges,
                                                         basis_labels,
                                                         base_shape_node_id=3,
                                                         connecting_nodes=tensor([0]))
        num_nodes = len(node_label)
        unique_nodes = len(edge_index.unique())

        # Assert
        self.assertEqual(num_nodes, syngraphs.SIZES["cycle"] + 3)
        self.assertEqual(num_nodes, unique_nodes)
        self.assertTrue(torch.all(node_label.eq(expected_label)))
        self.assertTrue(torch.all(edge_index.eq(expected_edges)))       

    def test_attach_shapes_80_houses_to_height_8_tree(self):
        """Test whether the correct number of nodes are attached when attaching 80 houses to a tree"""
        # Arrange
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=False,
                                                          graph_size=8,
                                                          shape="house",
                                                          num_shapes=80)
        basis_edges, basis_labels = syngraph._generate_basis(basis="Tree",
                                                              graph_size=8)
        connecting_nodes: Tensor = torch.randperm((2 ** 9) - 1)[:80]

        # Act
        edge_index, node_label = syngraph._attach_shapes(basis_edges,
                                                         basis_labels,
                                                         base_shape_node_id=511,
                                                         connecting_nodes=connecting_nodes)

        num_nodes = len(node_label)
        unique_nodes = len(edge_index.unique())

        # Assert
        self.assertEqual(torch.max(edge_index), 910)
        self.assertEqual(num_nodes, 911)
        self.assertEqual(num_nodes, unique_nodes)

    def test_attach_shapes_80_houses_to_300_node_BA(self):
        """Test whether the correct number of nodes are attached when attaching 80 houses to a Barabasi-Albert graph"""
        # Arrange
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Barabasi-Albert",
                                                          join=False,
                                                          graph_size=300,
                                                          shape="house",
                                                          num_shapes=80)
        basis_edges, basis_labels = syngraph._generate_basis(basis="Barabasi-Albert",
                                                              graph_size=300)
        connecting_nodes: Tensor = torch.randperm(300)[:80]

        # Act
        edge_index, node_label = syngraph._attach_shapes(basis_edges,
                                                         basis_labels,
                                                         base_shape_node_id=300,
                                                         connecting_nodes=connecting_nodes)

        num_nodes = len(node_label)
        unique_nodes = len(edge_index.unique())

        # Assert
        self.assertEqual(torch.max(edge_index), 699)
        self.assertEqual(num_nodes, 700)
        self.assertEqual(num_nodes, unique_nodes)


    def test_gen_graph_tree_correct(self):
        """Test whether the generate graph function correctly joins previous parts together"""
        # Arrange
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=False,
                                                          graph_size=8,
                                                          shape="house",
                                                          num_shapes=80)

        # Act
        edge_index, node_label = syngraph._gen_graph()

        num_nodes = len(node_label)
        unique_nodes = len(edge_index.unique())

        # Assert
        self.assertEqual(torch.max(edge_index), 910)
        self.assertEqual(num_nodes, 911)
        self.assertEqual(num_nodes, unique_nodes)

    def test_gen_graph_BA_correct(self):
        """Test whether the generate graph function correctly joins previous parts together"""
        # Arrange
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Barabasi-Albert",
                                                          join=False,
                                                          graph_size=300,
                                                          shape="house",
                                                          num_shapes=80)
        basis_edges, basis_labels = syngraph._generate_basis(basis="Barabasi-Albert",
                                                              graph_size=300)
        connecting_nodes: Tensor = torch.randperm(300)[:80]

        # Act
        edge_index, node_label = syngraph._attach_shapes(basis_edges,
                                                         basis_labels,
                                                         base_shape_node_id=300,
                                                         connecting_nodes=connecting_nodes)

        num_nodes = len(node_label)
        unique_nodes = len(edge_index.unique())

        # Assert
        self.assertEqual(torch.max(edge_index), 699)
        self.assertEqual(num_nodes, 700)
        self.assertEqual(num_nodes, unique_nodes)

    def test_join_correct_num_nodes(self):
        """Test whether join produces the correct number of nodes"""
        # Arrange
        num_join_edges = 350
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=True,
                                                          num_join_edges=num_join_edges,
                                                          graph_size=8,
                                                          shape="house",
                                                          num_shapes=80)

        # Act
        edge_index, node_label = syngraph._gen_graph()
        new_edge_index, new_node_label = syngraph._join(edge_index,
                                                        node_label)

        # Assert
        self.assertEqual(new_edge_index.shape[1], (2 * edge_index.shape[1]) + (2 * num_join_edges))
        self.assertEqual(len(new_node_label), 2 * len(node_label))

    def test_join_correct_node_labels(self):
        """Test whether join produces the correct node labels to create distinct graphs"""
        # Arrange
        num_join_edges = 350
        syngraph: syngraphs.SynGraph = syngraphs.SynGraph("data",
                                                          basis="Tree",
                                                          join=True,
                                                          num_join_edges=num_join_edges,
                                                          graph_size=8,
                                                          shape="house",
                                                          num_shapes=80)

        # Act
        edge_index, node_label = syngraph._gen_graph()
        new_edge_index, new_node_label = syngraph._join(edge_index,
                                                        node_label)

        unique_nodes = len(new_edge_index.unique())
        unique_labels = len(new_node_label.unique())

        # Assert
        self.assertEqual(unique_labels, 8)
        self.assertEqual(len(new_node_label), unique_nodes)

    def test_masks_cover_set(self):
        """Test whether the masks that the generated dataset provide perfectly covers the number of nodes present"""
        # Arrange
        syngraph: InMemoryDataset = syngraphs.SynGraph("data",
                                                        basis="Barabasi-Albert",
                                                        join=True,
                                                        graph_size=300,
                                                        shape="house",
                                                        num_shapes=80)

        # Act
        train_mask: NDArray = syngraph.data.train_mask
        test_mask: NDArray = syngraph.data.test_mask

        # Assert
        self.assertTrue(np.all(train_mask + test_mask == np.ones(len(train_mask))))


if __name__=='__main__':
    unittest.main()
