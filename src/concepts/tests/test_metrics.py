import unittest
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data
from .. import metrics

# Typing
from numpy.typing import NDArray
from networkx import Graph


class TestConceptMetrics(unittest.TestCase):
    def test_purity_more_than_max_raises_error(self):
        """Test whether passing the top graph with more than max nodes raises an error"""
        # Arrange
        max_nodes = 3
        G1: Graph = nx.cycle_graph(6)
        G2: Graph = nx.cycle_graph(2)

        # Assert
        with self.assertRaises(ValueError):
            metrics.purity([G1, G2], max_nodes=max_nodes)

    def test_purity_one_graph_raise_error(self):
        """Test whether providing no comparison graphs raises an error"""
        # Arrange
        G1: Graph = nx.cycle_graph(6)

        # Assert
        with self.assertRaises(ValueError):
            metrics.purity([G1])

    def test_purity_same_graph_is_0(self):
        """Test whether passing copies of the same graph results in purity score 0"""
        # Arrange
        G1: Graph = nx.cycle_graph(6)
        G2: Graph = nx.cycle_graph(6)

        # Act
        purity: float = metrics.purity([G1, G2])

        # Assert
        self.assertAlmostEqual(purity, 0.0)

    def test_purity_isomorphic_graphs_is_0(self):
        """Test whether passing isomorphic graphs results in purity score 0"""
        # Arrange 
        G1: Graph = nx.cycle_graph(6)
        G2: Graph = nx.Graph()
        G2.add_edges_from([(0,1), (0,5), (1,2), (2,3), (3,4), (4,5)])

        # Act
        purity: float = metrics.purity([G1, G2])

        # Assert
        # Make sure the graphs are not identical
        self.assertNotEqual(list(G1.nodes()), list(G2.nodes()))
        # But isomorphic so still 0.0
        self.assertAlmostEqual(purity, 0.0)

    def test_purity_adding_edge_is_1(self):
        """Test whether adding one edge to a graph results in purity score of 1"""
        # Arrange
        G1: Graph = nx.cycle_graph(6)
        G2: Graph = nx.cycle_graph(6)
        # Add a new edge between unconnected nodes
        G2.add_edge(1,5)

        # Act
        purity: float = metrics.purity([G1, G2])

        # Assert
        self.assertAlmostEqual(purity, 1.0)

    def test_purity_cycle_and_wheel_is_7(self):
        """Test whether a cycle graph of 6 nodes and a wheel graph of 7 has an edit distance of 7
        Wheel graph has a central node and spokes to a cycle
        Therefore both will have a cycle of 6 but with a difference of a central node and six edges"""
        # Arrange
        G1: Graph = nx.cycle_graph(6)
        G2: Graph = nx.wheel_graph(7)

        # Act
        purity: float = metrics.purity([G1, G2])

        # Assert
        self.assertAlmostEqual(purity, 7.0)

    def test_purity_is_average_of_GED(self):
        """Test whether multiple graphs result in the average GED"""
        # Arrange
        G1: Graph = nx.cycle_graph(6)
        G2: Graph = nx.cycle_graph(6)
        G3: Graph = nx.wheel_graph(7)

        # Act
        purity: float = metrics.purity([G1, G2, G3])

        # Assert
        self.assertAlmostEqual(purity, 3.5)

    def test_completeness_labels_as_input_is_1(self):
        """Test whether a 1-to-1 correspondence of labels to nodes yields completeness 1"""
        # Arrange
        clusters = 5
        nodes = 100
        activation: NDArray = np.random.randint(0,clusters,nodes)
        kmeans: KMeans = KMeans(n_clusters=clusters)
        kmeans.fit(activation.reshape(-1, 1))
        data: Data = Data(x=torch.tensor(activation.reshape(-1, 1)),
                          y=torch.tensor(activation),
                          train_mask=torch.tensor(np.where(np.arange(nodes) < 80, 1, 0), dtype=torch.bool),
                          test_mask=torch.tensor(np.where(np.arange(nodes) >= 80, 1, 0), dtype=torch.bool))

        # Act
        completeness: float = metrics.completeness(kmeans, torch.tensor(activation.reshape(-1, 1)), data)

        # Assert
        self.assertAlmostEqual(completeness, 1.0)

    def test_completeness_random_labels_to_input_is_not_1(self):
        """Test whether a random correspondence of labels to nodes yeilds completeness 0.5"""
        # Arrange
        clusters = 5
        nodes = 100
        activation: NDArray = np.random.randint(0,clusters,nodes).reshape(-1, 1)
        labels: NDArray = np.random.randint(0,clusters,nodes)
        kmeans: KMeans = KMeans(n_clusters=clusters)
        kmeans.fit(activation)
        data: Data = Data(x=torch.tensor(activation),
                          y=torch.tensor(labels),
                          train_mask=torch.tensor(np.where(np.arange(nodes) < 80, 1, 0), dtype=torch.bool),
                          test_mask=torch.tensor(np.where(np.arange(nodes) >= 80, 1, 0), dtype=torch.bool))

        # Act
        completeness: float = metrics.completeness(kmeans, torch.tensor(activation), data)

        # Assert
        self.assertNotAlmostEqual(completeness, 1.0)


if __name__=='__main__':
    unittest.main()
