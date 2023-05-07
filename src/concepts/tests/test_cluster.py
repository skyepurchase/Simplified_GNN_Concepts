import unittest
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data
from .. import cluster

# Typing
from numpy.typing import NDArray


class TestClustering(unittest.TestCase):
    def test_clusters_perfect_clustering(self):
        """Test whether data set up with 5 clusters is clustered perfectly into 5 clusters"""
        # Arrange
        clusters = 5
        data: NDArray = np.random.randint(0,clusters,100)
        kmeans: KMeans = KMeans(n_clusters=clusters)

        # Act
        kmeans.fit(data.reshape(-1,1))
        predictions = kmeans.predict(data.reshape(-1,1))

        data_count: NDArray = np.unique(data, return_counts=True)[1]
        pred_count: NDArray = np.unique(predictions, return_counts=True)[1]

        # Assert
        self.assertCountEqual(list(data_count), list(pred_count))


if __name__=='__main__':
    unittest.main()
