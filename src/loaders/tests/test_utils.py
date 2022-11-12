import math
import unittest
from numpy._typing import NDArray

import torch
import numpy as np
from torch.functional import Tensor

from ..utils import normalize_adjacency, precompute_features


class TestSimplifiedGraphConvolution(unittest.TestCase):
    def test_normalise_adjacency_identity_preserved(self):
        """Test whether an adjacency of only self-loops does not pass information between nodes"""
        # Arrange
        
        # Act
        adj_norm: Tensor = normalize_adjacency(torch.eye(500))

        # Assert
        self.assertTrue(np.allclose(np.eye(500), adj_norm.numpy()))

    def test_normalise_adjacency_correct(self):
        """Test whether a small hand-checked example is correctly computed"""
        # Arrange
        expected: NDArray = np.array([[2/3, 0, 1 / (2 * math.sqrt(3))],
                                       [0, 0.5, 1 / (2 * math.sqrt(2))],
                                       [1 / (2 * math.sqrt(3)), 1 / (2 * math.sqrt(2)), 0.5]])
        
        # Act
        adj_norm: Tensor = normalize_adjacency(torch.tensor([[1, 0, 1],
                                                             [0, 0, 1],
                                                             [1, 1, 1]]))
        adj_np_norm: NDArray = adj_norm.numpy()

        # Assert
        self.assertTrue(np.allclose(expected, adj_np_norm))

    def test_normalise_adjacency_infinity_removed(self):
        """Test whether infinities are removed from adjacencies normalization"""
        # Arrange
        
        # Act
        adj_norm: Tensor = normalize_adjacency(-torch.eye(500))

        # Assert
        self.assertTrue(torch.all(torch.zeros(500).eq(adj_norm)))

    def test_precompute_zero_step_is_constant_features(self):
        """Test whether a single step propagates"""
        # Arrange
        adj: Tensor = normalize_adjacency(torch.tensor([[1, 0, 1],
                                                        [0, 0, 1],
                                                        [1, 1, 1]]))
        feats: Tensor = torch.tensor([[0., 1.],
                                      [2., 3.],
                                      [4., 5.]])

        # Act
        out_feats, _ = precompute_features(feats, adj, 0)

        # Assert
        self.assertTrue(np.allclose(feats.numpy(), out_feats.numpy()))

    def test_precompute_two_step_is_correct(self):
        """Test whether two precomputed steps match hand-checked result"""
        # Arrange
        adj: Tensor = normalize_adjacency(torch.tensor([[1, 0, 1],
                                                        [0, 0, 1],
                                                        [1, 1, 1]]))
        feats: Tensor = torch.tensor([[0., 1.],
                                      [2., 3.],
                                      [4., 5.]])
        expected: Tensor = torch.tensor([[(3 + (14 * math.sqrt(2))) / (6 * math.sqrt(6)),
                                          (27 + (19 * math.sqrt(6)) + (105 * math.sqrt(2))) / (36 * math.sqrt(6))],
                                         [(8 + (3 * math.sqrt(2))) / (4 * math.sqrt(2)),
                                          (2 + (9 * math.sqrt(6)) + (20 * math.sqrt(3))) / (8 * math.sqrt(6))],
                                         [(6 + (11 * math.sqrt(2))) / (6 * math.sqrt(2)),
                                          ((14 * math.sqrt(2)) + (36 * math.sqrt(3)) + (55 * math.sqrt(6))) / (24 * math.sqrt(6))]])
        
        # Act
        out_feats, _ = precompute_features(feats, adj, 2)

        # Assert
        self.assertTrue(np.allclose(expected.numpy(), out_feats.numpy()))


if __name__=='__main__':
    unittest.main()

