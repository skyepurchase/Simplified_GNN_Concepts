import torch
import numpy as np
import scipy.sparse as sp
from time import perf_counter

from scipy.sparse import coo_matrix, dia_matrix
from numpy.typing import NDArray
from torch.functional import Tensor


def sparse_coo_to_torch_sparse_tensor(sparse_coo: coo_matrix) -> Tensor:
    """Convert scipy.sparse.coo_matrix to a torch.sparse.Tensor
    INPUT
        sparse_coo      : The sparse scipy.sparse COO matrix
    OUTPUT
        sparse_tensor   : The converted torch.sparse tensor"""

    indices: Tensor = torch.from_numpy(
        np.vstack((sparse_coo.row, sparse_coo.col)).astype(np.int64)
    )
    values: Tensor = torch.from_numpy(sparse_coo.data)
    shape = torch.Size(sparse_coo.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


def normalize_adjacency(adj : Tensor) -> Tensor:
    """Compute the "normalised" adjacency matrix used in feature precomputation.
    INPUT:
        adj     :   The adjacency matrix
    OUTPUT
        norm_adj:   The "normalised" adjacency matrix"""

    self_loop_adj: Tensor = adj + torch.eye(adj.shape[0])
    coo_adj: coo_matrix = coo_matrix(self_loop_adj)
    row_sum: NDArray = np.array(coo_adj.sum(1))
    D_inv_sqrt: NDArray = np.power(row_sum, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    diag_inv_sqrt: dia_matrix = sp.diags(D_inv_sqrt)
    norm_adj: coo_matrix = diag_inv_sqrt.dot(coo_adj).dot(diag_inv_sqrt).tocoo()
    
    return sparse_coo_to_torch_sparse_tensor(norm_adj).float() 


def precompute_features(features: Tensor,
                        adj: Tensor,
                        degree: int) -> tuple[Tensor, float]:
    """Precompute the features based on "normalised" adjacency and degree.
    INPUT
        features    : The input features from the graph
        adj         : "normalised" adjacency matrix from normalise_adjacency
        degree      : The number of message passing steps
    OUTPUT
        out_features: The precomputed features
        time        : The time taken to precompute features"""

    start: float = perf_counter()
    out_features: Tensor = torch.clone(features)
    for _ in range(degree):
        out_features: Tensor = torch.mm(adj, out_features)
    
    return out_features, perf_counter() - start


class SGC(torch.nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int) -> None:
        super(SGC, self).__init__()

        self.lin = torch.nn.Linear(num_features, num_classes)


    def reset_parameters(self):
        self.lin.reset_parameters()


    def forward(self, x):
        return self.lin(x)

