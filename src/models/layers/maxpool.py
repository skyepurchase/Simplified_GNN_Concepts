from torch import nn
from torch_geometric.nn.pool import global_max_pool


class Pool(nn.Module):
    def __init__(self) -> None:
        super(Pool, self).__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)

