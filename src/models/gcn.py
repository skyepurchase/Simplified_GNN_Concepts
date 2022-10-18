import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv

# typing
from torch import Tensor
from torch_geometric.typing import Adj


class GCN(nn.Module):
    def __init__(self, 
                 in_channels : int,
                 out_channels : int,
                 num_layers : int = 1,
                 hid_features : int = 16,
                 dropout : float = 0.2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList() 
        in_features = in_channels
        for _ in range(num_layers):
            conv = GCNConv(in_features,
                           hid_features)
            self.layers.append(conv)
            in_features = hid_features

        output_layer = GCNConv(in_features,
                               out_channels)
        self.layers.append(output_layer)

        print(self.layers)

    def forward(self, x : Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}), '
                f'{self.out_channels}, num_layers={self.num_layers}')

