import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from .layers import Pool

# typing
from torch import Tensor
from torch_geometric.typing import Adj


class GCN(nn.Module):
    def __init__(self, 
                 in_channels : int,
                 out_channels : int,
                 num_conv_layers : int = 1,
                 num_lin_layers : int = 1,
                 pooling : bool = False,
                 hid_features : list = [16]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_layers = num_conv_layers
        self.num_lin_layers = num_lin_layers
        self.pooling = pooling

        self.pool_layer = Pool()
        self.convs = nn.ModuleList() 

        in_features: int = in_channels
        for i in range(num_conv_layers):
            conv = GCNConv(in_features,
                           hid_features[i])
            conv.reset_parameters()

            self.convs.append(conv)
            in_features = hid_features[i]

        if pooling:
            in_features = in_features

        self.lins = nn.ModuleList()
        for i in range(num_lin_layers - 1):
            lin = Linear(in_features,
                         hid_features[num_conv_layers + i])
            lin.reset_parameters()

            self.lins.append(lin)
            in_features = hid_features[num_conv_layers + i]

        output_layer = Linear(in_features,
                              out_channels)
        self.lins.append(output_layer)

    def forward(self, x : Tensor, edge_index: Adj, batch: Tensor) -> Tensor:
        for layer in self.convs:
            x = layer(x, edge_index)
            x = F.relu(x)

        if self.pooling:
            x = self.pool_layer(x, batch)

        for layer in self.lins:
            x = layer(x)
            x = F.relu(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self) -> str:
        pooling = 'global max. pooling, ' if self.pooling else ''
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'num_layers=[{self.num_conv_layers}, {pooling}{self.num_lin_layers})')

