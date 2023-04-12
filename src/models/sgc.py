import torch
from torch.functional import Tensor
import torch.nn.functional as F

from .layers import Pool


class SGC(torch.nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int) -> None:
        super(SGC, self).__init__()

        self.lin = torch.nn.Linear(num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class SGCPlus(torch.nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int) -> None:
        super().__init__()

        self.lin1 = torch.nn.Linear(num_features, 20)
        self.lin2 = torch.nn.Linear(20, 20)
        self.lin3 = torch.nn.Linear(20, 20)
        self.classifier = torch.nn.Linear(20, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.lin1(x)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.relu(out)
        out = self.lin3(out)
        out = F.relu(out)
        out = self.classifier(out)

        return F.softmax(out)


class PoolSGC(SGC):
    def __init__(self,
                 num_features: int,
                 num_classes: int) -> None:
        super().__init__(num_features, num_classes)
        self.pool = Pool()

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = self.pool(x, batch)
        return super().forward(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'pooling=global max. pooling)')


class JumpSGC(SGC):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 jump_degree: int) -> None:
        super().__init__(num_features, num_classes)
        self.aggr = torch.nn.Linear(num_features * jump_degree, num_features)
        self.aggr.reset_parameters

    def forward(self, x: Tensor) -> Tensor:
        x_aggr = self.aggr(x)
        return super().forward(x_aggr)

