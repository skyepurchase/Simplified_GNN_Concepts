import torch.nn as nn
from .gcn import GCN
from .sgc import SGC


def get_model(name: str,
              num_features: int,
              num_classes: int,
              config: dict) -> nn.Module:
    if name == "gcn":
        return GCN(num_features, num_classes, **config)
    elif name == "sgc":
        return SGC(num_features, num_classes, **config)
    else:
        raise ValueError(f'Unsupported model {name}')

