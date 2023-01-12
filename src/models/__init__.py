from .activation_classifier import Activation_Classifier
from .gcn import GCN
from .sgc import SGC
from .layers import Pool

import pickle

# Typing
from typing import Callable
from torch import nn
from torch import Tensor
from torch_geometric.nn import GCNConv


ACTIVATION_DICT = {}
def _get_activation(idx: str) -> Callable:
    def hook(model: nn.Module, in_feats: tuple, out_feats: Tensor):
        ACTIVATION_DICT[idx] = out_feats.detach()
    return hook


def register_hooks(model: nn.Module) -> nn.Module:
    for name, m in model.named_modules():
        if isinstance(m, (GCNConv, nn.Linear, Pool)):
            m.register_forward_hook(_get_activation(f'{name}'))

    return model


def save_activation(path: str) -> None:
    with open(path, 'wb') as file:
        pickle.dump(ACTIVATION_DICT, file)


def get_model(name: str,
              num_features: int,
              num_classes: int,
              config: dict) -> nn.Module:
    """
    """ #TODO: Add Docstring
    if name == "gcn":
        return GCN(num_features, num_classes, **config)
    elif name == "sgc":
        return SGC(num_features, num_classes, **config)
    else:
        raise ValueError(f'Unsupported model {name}')

