from .gcn import GCN
from .sgc import SGC, JumpSGC, PoolSGC, SGCPlus
from .layers import Pool

import pickle

# Typing
from typing import Callable, Dict, Tuple
from torch.nn import Module, Linear
from torch import Tensor
from torch_geometric.nn import GCNConv


ACTIVATION_DICT = {}
def _get_activation(idx: str) -> Callable:
    """Create and return a hook that will store activations as the model trains
        INPUT
            idx     : The id of the layer
    """
    def hook(model: Module, in_feats: Tuple, out_feats: Tensor):
        """Add activation space as represented by the features generated by a layer"""
        ACTIVATION_DICT[idx] = out_feats.detach()
    return hook


def register_hooks(model: Module) -> Module:
    """Registering hooks for a specified subset of layers
    INPUT
        model   : Module model for hooks to be registered open
    OUTPUT
        model   : The resulting Module model"""
    for name, m in model.named_modules():
        if isinstance(m, (Linear, GCNConv, Pool)):
            m.register_forward_hook(_get_activation(f'{name}'))

    return model


def get_activation() -> Dict[str, Tensor]:
    """Return the current activation dictionary"""
    return ACTIVATION_DICT


def save_activation(path: str) -> None:
    """Store the current activations for each layer in a specified file
    INPUT
        path    : The path to the file"""
    with open(path, 'wb') as file:
        pickle.dump(ACTIVATION_DICT, file)


def get_model(name: str,
              num_features: int,
              num_classes: int,
              config: Dict) -> Module:
    """
    """ #TODO: Add Docstring
    if name == "gcn":
        return GCN(num_features, num_classes, **config)
    elif name == "sgc":
        return SGC(num_features, num_classes)
    elif name == "psgc":
        return PoolSGC(num_features, num_classes)
    elif name == "jsgc":
        return JumpSGC(num_features, num_classes, **config)
    else:
        raise ValueError(f'Unsupported model {name}')

