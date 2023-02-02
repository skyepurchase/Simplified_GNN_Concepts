from .pl_wrapper import LinearPoolWrapper, LinearWrapper, LinearValWrapper, GraphWrapper, GraphPoolWrapper
import torch.nn as nn


def get_wrapper(name: str,
                model: nn.Module,
                config: dict):
    """
    """ #TODO: Add Docstring
    if name == "Graph":
        return GraphWrapper(model, **config)
    if name == "Linear":
        return LinearValWrapper(model, **config)
    if name == "Pool":
        return GraphPoolWrapper(model, **config)
    if name == "GraphLinear":
        return LinearWrapper(model, **config)
    if name == "PoolLinear":
        return LinearPoolWrapper(model, **config)
    else:
        raise ValueError(f'Unsupported model wrapper {name}')

