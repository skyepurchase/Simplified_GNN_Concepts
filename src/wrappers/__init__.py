from .pl_wrapper import LinearWrapper, GraphWrapper 
import torch.nn as nn


def get_wrapper(name: str,
                model: nn.Module,
                config: dict):
    """
    """ #TODO: Add Docstring
    if name == "Graph":
        return GraphWrapper(model, **config)
    if name == "Linear":
        return LinearWrapper(model, **config)
    else:
        raise ValueError(f'Unsupported model wrapper {name}')

