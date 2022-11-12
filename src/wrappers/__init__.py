from .pl_wrapper import LinearWrapper, GraphWrapper 
import torch.nn as nn


def get_wrapper(name: str,
                model: nn.Module,
                learning_rate: int):
    if name == "Graph":
        return GraphWrapper(model, learning_rate)
    if name == "Linear":
        return LinearWrapper(model, learning_rate)
    else:
        raise ValueError(f'Unsupported model wrapper {name}')

