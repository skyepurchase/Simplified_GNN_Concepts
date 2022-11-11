from .pl_wrapper import PLModel
import torch.nn as nn


def get_wrapper(name: str,
                model: nn.Module,
                learning_rate: int):
    if name == "Wrapper":
        return PLModel(model, learning_rate)
    else:
        raise ValueError(f'Unsupported model wrapper {name}')

