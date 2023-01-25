from argparse import Namespace
import os.path as osp

from hyperopt import hp, fmin, tpe
import hyperopt

from torch import Tensor, nn, optim
from torch.types import Number
from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset

from datasets import get_dataset
from loaders import get_loaders
from models import get_model

# TODO: Add docstrings
def train_loop(args):
    model: nn.Module
    train_loader: DataLoader
    lr: float
    weight_decay: float
    model, train_loader, lr, weight_decay = args

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    running_loss: Number = 0. 
    iterations: int = 0

    for batch in train_loader:
        optimizer.zero_grad()

        z: Tensor = model(batch.x)[batch.train_mask]
        y: Tensor = batch.y[batch.train_mask]

        loss: Tensor = criterion(z, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        iterations += 1

    return running_loss / iterations  # Average loss 


def optimise(config: dict,
             dataset_name: str,
             args: Namespace) -> None:
    
    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           osp.join(args.root,
                                                    dataset_name))

    model: nn.Module = get_model(config["model"]["name"],
                                 dataset.num_features,
                                 dataset.num_classes,
                                 config["model"]["kwargs"])

    train_loader = get_loaders(config["sampler"]["name"],
                               dataset,
                               config["sampler"])[0]

    space = hp.choice('hyperparameters',
                      [
                          (model,
                           train_loader,
                           hp.uniform('weight_decay', args.min_decay, args.max_decay),
                           hp.uniform('learning_rate', args.min_lr, args.max_lr))
                      ])

    best = fmin(train_loop, space, algo=tpe.suggest, max_evals=args.epochs)
    
    print(hyperopt.space_eval(space, best))


if __name__=='__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Config file")
    parser.add_argument('--root', help="Root directory for dataset", default="/data")
    parser.add_argument('--min_decay', type=float, default=0., help="Minimum weight decay constant")
    parser.add_argument('--max_decay', type=float, default=0., help="Maximum weight decay constant")
    parser.add_argument('--min_lr', type=float, default=0.2, help="Minimum learning rate")
    parser.add_argument('--max_lr', type=float, default=0.2, help="Maximum learning rate")
    parser.add_argument('--epochs', type=int, default=60, help="The number of epochs to search for hyperparameters")
    args = parser.parse_args()

    with open(osp.abspath(args.config), 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = args.config.split('/')[-1]
        dataset_name = filename.split('.')[2]
        optimise(config, dataset_name, args)

