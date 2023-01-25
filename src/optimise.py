import os.path as osp
from os import mkdir

from hyperopt import hp, fmin, tpe
import hyperopt

from torch import Tensor, nn, optim, save

from datasets import get_dataset
from loaders import get_loaders
from models import get_model

#Typing
from argparse import Namespace
from torch_geometric.data import InMemoryDataset
from torch.utils.data import DataLoader
from torch.types import Number


DIR = osp.dirname(__file__)


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
             save_name: str,
             args: Namespace) -> None:

    save_path: str = osp.join(DIR, "../output", save_name)
    if not osp.exists(save_path):
        mkdir(save_path)
    
    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           osp.join(args.root,
                                                    dataset_name))

    model: nn.Module = get_model(config["model"]["name"],
                                 dataset.num_features,
                                 dataset.num_classes,
                                 config["model"]["kwargs"])

    train_loader: DataLoader = get_loaders(config["sampler"]["name"],
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
    _, _, wd, lr = hyperopt.space_eval(space, best)

    with open(osp.join(save_path, "hyperpot.txt"), "w") as file:
        print(f"Best weight decay: {wd}\nBest learning rate: {lr}")
        file.write(f"Over {args.epochs} epochs\nSearched for weight decay in [{args.min_decay, args.max_decay}] and found: {wd}\nSearched for learning rate in [{args.min_lr, args.max_lr}] and found: {lr}")


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
        save_name = filename.split('.')[0] + "-" + dataset_name
        optimise(config, dataset_name, save_name, args)

