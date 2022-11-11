from datetime import datetime
from torch import Tensor

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, InMemoryDataset
from torch_sparse import SparseTensor

from loaders import loaders
from datasets import get_dataset 
from models import models, sgc
from wrappers import wrappers

from pytorch_lightning import loggers, seed_everything
import pytorch_lightning as pl

import os.path as osp
from typing import Union


DIR = osp.dirname(__file__)


def main(experiment: str,
         config,
         args) -> None:
    seed_everything(args.seed)

    dataset: InMemoryDataset = get_dataset(args.dataset,
                                           osp.join(DIR,
                                                    args.root,
                                                    args.dataset))
    temp = dataset[0]
    if isinstance(temp, Data):
        data: Union[Data, Tensor] = temp
    else:
        raise ValueError('Expected dataset at index zero to be type {Data} received type {type(temp)}')

    model = models[config["model"]["name"]](dataset.num_features,
                                            dataset.num_classes,
                                            **config["model"]["kwargs"])

    pl_model = wrappers[config["wrapper"]["name"]](model, config["wrapper"]["learning_rate"])

    loader = loaders[config["sampler"]["name"]]

    train_loader = loader(data,
                          shuffle=True,
                          **config["sampler"]["train"],
                          num_workers=16)
    val_loader = loader(data,
                        shuffle=False,
                        **config["sampler"]["val"],
                        num_workers=16)

    time = datetime.now()
    version = f'{time.strftime("%Y%m%d-%H%M%S")}_{args.seed}'
    tb_logger = loggers.TensorBoardLogger(save_dir=osp.join(DIR,
                                                            config["trainer"]["dir"]),
                                          name=experiment,
                                          version=version)

    trainer = pl.Trainer(
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        logger=tb_logger,
        replace_sampler_ddp=False,
        strategy='ddp',
        max_epochs=config["trainer"]["max_epochs"],
        enable_progress_bar=True)

    trainer.fit(pl_model, train_loader, val_loader)

#     graph = to_networkx(data)
#     nx.draw(graph)
#     if config["model"]["name"] == "SGC":
#         adjacency: Tensor  = SparseTensor(row=data.edge_index[0], col=data.edge_index[1]).to_dense()
#         norm_adj: Tensor = sgc.normalize_adjacency(adjacency)
#         data, precompute_time = sgc.precompute_features(data.x, norm_adj, config["model"]["degree"])
#         print(f'PRECOMPUTE TIME: {precompute_time}')

#     plt.savefig('test.png')


if __name__=="__main__":
    
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Config file")
    parser.add_argument('--dataset', required=True, help="Dataset to run on")
    parser.add_argument('--root', required=True, help="Root directory for dataset")
    parser.add_argument('--seed', required=True, type=int, help="Seed for randomisation")
    args = parser.parse_args()

    with open(osp.abspath(args.config), 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = args.config.split('/')[-1]
        expr_name = args.dataset + "." + '.'.join(filename.split('.')[:-1])
        main(expr_name, config, args)

