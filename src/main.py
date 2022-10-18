from datetime import datetime

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from loaders import loaders
from models import models
from wrappers import wrappers

from pytorch_lightning import loggers
import pytorch_lightning as pl

import os.path as osp


def main(root: str, name: str, experiment: str) -> None:
    dataset = Planetoid(root=osp.join(root, name), name=name)
    data = dataset[0]

    loader = loaders["RandomNodeSampler"]

    train_loader = loader(data,
                          shuffle=True,
                          num_parts=10,
                          num_workers=16)
    val_loader = loader(data,
                        shuffle=False,
                        num_parts=10,
                        num_workers=16)

    model = models["gcn"](dataset.num_node_features,
                          dataset.num_classes)

    pl_model = wrappers["Wrapper"](model, 0.01)

    time = datetime.now()
    version = f'{time.strftime("%Y%m%d-%H%M%S")}_1337'
    tb_logger = loggers.TensorBoardLogger(save_dir="logs/",
                                          name=experiment,
                                          version=version)

    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        logger=tb_logger,
        replace_sampler_ddp=False,
        strategy='ddp',
        max_epochs=10,
        enable_progress_bar=True)

    trainer.fit(pl_model, train_loader, val_loader)

#     graph = to_networkx(data)
#     nx.draw(graph)
#     plt.savefig('test.png')


if __name__=="__main__":
    main("data", "Cora", "setuptest")

