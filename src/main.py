from argparse import Namespace
from datetime import datetime

from torch_geometric.data import InMemoryDataset

from loaders import get_loaders
from datasets import get_dataset 
from models import get_model
from wrappers import get_wrapper 

from pytorch_lightning import loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

import os.path as osp


DIR = osp.dirname(__file__)


# TODO: Add Docstrings
def main(experiment: str,
         dataset_name: str,
         config: dict,
         args: Namespace) -> None:
    seed_everything(args.seed)

    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           args.root)

    model = get_model(config["model"]["name"],
                      dataset.num_features,
                      dataset.num_classes,
                      config["model"]["kwargs"])

    pl_model = get_wrapper(config["wrapper"]["name"],
                           model,
                           config["wrapper"]["kwargs"])


    loaders = get_loaders(config["sampler"]["name"],
                          dataset,
                          config["sampler"])

    time = datetime.now()
    version = f'{time.strftime("%Y%m%d-%H%M%S")}_{args.seed}'
    tb_logger = loggers.TensorBoardLogger(save_dir=osp.join(DIR,
                                                            config["trainer"]["dir"]),
                                          name=experiment,
                                          version=version)

    checkpoint_name = experiment + f'_{str(args.seed)}-{time.strftime("%Y%m%d-%H%M%S")}'
    trainer = pl.Trainer(
#         callbacks=[
#             ModelCheckpoint(
#                 monitor="val_acc",
#                 dirpath=osp.join(DIR, "../checkpoints", checkpoint_name),
#                 filename="{epoch:02d}-{val_acc:.3f}-{val_loss:.2f}",
#                 save_last=True,
#                 mode="max",
#             ),
#         ],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        logger=tb_logger,
#        replace_sampler_ddp=False,
#        strategy='ddp',
        max_epochs=config["trainer"]["max_epochs"],
        enable_progress_bar=True)

    print(f'Running {experiment} with seed value {args.seed}')
    print(f'Saving models to {osp.join(DIR, "../checkpoints", checkpoint_name)}')
    
    if len(loaders) == 3:
        trainer.fit(pl_model, loaders[0], loaders[1])

        best_model = pl_model.load_from_checkpoint(
            osp.join(DIR, "../checkpoints", checkpoint_name, "last.ckpt")
        )
        trainer.test(best_model, dataloaders=loaders[2])
    elif len(loaders) == 2:
        trainer.fit(pl_model, loaders[0])

        best_model = pl_model.load_from_checkpoint(
            osp.join(DIR, "../checkpoints", checkpoint_name, "last.ckpt")
        )
        trainer.test(best_model, dataloaders=loaders[1])
    else:
        raise Exception(f"Not enough data loaders provided. Expected 2 or 3 received {len(loaders)}")

#     graph = to_networkx(data)
#     nx.draw(graph)
#     plt.savefig('test.png')


if __name__=="__main__":
    
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Config file")
    parser.add_argument('--root', help="Root directory for dataset", default='data/')
    parser.add_argument('--seed', type=int, help="Seed for randomisation", default=1337)
    args = parser.parse_args()

    with open(osp.abspath(args.config), 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = args.config.split('/')[-1]
        dataset_name = filename.split('.')[2]
        expr_name = dataset_name + "." + '.'.join(filename.split('.')[:-1])
        main(expr_name, dataset_name, config, args)

