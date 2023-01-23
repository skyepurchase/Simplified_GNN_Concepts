from datetime import datetime
from tqdm import tqdm
import os.path as osp
import pickle

import torch

from torch_geometric.data import InMemoryDataset

from loaders import get_loaders, save_precomputation
from datasets import get_dataset 
from models import get_activation, save_activation, get_model, register_hooks
from wrappers import get_wrapper 

from pytorch_lightning import loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# Typing
from torch_geometric.data import Data
from argparse import Namespace
from torch import Tensor

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
    model = register_hooks(model)

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

    save_filename = experiment + f'_{str(args.seed)}-{time.strftime("%Y%m%d-%H%M%S")}'
    checkpoint = False 
    callbacks = []
#     if experiment.split('.')[0] == "GCN":
#         callbacks.append(
#             EarlyStopping(
#                 monitor="train_monitor",
#                 stopping_threshold=0.05,
#                 patience=100000,
#                 check_on_train_epoch_end=True,
#                 verbose=True,
#             )
#         )
    if experiment.split('.')[0] == "SGC":
        checkpoint = True
        print(f'Saving models to {osp.join(DIR, "../checkpoints", save_filename)}')
        callbacks.append(
            ModelCheckpoint(
                monitor="val_acc",
                dirpath=osp.join(DIR, "../checkpoints", save_filename),
                filename="{epoch:02d}-{val_acc:.3f}-{val_loss:.2f}",
                save_last=True,
                mode="max",
            )
        )

    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        logger=tb_logger,
        max_epochs=config["trainer"]["max_epochs"],
        log_every_n_steps=50,
        enable_progress_bar=args.verbose)

    print(f'Running {experiment} with seed value {args.seed}')
    
    if config["sampler"]["name"] == "SGC":
        assert len(loaders) == 3
        save_precomputation(osp.join(DIR, "../activations", f'{save_filename}.pkl'))
        trainer.fit(pl_model, loaders[0], loaders[1])

        if checkpoint:
            best_model = pl_model.load_from_checkpoint(
                osp.join(DIR, "../checkpoints", save_filename, "last.ckpt"),
                model=model
            )
        else:
            best_model = pl_model

        trainer.test(best_model, dataloaders=loaders[2])
    elif config["sampler"]["name"] in ["DataLoader", "GraphLoader"]:
        assert len(loaders) > 1
        trainer.fit(pl_model, loaders[0])

        if checkpoint:
            best_model = pl_model.load_from_checkpoint(
                osp.join(DIR, "../checkpoints", save_filename, "last.ckpt"),
                model=model
            )
        else:
            best_model = pl_model

        trainer.test(best_model, dataloaders=loaders[1])

        if config["sampler"]["name"] == "GraphLoader":
            print("Evaluating on the entire dataset for concept extraction")
            best_model.eval()

            all_activations: dict[str, Tensor] = {}
            graph: Data
            for graph in tqdm(loaders[2], desc="Extracting activations"):
                _ = best_model(graph) # Evaluate to allow hooks to extract activations

                layer: str
                activation: Tensor
                for layer, activation in get_activation().items():
                    if layer in all_activations:
                        all_activations[layer] = torch.cat((all_activations[layer], activation))
                    else:
                        all_activations[layer] = activation

            with open(osp.join(DIR, "../activations", f'{save_filename}.pkl'), 'wb') as file:
                pickle.dump(all_activations, file)
#
#             trainer.test(best_model, dataloaders=loaders[2], verbose=False)
        else:
            save_activation(osp.join(DIR, "../activations", f'{save_filename}.pkl'))
    else:
        raise Exception(f"{config['sampler']['name']} is not supported")

#     graph = to_networkx(data)
#     nx.draw(graph)
#     plt.savefig('test.png')


if __name__=="__main__":
    
    import yaml
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Config file")
    parser.add_argument('--root', help="Root directory for dataset", default='data/')
    parser.add_argument('--seed', type=int, help="Seed for randomisation", default=1337)
    parser.add_argument('-v','--verbose', action="store_true", help="Whether to display progress bar during training", default=False)
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    with open(osp.abspath(args.config), 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = args.config.split('/')[-1]
        dataset_name = filename.split('.')[2]
        main(filename, dataset_name, config, args)

