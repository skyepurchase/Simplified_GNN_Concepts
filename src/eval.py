import pickle
import os.path as osp
import numpy as np
from os import mkdir
from pytorch_lightning import LightningModule

from loaders import get_loaders
from models import get_activation, get_model, register_hooks
from concepts.cluster import kmeans_cluster
from concepts.metrics import completeness, purity
from concepts.plotting import plot_samples
from datasets import get_dataset

import pytorch_lightning as pl

import torch
from torch import load, tensor

# Typing
from torch_geometric.data import Data, InMemoryDataset
from io import TextIOWrapper
from sklearn.cluster import KMeans
from torch import Tensor
from argparse import Namespace
from networkx import Graph
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Union

from wrappers import get_wrapper


DIR = osp.dirname(__file__)


def main(args: Namespace,
         dataset_name: str,
         save_name: str,
         config: dict) -> None:
    save_path: str = osp.join(DIR, "../output", save_name)

    if not osp.exists(save_path):
        mkdir(save_path)

    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           "data/")

    data: Data
    batch: Union[None, Tensor] = None
    if dataset_name in ["REDDIT-BINARY", "Mutagenicity"]:
        # Convert the batch into a valid data object to completeness scores
        full_loader: DataLoader = get_loaders("GraphLoader",
                                              dataset,
                                              {"test": {}, "train": {}})[2]

        all_graphs: Data = next(iter(full_loader))
        batch = all_graphs.batch

        class_labels_per_node: list[int] = []
        for batch_idx in all_graphs.batch:
            class_labels_per_node.append(all_graphs.y[batch_idx])

        train_mask: Tensor = tensor(np.random.rand(all_graphs.x.shape[0]) < 0.8)
        test_mask: Tensor = ~train_mask
        data = Data(x=all_graphs.x,
                    y=tensor(class_labels_per_node),
                    edge_index=all_graphs.edge_index,
                    batch=all_graphs.batch,
                    train_mask=train_mask,
                    test_mask=test_mask)
    else:
        temp = dataset[0]
        if isinstance(temp, Data):
           data = temp
        else:
            raise ValueError(f'Expected dataset at index zero to be type {Data} received type {type(temp)}')

    activation_list: dict[str, Tensor]
    if args.weights is not None:
        gnn: Module = get_model(config["model"]["name"],
                                dataset.num_features,
                                dataset.num_classes,
                                config["model"]["kwargs"])
        gnn.load_state_dict(load(args.weights))
        gnn = register_hooks(gnn)

        pl_model: LightningModule = get_wrapper(config["wrapper"]["name"],
                                                gnn,
                                                config["wrapper"]["kwargs"])

        if dataset_name in ["REDDIT-BINARY", "Mutagenicity"]:
            full_loader: DataLoader = get_loaders(config["sampler"]["name"],
                                                  dataset,
                                                  config["sampler"])[2]

            trainer = pl.Trainer(
                accelerator=config["trainer"]["accelerator"],
                devices=config["trainer"]["devices"],
                max_epochs=config["trainer"]["max_epochs"],
                enable_progress_bar=False)

            trainer.test(pl_model, dataloaders=full_loader, verbose=False)
            activation_list = get_activation()
        else:
            _ = gnn(data.x, data.edge_index, None)
            activation_list = get_activation()
    else:
        temp_activations: Union[list[Tensor], dict[str, Tensor]]
        with open(osp.join(DIR, "..", args.activation), 'rb') as file:
            temp_activations = pickle.loads(file.read())

        if isinstance(temp_activations, list):
            if batch is None:
                # If this is not graph classification we only expect dictionaries
                raise TypeError(f"Expected activations to be {dict} but received {list}.")
            else:
                # Check that the batch ids match
                # This means that the completeness and purity will correcly match
                assert torch.allclose(batch, temp_activations[1])
                activation_list = {"layers.0": temp_activations[0]}
        else:
            activation_list = temp_activations

    model_list: dict[str, KMeans]
    _, model_list = kmeans_cluster(activation_list, args.clusters)

    layer_graphs: dict[str, dict[int, list[Graph]]] = {}
    comp_file: TextIOWrapper = open(osp.join(save_path, f"{args.clusters}k-{args.hops}n-completeness.txt"), "w")
    pure_file = open(osp.join(save_path, f"{args.clusters}k-{args.hops}n-purity.txt"), "w")
    for layer, model in model_list.items():
        if "pool" in layer:
            continue
        elif "lin" in layer:
            continue
        else:
            comp_score = completeness(model_list[layer], activation_list[layer], data)
            print(f'{layer} completeness: {comp_score}')
            comp_file.write(f"{layer}: {comp_score}\n")

            if isinstance(data, Data):
                atom_colour: bool
                if dataset_name == "Mutagenicity":
                    atom_colour = True
                else:
                    atom_colour = False
                sample_graphs: dict[int, list[Graph]] = plot_samples(model,
                                                                     activation_list[layer],
                                                                     data,
                                                                     layer,
                                                                     args.clusters,
                                                                     "KMeans-Raw",
                                                                     args.num_graphs,
                                                                     args.hops,
                                                                     save_path,
                                                                     atom_colour)

                layer_graphs[layer] = sample_graphs

                concepts: list[Graph]
                pure_file.write(f"{layer}\n")
                for i, concepts in sample_graphs.items():
                    try:
                        avg_best_score = purity(concepts[:-1])
                        print(f'Concept {i} avg_score: {avg_best_score}')
                        pure_file.write(f"Concept {i}: {avg_best_score}\n")
                    except ValueError:
                        print(f'Concept {i} no score computed')
                        pure_file.write(f"Concept {i}: none\n")

    comp_file.close()
    pure_file.close()


if __name__=='__main__':
    
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', help="Path to the desired activation file")
    parser.add_argument('--clusters', type=int, required=True, help="Number of clusters, k in GCExplainer")
    parser.add_argument('--num_graphs', type=int, required=True, help="Number of graphs that are displayed per concept")
    parser.add_argument('--hops', type=int, required=True, help="Number of hops from the node of interest, n in GCExplainer")
    parser.add_argument('--config', help="Model config to extract concepts from")
    parser.add_argument('--weights', help="Path to the model trained weights to extract activations")
    args = parser.parse_args()

    if (args.config is not None) and (args.weights is not None):
        with open(osp.abspath(args.config), 'r') as config_file:
            config = yaml.safe_load(config_file)
            filename = args.config.split('/')[-1]
            dataset_name = filename.split('.')[2]
            save_name = filename.split('.')[0] + "-" + dataset_name
            main(args, dataset_name, save_name, config)
    elif args.activation is not None:
        filename = args.activation.split('/')[-1]
        dataset_name = filename.split('.')[2]
        save_name = filename.split('.')[0] + "-" + dataset_name
        main(args, dataset_name, save_name, {})

