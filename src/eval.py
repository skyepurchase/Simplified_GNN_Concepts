import pickle
import os.path as osp
from os import mkdir

from concepts.cluster import kmeans_cluster
from concepts.metrics import completeness, purity
from concepts.plotting import plot_samples
from datasets import get_dataset

# Typing
from torch_geometric.data import Data, InMemoryDataset
from io import TextIOWrapper
from sklearn.cluster import KMeans
from torch import Tensor
from argparse import Namespace
from networkx import Graph

from torch_geometric.data import Data, Dataset, InMemoryDataset

from concepts.cluster import kmeans_cluster
from concepts.metrics import purity
from concepts.plotting import plot_samples
from datasets import get_dataset

# Typing
from typing import Union
from sklearn.cluster import KMeans
from torch import Tensor


DIR = osp.dirname(__file__)


def main(args: Namespace,
         dataset_name: str,
         save_name: str) -> None:
    print("ONLY TESTED FOR SYNTHETIC ACTIVATIONS")

    save_path: str = osp.join(DIR, "../output", save_name)

    if not osp.exists(save_path):
        mkdir(save_path)

    activation_list: dict[str, Tensor]
    with open(osp.join(DIR, "..", args.activation), 'rb') as file:
        activation_list = pickle.loads(file.read())

    # TODO: Potentially implement the dimensionality reduction for SGC
    
    model_list: dict[str, KMeans]
    _, model_list = kmeans_cluster(activation_list, args.clusters)

    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           "data/")
    
    if dataset_name in ["REDDIT-BINARY", "MUTAG"]:
        data = dataset
    else:
        temp = dataset[0]
        if isinstance(temp, Data):
            data: Union[Data, Dataset] = temp
        else:
            raise ValueError(f'Expected dataset at index zero to be type {Data} received type {type(temp)}')

    layer_graphs: dict[str, dict[int, list[Graph]]] = {}
    comp_file: TextIOWrapper = open(osp.join(save_path, f"{args.clusters}k-{args.hops}n-completeness.txt"), "w")
    pure_file = open(osp.join(save_path, f"{args.clusters}k-{args.hops}n-purity.txt"), "w")
    for layer, model in model_list.items():
#         classifier = Activation_Classifier(model_list[layer],
#                                            activation_list[layer],
#                                            data)
        comp_score = completeness(model_list[layer], activation_list[layer], data)
        print(f'Layer {layer} completeness: {comp_score}')
        comp_file.write(f"{layer}: {comp_score}\n")

        layer_num = int(layer.split('.')[-1])

        if isinstance(data, Data):
            sample_graphs: dict[int, list[Graph]] = plot_samples(model,
                                                                 activation_list[layer],
                                                                 data.y,
                                                                 layer_num,
                                                                 args.clusters,
                                                                 "KMeans-Raw",
                                                                 args.num_graphs,
                                                                 data.edge_index.detach().numpy().T,
                                                                 args.hops,
                                                                 save_path)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', required=True, help="Path to the desired activation file")
    parser.add_argument('--clusters', type=int, required=True, help="Number of clusters, k in GCExplainer")
    parser.add_argument('--num_graphs', type=int, required=True, help="Number of graphs that are displayed per concept")
    parser.add_argument('--hops', type=int, required=True, help="Number of hops from the node of interest, n in GCExplainer")
    args = parser.parse_args()

    expr_name = args.activation.split('/')[-1]
    dataset_name = expr_name.split('.')[2]
    save_name = expr_name.split('.')[0] + "-" + dataset_name

    main(args, dataset_name, save_name)

