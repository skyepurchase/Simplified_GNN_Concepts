from io import TextIOWrapper
import math
import pickle
import numpy as np
import os.path as osp
from os import mkdir
from numpy.typing import NDArray

from concepts.cluster import kmeans_cluster, tsne_reduction
from concepts.plotting import plot_latent_space
from datasets import get_dataset

# Typing
from torch import Tensor
from argparse import Namespace
from typing import Dict, List, Tuple
from torch_geometric.data import Data, InMemoryDataset


DIR = osp.dirname(__file__)


def main(args: Namespace,
         dataset_name: str,
         save_name: str) -> None:
    """Compares the concepts of two different models on the same dataset by the mutual information of the clusters.
    Where multiple layers are present the concepts from each layer are compared to identify the layer with the highest mutual information.
    The latent space of this layer is visualised to demonstrate the similarity or lack thereof.
    INPUT
        args            : Arguments passed to the program such as number of clusters
        dataset_name    : The name of the dataset used
        save_name       : The folder to save results to
    OUTPUT
        None
    """
    save_path: str = osp.join(DIR, "../output", save_name)

    if not osp.exists(save_path):
        mkdir(save_path)

    dataset: InMemoryDataset = get_dataset(dataset_name, "data/")

    temp = dataset[0]
    if isinstance(temp, Data):
        labels: Tensor = temp.y
    else:
        raise TypeError(f"Expected dataset at index 0 to be type {Data} found type {type(temp)} instead")

    assert len(args.activations) > 1
    activation_list1: Dict[str, Tensor]
    activation_list2: Dict[str, Tensor]
    with open(osp.join(DIR, "..", args.activations[0]), "rb") as file:
        activation_list1 = pickle.loads(file.read())

    with open(osp.join(DIR, "..", args.activations[1]), "rb") as file:
        activation_list2 = pickle.loads(file.read())

    predictions1: Dict[str, NDArray]
    predictions2: Dict[str, NDArray]
    predictions1, _ = kmeans_cluster(activation_list1, args.clusters)
    predictions2, _ = kmeans_cluster(activation_list2, args.clusters)

    mutual_file: TextIOWrapper = open(osp.join(save_path, "similarity.txt"), "w")
    best_layer: Tuple[str, str] = ("", "")
    best_MI: float = 0
    for (name1, pred1), (name2, pred2) in zip(predictions1.items(), predictions2.items()):
        total: int = len(pred1)
        assert total == len(pred2)

        if ("pool" in name1) or ("pool" in name2):
            continue
        if ("lin" in name1) or ("lin" in name2):
            continue

        prob_U: List[int] = []      # Probability that a random node falls into concept i
        prob_V: List[int] = []      # Same as above
        prob_UV: List[List[int]] = []     # Probability that a random node falls into the same concept in both models
        for i in range(args.clusters):
            prob_U.append(np.sum(pred1 == i) / total)
            prob_V.append(np.sum(pred2 == i) / total)
           
            prob_UV_i: List[int] = []
            for j in range(args.clusters):
                nodes = np.where(pred1 == i, np.where(pred2 == j, 1, 0), 0)
                prob_UV_i.append(np.sum(nodes) / total)

            prob_UV.append(prob_UV_i)

        mutual_info: float = 0
        for i in range(args.clusters):
            mutual_info_i: float = 0
            for j in range(args.clusters):
                if prob_UV[i][j] == 0:
                    # given the multiplication by prob_UV the value does not matter
                    log_prob: float = 0
                else:
                    log_prob: float = math.log(prob_UV[i][j] / (prob_U[i] * prob_V[j]))
                mutual_info_i += prob_UV[i][j] * log_prob
            mutual_info += mutual_info_i

        print(f"{name1}: {mutual_info}")
        mutual_file.write(f"{name1} adjusted MI: {mutual_info}\n")

        if mutual_info >= best_MI:
            # Want latest layer possible so equality is desired
            best_MI = mutual_info
            best_layer = (name1, name2)

    print(f"best layers: {best_layer}")
    mutual_file.write(f"best layers for join: {best_layer}\n")

    # t-SNE dimensionality reduction to visualise the latent space
    latent_data = tsne_reduction([activation_list1[best_layer[0]], activation_list2[best_layer[1]]], 2)
    plot_latent_space(latent_data, labels, save_path, save_name.split("-")[:-2])


if __name__=='__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', nargs="+", help="Path to the desired activation file")
    parser.add_argument('--clusters', type=int, required=True, help="Number of clusters, k in GCExplainer")
    parser.add_argument('--hops', type=int, required=True, help="Number of hops from the node of interest, n in GCExplainer")
    args = parser.parse_args()

    filename = args.activations[0].split('/')[-1]
    comparison = args.activations[1].split('/')[-1]
    dataset_name = filename.split('.')[2]
    save_name = filename.split('.')[0] + "-" + comparison.split('.')[0] + "-" + dataset_name
    main(args, dataset_name, save_name)

