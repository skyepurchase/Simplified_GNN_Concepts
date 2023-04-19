from io import TextIOWrapper
import math
import pickle
import numpy as np
import os.path as osp
from os import mkdir
from numpy.typing import NDArray
from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information
from sklearn.metrics import adjusted_mutual_info_score

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
        activation_list1 = dict(filter(lambda x: x[0] != "layers.0", activation_list1.items()))

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

        bins: List[int] = [i for i in range(args.clusters + 1)]
       
        # Number of nodes in each models concept
        num_nodes1: NDArray = np.unique(pred1, return_counts=True)[1]
        num_nodes2: NDArray = np.unique(pred2, return_counts=True)[1]

        # Probability that a random node falls into a models concept
        prob_U: NDArray = num_nodes1 / total
        prob_V: NDArray = num_nodes2 / total

        cluster: NDArray = np.where(pred1 == 0, pred2, -1)
        
        # Number of nodes in model 1 concepts that are also in model 2 concepts
        contingency_matrix: NDArray = np.histogram(cluster, bins=bins)[0]
        for i in range(args.clusters - 1):
            cluster = np.where(pred1 == i+1, pred2, -1)
            contingency_matrix = np.vstack((contingency_matrix,
                                            np.histogram(cluster, bins=bins)[0]))

        # Probability of a node being in model 1 concept i and model 2 concept j
        prob_UV: NDArray = contingency_matrix / total
        log_prob_UV: NDArray = np.where(prob_UV > 0, np.log2(prob_UV), 0)

        norm_prob: NDArray = np.matmul(np.expand_dims(prob_U, 1),
                                       np.expand_dims(prob_V, 1).T)
        log_prob_norm: NDArray = np.where(norm_prob > 0, np.log2(norm_prob), 0)

        # Based on mutual information equation
        mutual_info_matrix: NDArray = np.multiply(prob_UV,
                                                  log_prob_UV - log_prob_norm)
        mutual_info: float = mutual_info_matrix.sum()

        # Based on entropy equation
        log_prob_U: NDArray = np.where(prob_U > 0, np.log2(prob_U), 0)
        entropy_U_matrix: NDArray = np.multiply(prob_U, log_prob_U)
        entropy_U: float = -entropy_U_matrix.sum()

        log_prob_V: NDArray = np.where(prob_V > 0, np.log2(prob_V), 0)
        entropy_V_matrix: NDArray = np.multiply(prob_V, log_prob_V)
        entropy_V: float = -entropy_V_matrix.sum()

        # Due to complexity of calculation expected mutual information is done by sklearn
        EMI: float = expected_mutual_information(contingency_matrix, args.clusters)

        AMI: float = (mutual_info - EMI) / (max(entropy_U, entropy_V) - EMI)

        print(f"{name1}: {AMI}")
        mutual_file.write(f"{name1} adjusted MI: {AMI}\n")

        if AMI >= best_MI:
            # Want latest layer possible so equality is desired
            best_MI = AMI 
            best_layer = (name1, name2)

    print(f"best layers: {best_layer}")
    mutual_file.write(f"best layers for join: {best_layer}\n")

    # t-SNE dimensionality reduction to visualise the latent space
    latent_data = tsne_reduction([activation_list1[best_layer[0]], activation_list2[best_layer[1]]], 2)

    if dataset_name == "BA-Shapes":
        plot_latent_space(latent_data,
                          labels,
                          ["BA", "ceiling", "floor", "roof"],
                          save_path,
                          save_name.split("-")[:-2])
    else:
        plot_latent_space(latent_data,
                          labels,
                          ["base", "motif"],
                          save_path,
                          save_name.split("-")[:-2])


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

