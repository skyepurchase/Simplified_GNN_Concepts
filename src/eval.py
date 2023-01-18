import pickle
import os.path as osp
from argparse import Namespace
from networkx import Graph

from torch_geometric.data import Data, InMemoryDataset

from concepts.cluster import kmeans_cluster
from concepts.metrics import purity
from concepts.plotting import plot_samples
from datasets import get_dataset

# Typing
from sklearn.cluster import KMeans
from torch import Tensor, ge

from models.activation_classifier import Activation_Classifier


DIR = osp.dirname(__file__)


def main(args: Namespace,
         dataset_name: str) -> None:
    print("ONLY TESTED FOR SYNTHETIC ACTIVATIONS")

    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           "data/")
    temp = dataset[0]
    if isinstance(temp, Data):
        data: Data = temp
    else:
        raise ValueError(f'Expected dataset at index zero to be type {Data} received type {type(temp)}')

    save_path: str = osp.join(DIR, "../output", dataset_name)

    activation_list: dict[str, Tensor]
    with open(osp.join(DIR, "..", args.activation), 'rb') as file:
        activation_list = pickle.loads(file.read())

    # TODO: Potentially implement the dimensionality reduction for SGC
    
    model_list: dict[str, KMeans]
    _, model_list = kmeans_cluster(activation_list, args.clusters)

    layer_graphs: dict[str, dict[int, list[Graph]]] = {}
    for layer, model in model_list.items():
        print(f"Layer {layer}")
        layer_num = int(layer.split('.')[-1])
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

    if args.purity:
        layer = input("Which layer to calculate the score on? ")
        concept = int(input("Which concept to calculate the score on? "))
        avg_best_score = purity(layer_graphs[layer][concept][:-1])
        print(f'Layer {layer} Concept {concept}\navg_score: {avg_best_score}')

    classifier = Activation_Classifier(model_list['layers.2'],
                                       activation_list['layers.2'],
                                       data)
    print(f'Layer layers.3 completeness: {classifier.get_accuracy()}')


if __name__=='__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', required=True, help="Path to the desired activation file")
    parser.add_argument('--clusters', type=int, required=True, help="Number of clusters, k in GCExplainer")
    parser.add_argument('--num_graphs', type=int, required=True, help="Number of graphs that are displayed per concept")
    parser.add_argument('--hops', type=int, required=True, help="Number of hops from the node of interest, n in GCExplainer")
    parser.add_argument('--purity', action='store_true', help="Whether to calculate purity as this is costly", default=False)
    args = parser.parse_args()

    expr_name = args.activation.split('/')[-1]
    dataset_name = expr_name.split('.')[2]

    main(args, dataset_name)

