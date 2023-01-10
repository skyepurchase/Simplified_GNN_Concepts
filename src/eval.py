import pickle
import os.path as osp
from argparse import Namespace

from torch_geometric.data import Data, InMemoryDataset

from concepts.cluster import kmeans_cluster
from concepts.plotting import plot_samples
from datasets import get_dataset

# Typing
from sklearn.cluster import KMeans


DIR = osp.dirname(__file__)


def main(args: Namespace,
         dataset_name: str) -> None:
    print("ONLY TESTED FOR BA-SHAPES ACTIVATIONS")

    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           "data/")
    temp = dataset[0]
    if isinstance(temp, Data):
        data: Data = temp
    else:
        raise ValueError(f'Expected dataset at index zero to be type {Data} received type {type(temp)}')

    save_path: str = osp.join(DIR, "../output", dataset_name)

    activation_list: dict    
    with open(osp.join(DIR, "..", args.activation), 'rb') as file:
        activation_list = pickle.loads(file.read())

    print(activation_list.keys())
    # TODO: Probably a better way to do this -> could ignore later on
    activation_list = {'layers.3': activation_list['layers.3']} # Just want the final layer

    # TODO: Potentially implement the dimensionality reduction for SGC
    
    model_list: list[KMeans]
    _, model_list = kmeans_cluster(activation_list, args.clusters)

    _ = plot_samples(model_list[0],
                     activation_list['layers.3'],
                     data.y,
                     3,
                     args.clusters,
                     "KMeans-Raw",
                     args.num_graphs,
                     data.edge_index.detach().numpy().T,
                     args.hops,
                     save_path)


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

    main(args, dataset_name)

