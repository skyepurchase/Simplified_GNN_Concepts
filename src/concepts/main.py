import pickle
import os.path as osp
from argparse import Namespace

from cluster import kmeans_cluster

# Typing
from numpy.typing import NDArray


DIR = osp.dirname(__file__)


def main(args: Namespace) -> None:
    print("Loading activations...")
    print("ONLY TESTED FOR BA-SHAPES ACTIVATIONS")

    activation_list: dict    
    with open(osp.join(DIR, "..", "..", args.activation), 'rb') as file:
        activation_list = pickle.loads(file.read())

    # TODO: Probably a better way to do this -> could ignore later on
    activation_list = {'layers.3': activation_list['layers.3']} # Just want the final layer

    # TODO: Potentially implement the dimensionality reduction for SGC
    
    prediction_list: list[NDArray] = kmeans_cluster(activation_list, args.clusters)
    print(prediction_list)


if __name__=='__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', required=True, help="Path to the desired activation file")
    parser.add_argument('--clusters', type=int, required=True, help="Number of clusters, k in GCExplainer")
    args = parser.parse_args()

    main(args)

