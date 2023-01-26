import pickle
import os.path as osp
from os import mkdir

from models import get_activation, get_model, register_hooks
from concepts.cluster import kmeans_cluster
from concepts.metrics import completeness, purity
from concepts.plotting import plot_samples
from datasets import get_dataset

from torch import load, cat, max

from torch_geometric.data import Data, Dataset, InMemoryDataset

# Typing
from torch_geometric.data import Data, InMemoryDataset
from io import TextIOWrapper
from sklearn.cluster import KMeans
from torch import Tensor
from argparse import Namespace
from networkx import Graph
from typing import Union, Tuple
from sklearn.cluster import KMeans
from torch import Tensor
from torch.nn import Module


DIR = osp.dirname(__file__)


#def convert(dataset: Dataset) -> Tuple[Data, Data]:
#    """Convert a graph classification Dataset into a Data object for activation extraction and Data object with node labels"""
#    x: Union[Tensor, None] = None
#    y: Union[Tensor, None] = None
#    edge_index: Union[Tensor, None] = None
#
#    graph: Union[Data, Dataset]
#    for graph in dataset:
#        if isinstance(graph, Data):
#            if (x is None) or (y is None) or (edge_index is None):
#                x = graph.x
#                edge_index = graph.edge_index
#                y = graph.y
#
#                if isinstance(x, Tensor) and isinstance(y, Tensor):
#                    print(x.shape)
#                    y = y.repeat(x.shape)
#                else:    
#                    raise TypeError("Expected x and y to have a value but received {None}")
#                breakpoint()
#            else:
#                new_graph_edge: Tensor = graph.edge_index + x.shape[0] 
#                edge_index = cat((edge_index, new_graph_edge), dim=1)
#
#                new_x: Tensor = graph.x
#                x = cat((x, new_x))
#
#                new_y: Tensor = graph.y.repeat(new_x.shape)
#                y = cat((y, new_y))
#                breakpoint()
#        else:
#            raise TypeError(f"Expected graph to be type {Data} but received type {type(graph)} instead")
#
#    return Data(), Data()


def main(args: Namespace,
         dataset_name: str,
         save_name: str,
         config: dict) -> None:
    print("ONLY TESTED FOR SYNTHETIC ACTIVATIONS")

    save_path: str = osp.join(DIR, "../output", save_name)

    if not osp.exists(save_path):
        mkdir(save_path)

    dataset: InMemoryDataset = get_dataset(dataset_name,
                                           "data/")

    if dataset_name in ["REDDIT-BINARY", "MUTAG"]:
        data: Union[Data, Dataset] = dataset
    else:
        temp = dataset[0]
        if isinstance(temp, Data):
            data: Union[Data, Dataset] = temp
        else:
            raise ValueError(f'Expected dataset at index zero to be type {Data} received type {type(temp)}')

    if args.weights is not None:
        gnn: Module = get_model(config["model"]["name"],
                                dataset.num_features,
                                dataset.num_classes,
                                config["model"]["kwargs"])
        gnn.load_state_dict(load(args.weights))
        gnn = register_hooks(gnn)
        gnn.eval()

        if isinstance(data, Data):
            _ = gnn(data.x, data.edge_index, None)
            activation_list: dict[str, Tensor] = get_activation()
        elif isinstance(data, Dataset):
            test_data: Union[Data, Dataset] = dataset[0]
            if isinstance(test_data, Data):
                data = test_data

                _ = gnn(test_data.x, test_data.edge_index, None)
                activation_list: dict[str, Tensor] = get_activation()
            else:
                raise Exception("Well if it doesn't work this is temporary")
        else:
            raise Exception("Something went wrong")
    else:
        activation_list: dict[str, Tensor]
        with open(osp.join(DIR, "..", args.activation), 'rb') as file:
            activation_list = pickle.loads(file.read())

    # TODO: Potentially implement the dimensionality reduction for SGC

    model_list: dict[str, KMeans]
    _, model_list = kmeans_cluster(activation_list, args.clusters)

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

