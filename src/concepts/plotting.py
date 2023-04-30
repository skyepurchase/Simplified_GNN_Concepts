import numpy as np
import pandas as pd
import os.path as osp
from torch_geometric.data import Data
from tqdm import tqdm

import torch
from torch import argmax

import matplotlib.pyplot as plt
import networkx as nx

# Typing
from matplotlib.pyplot import FigureBase
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from torch import Tensor
from typing import Iterable, Tuple, Union, List


PURITY_TEST = True


def _get_subgraphs(top_indices: NDArray,
                   y: Tensor,
                   edges: NDArray,
                   hops: int,
                   data: Union[Data, None] = None) -> Tuple[list[nx.Graph],
                                                            list[list[Union[int,str]]],
                                                            list[Tensor],
                                                            list[dict[int, str]]]:
    graphs: list[nx.Graph] = []
    color_maps: list[list[Union[int,str]]] = []
    labels: list[Tensor] = []
    node_labels: list[dict[int,str]] = []

    # Converted to dataframe for easier data manipulation
    df = pd.DataFrame(edges)

    for idx in top_indices:
        neighbours: list[Tensor] = [] 
        neighbours.append(idx)

        # Selection neighbours up to 'hops' connection away
        for _ in range(hops):
            new_neighbours: list[Tensor] = []
            for e in edges:
                # If either node of an edge is in the current neighbours then it is a "new" neighbour
                if (e[0] in neighbours) or (e[1] in neighbours):
                    new_neighbours.append(e[0])
                    new_neighbours.append(e[1])

            neighbours = neighbours + new_neighbours
            # As there will always be duplicates the set is taken
            neighbours = list(set(neighbours))

        new_G: nx.Graph = nx.Graph()

        # Filter the edge dataframe by those where both ends of the edge are present
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        remaining_edges = df_neighbours.to_numpy() # Convert back to a matrix
        new_G.add_edges_from(remaining_edges) # Add these edges to the graphs

        if PURITY_TEST:
            # Set the data for a concept as the concept label
            # This means that purity must include the node of interest
            label_dict = dict((i, y[idx]) for i in new_G.nodes)
            nx.set_node_attributes(new_G, label_dict, "label")

        # Adding label attributes for better purity score
        color_map: list[Union[str, int]] = []
        node_label: dict[int, str] = {}
        if data is None:
            for node in new_G:
                if node in top_indices:
                    color_map.append('green') # Node of interest (that is node to be classified)
                else:
                    color_map.append('pink') # Node of concept (that is node to help classification)
        else:
            atoms: list[Tuple[str, str]] = [
                ("black", "C"),
                ("red", "O"),
                ("green", "Cl"),
                ("white", "H"),
                ("darkblue", "N"),
                ("green", "F"),
                ("crimson", "Br"),
                ("yellow", "S"),
                ("orange", "P"),
                ("indigo", "I"),
                ("purple", "Na"),
                ("purple", "K"),
                ("violet", "Li"),
                ("lightgreen", "Ca")
            ]
            for node in new_G:
                atom_id: int = int(argmax(data.x[node]))
                color_map.append(atoms[atom_id][0])
                node_label[node] = f"{atoms[atom_id][1]}"

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])
        node_labels.append(node_label)

    return graphs, color_maps, labels, node_labels


def plot_samples(clustering_model: KMeans,
                 activation: Tensor,
                 data: Data,
                 layer_name: str,
                 clusters: int,
                 clustering_type: str,
                 num_graphs_viewable: int,
                 hops: int,
                 save_path: str,
                 atom_color: bool = False) -> dict[int, list[nx.Graph]]:
    # distance to each centre for each node in the graph
    activation_distances: NDArray = clustering_model.transform(activation)
    
    fig: FigureBase
    fig, axes = plt.subplots(clusters, num_graphs_viewable, figsize=(18, 3 * clusters + 2))
    fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for Raw Activations of {layer_name}', y=1.005)

    l: list[int] = list(range(0, clusters))
    sample_graphs: dict[int, list[nx.Graph]] = {}

    for cluster_id, ax_list in tqdm(zip(l, axes), desc="plotting concepts"):
        # distances to centres for this centre
        distances: NDArray = activation_distances[:,cluster_id]

        # Sort the distances and choose the top nodes (by index) to view
        # This samples the quintessential examples of the concept
        # The index is important to locate within the coo adjacency matrix
        top_indices: NDArray = np.argsort(distances)[::][:num_graphs_viewable]

        node_data: Union[Data, None]
        if atom_color:
            node_data = data
        else:
            node_data = None

        top_graphs: list[nx.Graph]
        color_maps: list[list[Union[str,int]]]
        labels: list[Tensor]
        top_graphs, color_maps, labels, node_labels = _get_subgraphs(top_indices,
                                                                     data.y,
                                                                     data.edge_index.detach().numpy().T,
                                                                     hops,
                                                                     node_data)

        # Draw each quintessential graph for the concept
        if atom_color:
            for ax, new_G, color_map, g_label, n_labels in zip(ax_list, top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax, labels=n_labels)
                ax.set_title(f'label {g_label}', fontsize=14)
        else:
            for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f'label {g_label}', fontsize=14)

        # select the "ideal" graphs and it's distance to the cluster
        sample_graphs[cluster_id] = top_graphs[:3]

    plt.savefig(osp.join(save_path, f'{layer_name}_{clustering_type}_{clusters}k_{hops}n_{num_graphs_viewable}_view.png'))

    return sample_graphs


def plot_latent_space(latent_data_list: List[NDArray],
                      labels: Tensor,
                      label_names: List[str],
                      save_path: str,
                      names: List[str]) -> None:
    """Plot the 2D dimensionality reductions to visualise the latent space of different models and layers.
    INPUT
        latent_data     : The list of DR data
        label_names     : Names for the labels
        data            : The labels for the dataset
        save_path       : The path to save the figures
    OUTPUT
        None"""
 
    fig: FigureBase
    fig, axes = plt.subplots(1, len(latent_data_list), dpi=200)
    fig.suptitle(f'Latent space of {", ".join(names)}')
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    name: Iterable = iter(names)
    for ax, data in zip(axes, latent_data_list):
        for i in range(torch.max(labels) + 1):
            ax.scatter(data[labels == i,0],
                       data[labels == i,1],
                       c=colors[i],
                       label=label_names[i])
        ax.set_title(next(name))

    # For easier comparison keep the axes the same width and height
    x_min = min([ax.get_xlim()[0] for ax in axes])
    x_max = max([ax.get_xlim()[1] for ax in axes])
    y_min = min([ax.get_ylim()[0] for ax in axes])
    y_max = max([ax.get_ylim()[1] for ax in axes])

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    axes[1].legend()
    plt.savefig(osp.join(save_path, f"latent_space.png"))

