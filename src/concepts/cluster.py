from sklearn.manifold import TSNE
import torch
import os.path as osp
from tqdm import tqdm

from sklearn.cluster import KMeans

# Typing
from numpy.typing import NDArray
from typing import Tuple, List

DIR = osp.dirname(__file__)


def kmeans_cluster(activation_list: dict[str, torch.Tensor],
                   clusters: int) -> Tuple[dict[str, NDArray], dict[str, KMeans]]:

    prediction_list: dict[str, NDArray] = {}
    model_list: dict[str, KMeans] = {}
    for layer, activations in tqdm(activation_list.items(), desc="Clustering activations"):
        activation: NDArray = torch.squeeze(activations).detach().numpy()
        kmeans = KMeans(n_clusters=clusters, random_state=0) # Random state 0 for determinism

        success = True
        try:
            kmeans = kmeans.fit(activation)
        except ValueError:
            success = False
        except Exception as e:
            raise e

        if success:
            predictions: NDArray = kmeans.predict(activation)

            prediction_list[layer] = predictions
            model_list[layer] = kmeans

    return prediction_list, model_list


def tsne_reduction(activation_list: list[torch.Tensor],
                   components: int = 2) -> list[NDArray]:
    """Reduce the dimension of the activation space using t-SNE to produce a compact representation of the latent space
    INPUT
        activation_list     : A list of the activation spaces
    OUTPUT
        components          : A list of the latent spaces"""
    latent_data: List[NDArray] = []

    for activations in activation_list:
        activation: NDArray = torch.squeeze(activations).detach().numpy()
        tsne_model: TSNE = TSNE(n_components=components)
        data = tsne_model.fit_transform(activation)
        latent_data.append(data)

    return latent_data

