from typing import Tuple
import torch
import os.path as osp

from sklearn.cluster import KMeans

# Typing
from numpy.typing import NDArray

DIR = osp.dirname(__file__)


def kmeans_cluster(activation_list: dict,
                   clusters: int) -> Tuple[list[NDArray], list[KMeans]]:

    prediction_list: list[NDArray] = []
    model_list: list[KMeans] = []
    for _, activations in activation_list.items():
        activation: NDArray = torch.squeeze(activations).detach().numpy()
        kmeans = KMeans(n_clusters=clusters, random_state=0) # Random state 0 for determinism

        kmeans = kmeans.fit(activation)
        predictions: NDArray = kmeans.predict(activation)

        prediction_list.append(predictions)
        model_list.append(kmeans)

    return prediction_list, model_list
