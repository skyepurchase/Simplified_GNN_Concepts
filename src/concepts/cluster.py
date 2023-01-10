import torch
import os.path as osp

from sklearn.cluster import KMeans

# Typing
from numpy.typing import NDArray

DIR = osp.dirname(__file__)


def kmeans_cluster(activation_list: dict,
                   clusters: int) -> list[NDArray]:

    prediction_list: list[NDArray] = []
    for _, activations in activation_list.items():
        activation: NDArray = torch.squeeze(activations).detach().numpy()
        kmeans = KMeans(n_clusters=clusters, random_state=0) # Random state 0 for determinism
        kmeans = kmeans.fit(activation)
        predictions: NDArray = kmeans.predict(activation)
        prediction_list.append(predictions)

    return prediction_list
