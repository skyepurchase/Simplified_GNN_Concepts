import torch
import os.path as osp

from sklearn.cluster import KMeans

# Typing
from numpy.typing import NDArray
from typing import Tuple

DIR = osp.dirname(__file__)


def kmeans_cluster(activation_list: dict[str, torch.Tensor],
                   clusters: int) -> Tuple[dict[str, NDArray], dict[str, KMeans]]:

    prediction_list: dict[str, NDArray] = {}
    model_list: dict[str, KMeans] = {}
    for layer, activations in activation_list.items():
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
