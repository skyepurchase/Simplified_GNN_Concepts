import numpy as np

import torch
from torch import nn

from sklearn import tree

# Typing
from sklearn.cluster import KMeans
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.data import Data, Dataset
from typing import Tuple, Union


class Activation_Classifier(nn.Module):
    def __init__(self,
                 model: KMeans,
                 activations: Tensor,
                 data: Union[Data, Dataset]) -> None:
        super().__init__()
        self.data: Union[Data, Dataset] = data
        if isinstance(self.data, Dataset):
            #TODO: discuss the difference in shuffling
            temp = self.data.shuffle()
            if isinstance(temp, Dataset):
                self.data = temp
            else:
                raise ValueError(f"Expected shuffled data to be type {Dataset} received type {type(temp)}")
            self.train_idx: int = int(len(self.data) * 0.8)

        activation: NDArray = torch.squeeze(activations).detach().numpy()
        self.predictions: NDArray = model.predict(activation)

        self.accuracy: float
        self.classifier, self.accuracy = self._train()

    def _train(self) -> Tuple[tree.DecisionTreeClassifier, float]:
        train_data: NDArray
        test_data: NDArray
        train_data, test_data = self._generate_dataset()

        classifier: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier()

        if isinstance(self.data, Data):
            classifier = classifier.fit(train_data, self.data.y[self.data.train_mask])
            temp = classifier.score(test_data, self.data.y[self.data.test_mask])
        elif isinstance(self.data, Dataset):
            raise NotImplementedError
            # Code not suitable yet
            train_y = [self.data.get(i).y for i in range(self.train_idx)]
            test_y = [self.data.get(i + self.train_idx).y for i in range(self.train_idx)]
            classifier = classifier.fit(train_data, train_y)
            temp = classifier.score(test_data, test_y)
        else:
            raise ValueError(f"Expected self.data to be type {Data} or {Dataset} received type {type(self.data)}")

        if temp is not None:
            accuracy: float = float(temp)
        else:
            raise ValueError(f'Classifier accuracy returned {None}')

        return classifier, accuracy

    def _node_to_concept(self, node: int) -> int:
        """Mapping a node id to its relative concept described by the cluster model"""
        return self.predictions[node]

    def _generate_dataset(self) -> Tuple[NDArray, NDArray]:
        if isinstance(self.data, Data):
            concept_data: NDArray = np.array([[self._node_to_concept(node_idx)] for node_idx in range(len(self.data.x))])
            train_data: NDArray = concept_data[self.data.train_mask]
            test_data: NDArray = concept_data[self.data.test_mask]
        elif isinstance(self.data, Dataset):
            raise NotImplementedError
            # Code not suitable yet
            concept_data: NDArray = np.array([[self._node_to_concept(node_idx)] for node_idx in range(len(self.data))])
            train_data: NDArray = concept_data[:self.train_idx]
            test_data: NDArray = concept_data[self.train_idx:]
        else:
            raise ValueError(f"Expected self.data to be type {Data} or {Dataset} received type {type(self.data)}")

        return train_data, test_data

    def concept_to_label(self, concept: int) -> NDArray:
        return self.classifier.predict(concept)

    def get_accuracy(self) -> float:
        return self.accuracy

