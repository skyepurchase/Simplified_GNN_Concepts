import networkx as nx

# Typing
from networkx import Graph
from typing import Union


def purity(Graphs: list[Graph],
           max_nodes: int = 13) -> float:
    purity: float = 0
    num_pairs: int = 0

    top_graph: Graph = Graphs[0]
    if top_graph.number_of_nodes() > max_nodes:
        raise  ValueError(f"Top graph has more than max_nodes ({max_nodes})")
    else:
        for G1 in Graphs[1:]:
            if G1.number_of_nodes() > max_nodes:
                raise  ValueError(f"Top graph has more than max_nodes ({max_nodes})")

            score: Union[float, None] = nx.graph_edit_distance(top_graph, G1)

            if score is not None:
                purity += score
                num_pairs += 1
            else:
                raise ValueError(f'score for graphs {top_graph} and {G1} resulted in {None}')

        return purity / num_pairs


def completeness():
    pass

