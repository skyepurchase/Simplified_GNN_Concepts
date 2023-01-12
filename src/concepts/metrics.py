import networkx as nx
from itertools import combinations

# Typing
from networkx import Graph


def purity(Graphs: list[Graph]): # TODO: graph_edit_distance does not have typing, maybe implement this algorithm ourselves?
    purity = 0
    num_pairs = 0
    for G1, G2 in combinations(Graphs, 2):
        score = nx.graph_edit_distance(G1, G2)
        
        if score is not None:
            purity += score
            num_pairs += 1
        else:
            raise ValueError(f'score for graphs {G1} and {G2} resulted in {None}')

    return purity / num_pairs


def completeness():
    pass

