import csv
import random
import numpy as np
from ActivationGame import ActivationGameWorld
import networkx as nx


def char_tag(c):
    return {"King": "K", "Knight": "N"}.get(c.chartype, "F")


def build_graph(world):
    G = nx.DiGraph()
    nodes = []

    for i, c in enumerate(world.characters):
        cid = f"{char_tag(c)}{i}"
        nodes.append(cid)

    edges = []
    for i, c1 in enumerate(world.characters):
        id1 = f"{char_tag(c1)}{i}"
        for j, c2 in enumerate(world.characters):
            if i != j and c1.inRange(c2.location):
                id2 = f"{char_tag(c2)}{j}"
                edges.append((id1, id2))

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G, nodes


def shortest_subgraph(G, start, end):
    path = nx.shortest_path(G, start, end)
    return G.subgraph(path).copy()


def oracle_score(world):

    G, nodes = build_graph(world)

    subgraphs = [
        shortest_subgraph(G, nodes[0], nodes[-k]) for k in range(1, world.nkings + 1)
    ]

    G_union = nx.compose_all(subgraphs)

    n_nodes = len(G_union.nodes()) - (world.nkings + 1)

    return 1 + n_nodes * 2


# -------------------------------
if __name__ == "__main__":
    runs = 10000
    score = np.zeros(runs)

    for r in range(runs):

        seed = random.randrange(2**32)
        random.seed(seed)

        world = ActivationGameWorld()

        baseline = oracle_score(world)

        score[r] = baseline


    mean = np.mean(score[score != 0])
    print(mean)
