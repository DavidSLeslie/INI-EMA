import csv
import random
import numpy as np
from ActivationGame import ActivationGameWorld
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import math
from collections import defaultdict


def steiner_subgraph(G, start, targets):
    k = len(targets)
    idx = {t:i for i,t in enumerate(targets)}

    INF = 10**9
    dp = defaultdict(lambda: defaultdict(lambda: INF))
    parent = {}

    # init
    for t in targets:
        dp[t][1 << idx[t]] = 1
        parent[(t, 1 << idx[t])] = None

    # DP
    for mask in range(1, 1<<k):

        # merge subsets
        for v in G.nodes:
            sub = mask
            while sub:
                other = mask ^ sub
                if other:
                    val = dp[v][sub] + dp[v][other] - 1
                    if val < dp[v][mask]:
                        dp[v][mask] = val
                        parent[(v,mask)] = ("merge", sub, other)
                sub = (sub-1) & mask

        # relax edges (reverse propagation)
        improved = True
        while improved:
            improved = False
            for u,v in G.edges:
                if dp[v][mask] + 1 < dp[u][mask]:
                    dp[u][mask] = dp[v][mask] + 1
                    parent[(u,mask)] = ("edge", v)
                    improved = True

    # Graph reconstruction
    used_nodes = set()
    used_edges = set()
    full = (1<<k)-1

    def backtrack(v, mask):
        used_nodes.add(v)
        p = parent.get((v,mask))
        if not p:
            return
        if p[0] == "edge":
            u = p[1]
            used_edges.add((v,u))
            backtrack(u, mask)
        else:
            _, m1, m2 = p
            backtrack(v, m1)
            backtrack(v, m2)

    backtrack(start, full)

    H = G.subgraph(used_nodes).copy()
    H.add_edges_from(used_edges)

    return H


def char_tag(c):
    return {"King": "K", "Knight": "N"}.get(c.chartype, "F")


def build_graph(world_chars):
    G = nx.DiGraph()
    nodes = []

    for i, c in enumerate(world_chars):
        cid = f"{char_tag(c)}{i}"
        nodes.append(cid)

    edges = []
    for i, c1 in enumerate(world_chars):
        id1 = f"{char_tag(c1)}{i}"
        for j, c2 in enumerate(world_chars):
            if i != j and c1.inRange(c2.location):
                id2 = f"{char_tag(c2)}{j}"
                edges.append((id1, id2))
                if char_tag(c2) == "K":
                    edges.append((id2, id1))

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G, nodes


def shortest_subgraph(G, start, end):
    path = nx.shortest_path(G, start, end)
    return G.subgraph(path).copy()


def oracle_score(world):

    G, nodes = build_graph(world.characters)

    subgraphs = [
        shortest_subgraph(G, nodes[0], nodes[-k]) for k in range(1, world.nkings + 1)
    ]

    G_union = nx.compose_all(subgraphs)
    
    G_steiner = steiner_subgraph(G, nodes[0], targets=nodes[-4::])

    n_nodes_union = len(G_union.nodes()) - (world.nkings + 1)
    n_nodes_steiner = len(G_steiner.nodes()) - (world.nkings + 1)

    return (1 + n_nodes_union * 2, 1 + n_nodes_steiner * 2)


if __name__ == "__main__":

    runs = 1000
    score = np.zeros(runs)
    optimal = np.zeros(runs)
    matching = np.zeros(runs)

    for r in range(runs):

        seed = random.randrange(2**32)
        random.seed(seed)

        world = ActivationGameWorld()

        baseline, opt = evaluate_world(world)

        score[r] = baseline
        optimal[r] = opt
        
        if baseline == opt:
            matching[r] = 1
            
        if baseline < opt:
            "PROBLEM"       # So far this has never happened in 10k+ runs


    mean = np.mean(score[score != 0])
    minimum, maximum = score.min(), score.max()
    print("Union Score:", mean, minimum, maximum)
    mean = np.mean(optimal[optimal != 0])
    minimum, maximum = optimal.min(), optimal.max()
    print("Optimal Score:", mean, minimum, maximum)
    matches = len(matching[matching == 1])
    print(f"Union matches Optimal: {matches / runs * 100}%")
