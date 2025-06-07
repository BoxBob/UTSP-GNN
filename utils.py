import numpy as np
import networkx as nx

def generate_tsp_instance(n_cities, seed=None):
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.rand(n_cities, 2)
    return coords

def compute_distance_matrix(coords):
    n = coords.shape[0]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return D

def optimal_tour_length(D):
    n = D.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=D[i, j])
    cycle = nx.approximation.traveling_salesman_problem(G, cycle=True, weight='weight')
    length = sum(D[cycle[i], cycle[(i+1)%n]] for i in range(n))
    return length, cycle

def gap_percent(pred, opt):
    return 100.0 * (pred - opt) / opt
