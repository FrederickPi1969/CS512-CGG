import numpy as np
import networkx as nx
import string
import random
import queue
import copy
from itertools import combinations, permutations

# self multiply through kronecker product
def self_multiply(G, n = 1):
    adj_matrix = nx.adjacency_matrix(G).todense()
    new_matrix = adj_matrix
    for i in range(n):
        new_matrix = np.kron(new_matrix, adj_matrix)
    return nx.convert_matrix.from_numpy_matrix(new_matrix)

# self repeating substructure
def self_repetition(G, n = 1, linknode = 0):
    new_graph = nx.null_graph()
    new_graph.add_node(0)
    for i in range(n):
        new_graph = nx.disjoint_union(new_graph, G.copy())
        new_graph.add_edge(0, i * len(G) + linknode + 1)
    return new_graph

# densify via triad closing
def densify_to(Graph, target_density = 1):
    G = copy.deepcopy(Graph)
    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()], reverse = True)
    while nx.transitivity(G) < min(target_density, 1):
        while centerlist and centerlist[0][0] == 1:
            centerlist.pop(0)
        if not centerlist:
            break
        candidates = permutations(list(G.nodes()), r = 2) - G.edges()
        G.add_edge(*sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in candidates])[0][1])
        centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist], reverse = True)
    return G

# densify via triad closing
def densify(Graph, increase_density = 0.1):
    G = copy.deepcopy(Graph)
    target_density = nx.transitivity(G) + increase_density
    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()], reverse = True)
    while nx.transitivity(G) < min(target_density, 1):
        while centerlist and centerlist[0][0] == 1:
            centerlist.pop(0)
        if not centerlist:
            break
        candidates = permutations(list(G.nodes()), r = 2) - G.edges()
        G.add_edge(*sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in candidates])[0][1])
        centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist], reverse = True)
    return G

# sparsify via triad breaking
def sparsify_to(Graph, target_density = 0):
    G = copy.deepcopy(Graph)
    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()])
    while nx.transitivity(G) > max(target_density, 0):
        while centerlist and centerlist[0][0] == 0:
            centerlist.pop(0)
        if not centerlist:
            break
        candidates = set(permutations(list(G.nodes()), r = 2)).intersection(set(G.edges()))
        G.remove_edge(*sorted([(max(G.degree[i], G.degree[j]) - min(G.degree[i], G.degree[j]), (i, j)) for (i, j) in candidates])[0][1])
        centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist])
    return G

# sparsify via triad breaking
def sparsify(Graph, decrease_density = 0.1):
    G = copy.deepcopy(Graph)
    target_density = nx.transitivity(G) - decrease_density
    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()])
    while nx.transitivity(G) > max(target_density, 0):
        while centerlist and centerlist[0][0] == 0:
            centerlist.pop(0)
        if not centerlist:
            break
        candidates = set(permutations(list(G.nodes()), r = 2)).intersection(set(G.edges()))
        G.remove_edge(*sorted([(max(G.degree[i], G.degree[j]) - min(G.degree[i], G.degree[j]), (i, j)) for (i, j) in candidates])[0][1])
        centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist])
    return G

# sparsify via triad breaking
def sparsify_coherent_test(Graph, decrease_density = 0.1):
    G = copy.deepcopy(Graph)
    target_density = nx.transitivity(G) - decrease_density
    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()], reverse = True)
    while nx.transitivity(G) > max(target_density, 0):
        while centerlist and centerlist[0][0] == 0:
            centerlist.pop(0)
        if not centerlist:
            break
        candidates = set(permutations(list(G.nodes()), r = 2)).intersection(set(G.edges()))
        G.remove_edge(*sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in candidates])[0][1])
        centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist], reverse = True)
    return G

def add_edge_coherent(Graph, n = 1, descending = True):
    G = copy.deepcopy(Graph)
    edgelist = sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in permutations(list(G.nodes()), r = 2) - G.edges()], reverse = descending)
    for j in range(min(n, len(edgelist))):
        G.add_edge(*edgelist[j][1])
    return G

def remove_edge_coherent(Graph, n = 1, descending = True):
    G = copy.deepcopy(Graph)
    edgelist = sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in G.edges()], reverse = descending)
    for j in range(min(n, len(edgelist))):
        G.remove_edge(*edgelist[j][1])
    return G

def remove_edge_difference(Graph, n = 1):
    G = copy.deepcopy(Graph)
    edgelist = sorted([(max(G.degree[i], G.degree[j]) - min(G.degree[i], G.degree[j]), (i, j)) for (i, j) in G.edges()])
    for j in range(min(n, len(edgelist))):
        G.remove_edge(*edgelist[j][1])
    return G

def add_node(Graph, namelist = None, n = 1, m = 1, descending = True):
    G = copy.deepcopy(Graph)
    namelist = namelist if namelist else [len(G) + i for i in range(n)]
    for i in range(min(n, len(namelist))):
        nodelist = sorted([(G.degree[i], i) for i in G.nodes()], reverse = descending)
        G.add_node(namelist[i])
        for j in range(min(m, len(nodelist))):
            G.add_edge(namelist[i], nodelist[j][1])
    return G


def remove_node(Graph, n = 1, descending = False):
    G = copy.deepcopy(Graph)
    nodelist = sorted([(G.degree[i], i) for i in G.nodes()], reverse = descending)
    print(nodelist)
    for j in range(min(n, len(nodelist))):
        G.remove_node(nodelist[j][1])
    return G
