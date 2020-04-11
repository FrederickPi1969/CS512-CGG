import numpy as np
import networkx as nx
import string
import random
import queue
from itertools import combinations

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
def densify(G, target_density = 1):
    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()], reverse = True)
    while nx.transitivity(G) < min(target_density, 1):
        while centerlist[0][0] == 1:
            centerlist.pop(0)
        candidates = combinations(list(G.nodes()), r = 2) - G.edges()
        G.add_edge(*sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in candidates])[0][1])
        centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist], reverse = True)

# densify via triad breaking
def sparsify(G, target_density = 0):
    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()])
    while nx.transitivity(G) > max(target_density, 0):
        while centerlist[0][0] == 0:
            centerlist.pop(0)
        candidates = set(combinations(list(G.nodes()), r = 2)).intersection(set(G.edges()))
        G.remove_edge(*sorted([(max(G.degree[i], G.degree[j]) - min(G.degree[i], G.degree[j]), (i, j)) for (i, j) in candidates])[0][1])
        centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist])

def add_edge_coherent(G, n = 1, descending = True):
    edgelist = sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in combinations(list(G.nodes()), r = 2) - G.edges()], reverse = descending)
    for j in range(min(n, len(edgelist))):
        G.add_edge(*edgelist[j][1])

def remove_edge_coherent(G, n = 1, descending = False):
    edgelist = sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in G.edges()], reverse = descending)
    for j in range(min(n, len(edgelist))):
        G.remove_edge(*edgelist[j][1])

def remove_edge_difference(G, n = 1):
    edgelist = sorted([(max(G.degree[i], G.degree[j]) - min(G.degree[i], G.degree[j]), (i, j)) for (i, j) in G.edges()])
    for j in range(min(n, len(edgelist))):
        G.remove_edge(*edgelist[j][1])
        

def add_node(G, namelist = None, n = 1, m = 1, descending = True):
    namelist = namelist if namelist else [len(G) + i for i in range(n)]
    for i in range(min(n, len(namelist))):
        nodelist = sorted([(G.degree[i], i) for i in G.nodes()], reverse = descending)
        G.add_node(namelist[i])
        for j in range(min(m, len(nodelist))):
            G.add_edge(namelist[i], nodelist[j][1])


def remove_node(G, n = 1, descending = False):
    nodelist = sorted([(G.degree[i], i) for i in G.nodes()], reverse = descending)
    print(nodelist)
    for j in range(min(n, len(nodelist))):
        G.remove_node(nodelist[j][1])
