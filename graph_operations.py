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
def densify(G, n = 1, target_density = 1):
    centerlist = [(nx.clustering(G, i), i) for i in G.nodes()].sort()
    while nx.transitivity(G) < max(target_density, 1):
        while centerlist[0][0] == 1:
            centerlist.pop(0)
        candidates = combinations(list(G.nodes()), r = 2) - G.edges()
        G.add_edge(*[(G.degree[i] * G.degree[j], (i, j)) for (i, j) in candidates].sort()[0][1])
        centerlist = [(nx.clustering(G, j), j) for (i, j) in centerlist].sort()

# densify via triad breaking
def sparsify(G, n = 1, target_density = 0):
    centerlist = [(nx.clustering(G, i), i) for i in G.nodes()].sort(reverse = True)
    while nx.transitivity(G) > max(target_density, 1):
        while centerlist[0][0] == 0:
            centerlist.pop(0)
        candidates = intersection(set(combinations(list(G.nodes()), r = 2)), set(G.edges()))
        G.remove_edge(*[(max(G.degree[i], G.degree[j]) - min(G.degree[i], G.degree[j]), (i, j)) for (i, j) in candidates].sort()[0][1])
        centerlist = [(nx.clustering(G, j), j) for (i, j) in centerlist].sort()

def add_edge_coherent(G, n = 1, rev = True):
    for i in range(n):
        edgelist = [(G.degree[i] * G.degree[j], (i, j)) for (i, j) in G.edges()].sort(reverse = rev)
        for j in range(min(m, len(edgelist))):
            G.remove_edge(*edgelist[j][1])

def remove_edge_coherent(G, n = 1, rev = True):
    for i in range(n):
        edgelist = [(G.degree[i] * G.degree[j], (i, j)) for (i, j) in G.edges()].sort(reverse = rev)
        for j in range(min(m, len(edgelist))):
            G.remove_edge(*edgelist[j][1])

def remove_edge_difference(G, n = 1):
    for i in range(n):
        edgelist = [(max(G.degree[i], G.degree[j]) - min(G.degree[i], G.degree[j]), (i, j)) for (i, j) in G.edges()].sort()
        for j in range(min(m, len(edgelist))):
            G.remove_edge(*edgelist[j][1])
        

def add_node(G, namelist = None, n = 1, m = 1):
    namelist = namelist if namelist else [len(G) + i for i in range(n)]
    for i in range(min(n, len(namelist))):
        nodelist = [(G.degree[i], i) for i in G.nodes()].sort()
        G.add_node(namelist[i])
        for j in range(min(m, len(nodelist))):
            G.add_edge(namelist[i], nodelist[j][1])


def remove_node(G, n = 1, weighted = True):
    for i in range(n):
        nodelist = [(G.degree[i], i) for i in G.nodes()].sort()
        for j in range(min(m, len(nodelist))):
            G.remove_node(nodelist[j][1])
