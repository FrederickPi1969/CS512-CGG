import numpy as np
import networkx as nx
import string
import random
import queue
import copy
import math
from itertools import combinations, permutations

sigmoid_upper_bound = 0.999
sigmoid_lower_bound = 0.001

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

def modify_transitivity(G, alpha = 0.0, sigmoid = False):
    if G.number_of_nodes() < 3:
        return G
    
    target_density = nx.transitivity(G) + alpha
    if sigmoid: 
        current_density = max(min(nx.transitivity(G), sigmoid_upper_bound), sigmoid_lower_bound)
        target_sigmoid = np.log(current_density / (1 - current_density)) + alpha
        target_density = 1 / (1 + np.exp(0 - target_sigmoid))

    centerlist = sorted([(nx.clustering(G, i), i) for i in G.nodes()], reverse = True)
    if alpha > 0:
        while nx.transitivity(G) < min(target_density, 1):
            while centerlist and centerlist[0][0] == 1:
                centerlist.pop(0)
            if not centerlist:
                break
            candidates = permutations(list(G.nodes()), r = 2) - G.edges()
            G.add_edge(*sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in candidates])[0][1])
            centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist], reverse = True)
    else:		
        while nx.transitivity(G) > max(target_density, 0):
            while centerlist and centerlist[0][0] == 0:
                centerlist.pop(0)
            if not centerlist:
                break
            candidates = set(permutations(list(G.nodes()), r = 2)).intersection(set(G.edges()))
            G.remove_edge(*sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in candidates])[0][1])
            centerlist = sorted([(nx.clustering(G, j), j) for (i, j) in centerlist], reverse = True)

    return G

def modify_density(G, alpha = 0.0, sigmoid = False):
    if G.number_of_nodes() < 2:
        return G
    target_density = nx.density(G) + alpha
    if sigmoid:
        current_density = max(min(nx.density(G), sigmoid_upper_bound), sigmoid_lower_bound)
        target_sigmoid = np.log(current_density / (1 - current_density)) + alpha
        target_density = 1 / (1 + np.exp(0 - target_sigmoid))

    i = 0
    if alpha > 0:
        edgelist = sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in permutations(list(G.nodes()), r = 2) - G.edges()], reverse = True)
        while nx.density(G) < min(target_density, 1):
            G.add_edge(*edgelist[i][1])
            i += 1
    else:
        edgelist = sorted([(G.degree[i] * G.degree[j], (i, j)) for (i, j) in G.edges()], reverse = True)
        while nx.density(G) > max(target_density, 0):
            G.remove_edge(*edgelist[i][1])
            i += 1    
    return G

def modify_node_count(G, alpha = 0, max_nodes = 12):
    if alpha > 0:
        nodes_to_add = min(alpha, max_nodes - G.number_of_nodes())
        current_density = nx.density(G)
        namelist = [G.number_of_nodes() + i for i in range(nodes_to_add)]
        for i in range(nodes_to_add):
            nodelist = sorted([(G.degree[i], i) for i in G.nodes()], reverse = True)
            G.add_node(namelist[i])
            for j in range(int(current_density * (G.number_of_nodes() - 1))):
                G.add_edge(namelist[i], nodelist[j][1])
    else:
        nodes_to_remove = min(0 - alpha, G.number_of_nodes())
        current_density = nx.density(G)
        for i in range(nodes_to_remove):
            nodelist = sorted([(abs(G.degree[i] / (G.number_of_nodes() - 1) - current_density), i) for i in G.nodes()], reverse = True)
            G.remove_node(nodelist[0][1])

    return G

def forest_fire(G, alpha = 0, max_nodes = 12, burn_prob = 0.2):
    if alpha < 0:
        return None

    for i in range(alpha):
        if G.number_of_nodes() == max_nodes:
            break
        links = set()
        visited = queue.Queue()
        ambassador = random.choice(list(G.nodes()))
        visited.put(ambassador)
        while not visited.empty():
            top = visited.get()
            if top in links:
                continue
            links.add(top)
            neighbors = set(G.neighbors(top)) - links
            for i in neighbors:
                dice = random.uniform(0, 1)
                if dice < burn_prob:
                    visited.put(i)

        new_node = G.number_of_nodes()
        G.add_node(new_node)
        for link in links:
            G.add_edge(new_node, link)
    return G
