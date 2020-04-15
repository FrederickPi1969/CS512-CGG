import networkx as nx
import torch
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
from graph_operations import *
import math

class DensityTransform:
    def __init__(self):
        pass
    
    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.randint(1, self.alpha_max)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = 1 - max([nx.transitivity(g) for g in graphs])
            alpha_val *= scale
            return alpha_val
        else: 
            scale = max([nx.transitivity(g) for g in graphs])
            alpha_val *= scale
            alpha_val = -alpha_val
        return alpha_val

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        if alpha > 0:
            return densify(graph, alpha)
        else: 
            return sparsify(graph, 0 - alpha)

## transform via adding edges
class EdgeTransform:
    def __init__(self):
        pass
    
    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.randint(1, self.alpha_max)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = max([(len(g) * (len(g) - 1) / 2) - g.number_of_edges() for g in graphs])
            alpha_val *= scale
            return alpha_val
        else: 
            scale = max([g.number_of_edges() for g in graphs])
            alpha_val *= scale
            alpha_val = -alpha_val
        return alpha_val

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        if alpha > 0:
            return add_edge_coherent(graph, int(round(alpha)))
        else: 
            return remove_edge_coherent(graph, int(round(0 - alpha)))

## transform via adding nodes
class NodeTransform:
    def __init__(self, max_graph_nodes):
        self.max_graph_nodes = max_graph_nodes
    
    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.randint(1, self.alpha_max)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = max([max_graph_nodes - len(g) for g in graphs])
            alpha_val *= scale
            return alpha_val
        else: 
            scale = max([len(g) for g in graphs])
            alpha_val *= scale
            alpha_val = -alpha_val
        return alpha_val

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        if alpha > 0:
            return add_node(graph, n = int(round(alpha)))
        else: 
            return remove_node(graph, n = int(round(0 - alpha)))

class KroneckerTransform:
    def __init__(self, max_graph_nodes):
        self.max_graph_nodes = max_graph_nodes
    
    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.randint(1, self.alpha_max)
        scale = max([math.log(max_graph_nodes, len(g)) for g in graphs])
        alpha_val *= scale
        return alpha_val

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        return self_multiply(graph, n = int(round(alpha)))

class MultiplyTransform:
    def __init__(self, max_graph_nodes):
        self.max_graph_nodes = max_graph_nodes
    
    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.randint(1, self.alpha_max)
        scale = max([(max_graph_nodes - 1) / len(g) for g in graphs])
        alpha_val *= scale
        return alpha_val

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        return self_repetition(graph, n = int(round(alpha)))
