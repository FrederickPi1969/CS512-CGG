from graph_operations import *
import math
import random
import numpy as np
import torch
import networkx as nx
import multiprocessing

class DensityTransform:
    def __init__(self):
        self.alpha = 0
        try:
            self.cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            self.cpus = 1

    def transitivity_batch(self, g):
        return nx.transitivity(nx.convert_matrix.from_numpy_matrix(g))
    def densify_batch(self, g):
        return nx.adjacency_matrix(densify(nx.convert_matrix.from_numpy_matrix(g), self.alpha)).todense()
    def sparsify_batch(self, g):
        return nx.adjacency_matrix(sparsify(nx.convert_matrix.from_numpy_matrix(g), self.alpha)).todense()
    
    def get_train_alpha(self, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = np.random.uniform(0, 1)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = 1 - max(multiprocessing.Pool(processes = self.cpus).map(self.transitivity_batch, graphs_adj_matrices))
            alpha_val *= scale
            return (alpha_val, alpha_val)
        else: 
            scale = max(multiprocessing.Pool(processes = self.cpus).map(self.transitivity_batch, graphs_adj_matrices))
            alpha_val *= scale
            alpha_val = -alpha_val
            return (alpha_val, alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        if alpha > 0:
            self.alpha = alpha
            edited_adj_matrices = multiprocessing.Pool(processes = self.cpus).map(self.densify_batch, graphs_adj_matrices)
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices)), -1)
        else:
            self.alpha = 0 - alpha
            edited_adj_matrices = multiprocessing.Pool(processes = self.cpus).map(self.sparsify_batch, graphs_adj_matrices)
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices)), -1)

## transform via adding edges
class EdgeTransform:
    def __init__(self):
        pass
    
    def get_train_alpha(self, batch_size, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = np.random.uniform(0, 1)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = max([(len(g) * (len(g) - 1) / 2) - g.number_of_edges() for g in graphs])
            alpha_val *= scale
            return (math.log(alpha_val, 10), alpha_val)
        else: 
            scale = max([g.number_of_edges() for g in graphs])
            alpha_val *= scale
            alpha_val = -alpha_val
        return (math.log(alpha_val, 10), alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        if alpha > 0:
            return add_edge_coherent(graph, int(round(alpha)))
        else: 
            return remove_edge_coherent(graph, int(round(0 - alpha)))

## transform via adding nodes
class NodeTransform:
    def __init__(self, max_graph_nodes = 24):
        self.max_graph_nodes = max_graph_nodes

    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.uniform(0, 1)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = max([max_graph_nodes - len(g) for g in graphs])
            alpha_val *= scale
            return (math.log(alpha_val, 10), alpha_val)
        else:
            scale = max([len(g) for g in graphs])
            alpha_val *= scale
            alpha_val = -alpha_val
        return (math.log(alpha_val, 10), alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        if alpha > 0:
            return add_node(graph, n = int(round(alpha)))
        else:
            return remove_node(graph, n = int(round(0 - alpha)))

class KroneckerTransform:
    def __init__(self, max_graph_nodes = 24):
        self.max_graph_nodes = max_graph_nodes

    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.uniform(0, 1)
        scale = max([math.log(max_graph_nodes, len(g)) for g in graphs])
        alpha_val *= scale
        return (math.log(alpha_val), alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        return self_multiply(graph, n = int(round(alpha)))

class MultiplyTransform:
    def __init__(self, max_graph_nodes = 24):
        self.max_graph_nodes = max_graph_nodes

    def get_train_alpha(self, batch_size, graphs):
        alpha_val = np.random.uniform(0, 1)
        scale = max([(max_graph_nodes - 1) / len(g) for g in graphs])
        alpha_val *= scale
        return (math.log(alpha_val), alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph):
        return self_repetition(graph, n = int(round(alpha)))
