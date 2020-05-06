from utils import *
from graph_operations import *
import math
import random
import numpy as np
import torch
import networkx as nx

class DensityTransform:
    def __init__(self):
        pass
    
    def get_train_alpha(self, graph_adj_tensors):
        # graphs_adj_matrices = reshapeMatrix(list(np.squeeze(graph_adj_tensors.numpy())))
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = np.random.uniform(0, 1)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = 1 - max([nx.transitivity(nx.convert_matrix.from_numpy_matrix(g)) for g in graphs_adj_matrices])
            alpha_val *= scale
            return (alpha_val, alpha_val)
        else: 
            scale = max([nx.transitivity(nx.convert_matrix.from_numpy_matrix(g)) for g in graphs_adj_matrices])
            alpha_val *= scale
            alpha_val = -alpha_val
            return (alpha_val, alpha_val)


    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        # max_n_node = graphs_adj_matrices[0].shape()[1]
        # graphs_adj_matrices = reshapeMatrix(graphs_adj_matrices)
        if alpha > 0:
            edited_adj_matrices = [nx.adjacency_matrix(densify(nx.convert_matrix.from_numpy_matrix(g), alpha)).todense() for g in graphs_adj_matrices]
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)
        else: 
            edited_adj_matrices = [nx.adjacency_matrix(sparsify_coherent_test(nx.convert_matrix.from_numpy_matrix(g), 0 - alpha)).todense() for g in graphs_adj_matrices]
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)
            

## transform via adding edges
class EdgeTransform:
    def __init__(self):
        pass
    
    def get_train_alpha(self, graph_adj_tensors):
        graphs_adj_matrices = reshapeMatrix(list(np.squeeze(graph_adj_tensors.numpy())))
        alpha_val = np.random.uniform(0, 1)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = max([(g.shape[0] * (g.shape[0] - 1) / 2) - nx.convert_matrix.from_numpy_matrix(g).number_of_edges() for g in graphs_adj_matrices])
            alpha_val *= scale
            return (alpha_val, alpha_val)
        else: 
            scale = max([nx.convert_matrix.from_numpy_matrix(g).number_of_edges() for g in graphs_adj_matrices])
            alpha_val *= scale
            alpha_val = -alpha_val
        return (alpha_val, alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        graph_shape = graphs_adj_matrices[0].shape
        graphs_adj_matrices = reshapeMatrix(graphs_adj_matrices)
        if alpha > 0:
            edited_adj_matrices = [padMatrix(nx.adjacency_matrix(add_edge_coherent(nx.convert_matrix.from_numpy_matrix(g), int(round(alpha)))).todense(), graph_shape[0]) if g.shape != (0,0) else np.zeros(graph_shape) for g in graphs_adj_matrices]
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)
        else: 
            edited_adj_matrices = [padMatrix(nx.adjacency_matrix(remove_edge_coherent(nx.convert_matrix.from_numpy_matrix(g), 0 - int(round(alpha)))).todense(), graph_shape[0]) if g.shape != (0,0) else np.zeros(graph_shape) for g in graphs_adj_matrices]
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)

## transform via adding nodes
class NodeTransform:
    def __init__(self, max_graph_nodes = 12):
        self.max_graph_nodes = max_graph_nodes

    def get_train_alpha(self, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = np.random.uniform(0, 1)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            scale = max([max_graph_nodes - g.shape[0] for g in graphs_adj_matrices])
            alpha_val *= scale
            return (math.log(alpha_val, 10), alpha_val)
        else:
            scale = max([g.shape[0] for g in graphs_adj_matrices])
            alpha_val *= scale
            alpha_val = -alpha_val
        return (math.log(alpha_val, 10), alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        if alpha > 0:
            edited_adj_matrices = [nx.adjacency_matrix(add_node(nx.convert_matrix.from_numpy_matrix(g), n = int(round(alpha)))).todense() for g in graphs_adj_matrices]
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)
        else: 
            edited_adj_matrices = [nx.adjacency_matrix(remove_node(nx.convert_matrix.from_numpy_matrix(g), n = int(round(0 - alpha)))).todense() for g in graphs_adj_matrices]
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)

class KroneckerTransform:
    def __init__(self, max_graph_nodes = 12):
        self.max_graph_nodes = max_graph_nodes

    def get_train_alpha(self, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = np.random.uniform(0, 1)
        scale = max([math.log(max_graph_nodes, g.shape[0]) for g in graphs_adj_matrices])
        alpha_val *= scale
        return (math.log(alpha_val), alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        edited_adj_matrices = [nx.adjacency_matrix(self_multiply(nx.convert_matrix.from_numpy_matrix(g), n = int(round(alpha)))).todense() for g in graphs_adj_matrices]
        return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)

class MultiplyTransform:
    def __init__(self, max_graph_nodes = 12):
        self.max_graph_nodes = max_graph_nodes

    def get_train_alpha(self, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = np.random.uniform(0, 1)
        scale = max([(max_graph_nodes - 1) / g.shape[0] for g in graphs_adj_matrices])
        alpha_val *= scale
        return (math.log(alpha_val), alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        edited_adj_matrices = [nx.adjacency_matrix(self_repetition(nx.convert_matrix.from_numpy_matrix(g), n = int(round(alpha)))).todense() for g in graphs_adj_matrices]
        return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)
