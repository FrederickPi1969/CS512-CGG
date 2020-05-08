from utils import *
from graph_operations import *
import math
import random
import numpy as np
import torch
import networkx as nx

class GraphTransform:
    def __init__(self, max_n_nodes, operation, sigmoid):
        self.max_n_nodes = max_n_nodes
        self.operation = operation
        self.sigmoid = sigmoid
    
    def get_train_alpha(self, graph_adj_tensors):
<<<<<<< HEAD
        if self.sigmoid:
            alpha_val = np.random.uniform(-4, 4)
        else:
            alpha_val = np.random.uniform(-0.2, 0.2)
        return (alpha_val, alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors, graph_node_counts):
        graphs = torch_tensor_to_graphs(graph_adj_tensors, graph_node_counts)
        edited_graphs = None
        print(alpha)
        if self.operation == "transitivity":
            edited_graphs = [modify_transitivity(g, alpha) for g in graphs]
        else: 
            edited_graphs = [modify_density(g, alpha) for g in graphs]
        return graphs_to_torch_tensor(edited_graphs, self.max_n_nodes)


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
