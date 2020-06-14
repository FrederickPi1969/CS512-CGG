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

    def calibrate_node_num(self, graph_tensors, edited_graphs):
        for i,graph in enumerate(edited_graphs):
            for j in range(graph.number_of_nodes()):
                graph_tensors[i][j,j] = 1
        return graph_tensors
    
    def get_train_alpha(self, graph_adj_tensors):
        if self.operation == "node_count":
            alpha_val = np.random.randint(-3, 4)
        elif self.operation == "forest_fire":
            alpha_val = np.random.randint(1, 3)
        elif self.sigmoid:
            alpha_val = np.random.uniform(-4, 4)
        else:
            alpha_val = np.random.uniform(0,0.5)

        return (alpha_val, alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors, graph_node_counts):
        graphs = torch_tensor_to_graphs(graph_adj_tensors, graph_node_counts)
        edited_graphs = None
        # print(alpha)
        if self.operation == "transitivity":
            edited_graphs = [modify_transitivity(g, alpha) for g in graphs]
        elif self.operation == "density": 
            edited_graphs = [modify_density(g, alpha) for g in graphs]
        elif self.operation == "forest_fire":
            edited_graphs = [forest_fire(g, alpha, self.max_n_nodes) for g in graphs]
        else:
            edited_graphs = [modify_node_count(g, alpha, self.max_n_nodes) for g in graphs]
        return self.calibrate_node_num(graphs_to_torch_tensor(edited_graphs, self.max_n_nodes), edited_graphs)


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
