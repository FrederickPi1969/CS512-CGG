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
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = random.uniform(0.1, 0.25)
        coin = random.uniform(0, 1)
        if coin <= 0.5:
            ##scale = 1 - max([nx.transitivity(nx.convert_matrix.from_numpy_matrix(g)) for g in graphs_adj_matrices])
            scale = 1
            alpha_val *= scale
            print(alpha_val)
            return (alpha_val, alpha_val)
        else:
            #scale = min([nx.transitivity(nx.convert_matrix.from_numpy_matrix(g)) for g in graphs_adj_matrices])
            scale = 1
            alpha_val *= scale
            alpha_val = -alpha_val
            print(alpha_val)
            return (alpha_val, alpha_val)


    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
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
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        alpha_val = random.randint(1, graphs_adj_matrices[0].shape[0])
        coin = random.uniform(0, 1)
        if coin <= 0.5:
            ## scale = max([(g.shape[0] * (g.shape[0] - 1) / 2) - nx.convert_matrix.from_numpy_matrix(g).number_of_edges() for g in graphs_adj_matrices])
            scale = 1
            alpha_val *= scale
            print(alpha_val)
            return (alpha_val, alpha_val)
        else:
            ## scale = max([nx.convert_matrix.from_numpy_matrix(g).number_of_edges() for g in graphs_adj_matrices])
            scale = 1
            alpha_val *= scale
            alpha_val = -alpha_val
            print(alpha_val)
        return (alpha_val, alpha_val)

    # transform graph based on alpha
    def get_target_graph(self, alpha, graph_adj_tensors):
        graphs_adj_matrices = list(np.squeeze(graph_adj_tensors.numpy()))
        nx.draw(nx.convert_matrix.from_numpy_matrix(graphs_adj_matrices[0]), with_labels = True)
        if alpha > 0:
            edited_adj_matrices = []
            for i in graphs_adj_matrices:
                original_shape = i.shape
                nonzeros = np.nonzero(i)
                number_of_nodes = nonzeros[0][len(nonzeros[0]) - 1]
                unpadded_matrix = i[np.ix_(list(range(0, number_of_nodes)), list(range(0, number_of_nodes)))]
                edited_matrix = nx.adjacency_matrix(add_edge_coherent(nx.convert_matrix.from_numpy_matrix(unpadded_matrix), int(round(alpha)))).todense()
                padded_matrix = np.zeros(original_shape)
                padded_matrix[:edited_matrix.shape[0], :edited_matrix.shape[1]] = edited_matrix
                edited_adj_matrices.append(padded_matrix)
            ##edited_adj_matrices = [nx.adjacency_matrix(add_edge_coherent(nx.convert_matrix.from_numpy_matrix(g), int(round(alpha)))).todense() for g in graphs_adj_matrices]
            nx.draw(nx.convert_matrix.from_numpy_matrix(edited_adj_matrices[0]), with_labels = True)
            return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)
        else:
            edited_adj_matrices = []
            for i in graphs_adj_matrices:
                original_shape = i.shape
                nonzeros = np.nonzero(i)
                number_of_nodes = nonzeros[0][len(nonzeros[0]) - 1]
                unpadded_matrix = i[np.ix_(list(range(0, number_of_nodes)), list(range(0, number_of_nodes)))]
                edited_matrix = nx.adjacency_matrix(remove_edge_coherent(nx.convert_matrix.from_numpy_matrix(unpadded_matrix), 0 - int(round(alpha)))).todense()
                padded_matrix = np.zeros(original_shape)
                padded_matrix[:edited_matrix.shape[0], :edited_matrix.shape[1]] = edited_matrix
                edited_adj_matrices.append(padded_matrix)
            ##edited_adj_matrices = [nx.adjacency_matrix(remove_edge_coherent(nx.convert_matrix.from_numpy_matrix(g), 0 - int(round(alpha)))).todense() for g in graphs_adj_matrices]
            nx.draw(nx.convert_matrix.from_numpy_matrix(edited_adj_matrices[0]), with_labels = True)
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
