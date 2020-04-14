import networkx as nx
import torch
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
from graph_operations import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from networkx.generators import random_graphs


def edit_graph(A, dataArgs, modelArgs, **args):
    shape_list = A.shape
    edited_list = []
    if len(A.shape) > 2:
        if A.shape[-1] == 1:
            A = A.squeeze(-1)
    if len(A.shape) > 2:
        for graph in A:
            graph = graph.numpy().reshape(dataArgs['max_n_node'], dataArgs['max_n_node'])
            graph = nx.from_numpy_matrix(graph)
            edited_list.append(graph)

    for i,graph in enumerate(edited_list):
        if modelArgs['edit_method'] == 'densify':
            densify(graph)
        elif modelArgs['edit_method'] == 'sparsify':
            sparsify(graph)
        else
            raise ValueError
        graph = nx.to_numpy_matrix(graph)
        edited_list[i] = graph
    edited_list = np.array(edited_list).reshape(shape_list)
    edit_A = torch.from_numpy(edited_list).float()
    A = A.unsqueeze(-1)
    assert A.shape == edit_A.shape
    return edit_A

def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def preprocess_adj_numpy(adj, symmetric=True):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj_numpy(adj, symmetric)
    return adj


def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor


def preprocess_adj_tensor_with_identity(adj_tensor, symmetric=True):
    # expected torch tensor as adj_tensor!!!
    adj_tensor = adj_tensor.numpy()
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = torch.from_numpy(np.array(adj_out_tensor))
    return adj_out_tensor


def preprocess_adj_tensor_with_identity_concat(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
    return adj_out_tensor

def preprocess_adj_tensor_concat(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
    return adj_out_tensor

def preprocess_edge_adj_tensor(edge_adj_tensor, symmetric=True):
    edge_adj_out_tensor = []
    num_edge_features = int(edge_adj_tensor.shape[1]/edge_adj_tensor.shape[2])

    for i in range(edge_adj_tensor.shape[0]):
        edge_adj = edge_adj_tensor[i]
        edge_adj = np.split(edge_adj, num_edge_features, axis=0)
        edge_adj = np.array(edge_adj)
        edge_adj = preprocess_adj_tensor_concat(edge_adj, symmetric)
        edge_adj_out_tensor.append(edge_adj)

    edge_adj_out_tensor = np.array(edge_adj_out_tensor)
    return edge_adj_out_tensor


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian



def sort_adjacency(g, a, attr):
    node_k1 = dict(g.degree())  ## sort by degree
    node_k2 = nx.average_neighbor_degree(g)  ## sort by neighbor degree
    node_closeness = nx.closeness_centrality(g)
    node_betweenness = nx.betweenness_centrality(g)

    node_sorting = list()

    for node_id in g.nodes():
        node_sorting.append(
            (node_id, node_k1[node_id], node_k2[node_id], node_closeness[node_id], node_betweenness[node_id]))

    node_descending = sorted(node_sorting, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    mapping = dict()

    for i, node in enumerate(node_descending):
        mapping[node[0]] = i

        temp = attr[node[0]]  ## switch node attributes according to sorting
        attr[node[0]] = attr[i]
        attr[i] = temp

    a = nx.adjacency_matrix(g, nodelist=mapping.keys()).todense()  ## switch graph node ids according to sorting

    return g, a, attr


def pad_data(a, attr, max_n_node):
    np.fill_diagonal(a, 1.0)  ## fill the diagonal with fill_diag

    max_a = np.zeros([max_n_node, max_n_node])
    max_a[:a.shape[0], :a.shape[1]] = a
    max_a = np.expand_dims(max_a, axis=2)

    zeroes = np.zeros((max_n_node - attr.shape[0]))
    attr = np.concatenate((attr, zeroes))

    attr = np.expand_dims(attr, axis=1)

    return max_a, attr


def unpad_data(max_a, attr):
    keep = list()
    max_a = np.reshape(max_a, (max_a.shape[0], max_a.shape[1]))

    max_a[max_a > 0.5] = 1.0
    max_a[max_a <= 0.5] = 0.0

    for i in range(0, max_a.shape[0]):
        if max_a[i][i] > 0:
            keep.append(i)

    a = max_a
    a = a[:, keep]  # keep columns
    a = a[keep, :]  # keep rows

    attr = np.reshape(attr, (attr.shape[0]))

    attr = attr[:len(keep)]  ## shorten
    g = nx.from_numpy_matrix(a)

    return g, a, attr


# @title Graph Generation Methods
def generate_graph(n, p):
    g = random_graphs.erdos_renyi_graph(n, p, seed=None, directed=False)
    a = nx.adjacency_matrix(g)

    return g, a


def generate_attr(g, n, p, dataArgs):
    attr, attr_param = None, None
    if dataArgs["node_attr"] == "none":
        attr = np.ones((n)) * 0.5

        attr_param = 0

    if dataArgs["node_attr"] == "random":
        attr = np.random.rand(n)

        attr_param = np.random.rand(1)

    if dataArgs["node_attr"] == "degree":
        attr = np.asarray([int(x[1]) for x in sorted(g.degree())])
        attr = (attr + 1) / (dataArgs["max_n_node"] + 1)

        attr_param = np.random.rand(1)

    if dataArgs["node_attr"] == "uniform":
        uniform_attr = np.random.rand(1)
        attr = np.ones((n)) * uniform_attr

        attr_param = uniform_attr

    if dataArgs["node_attr"] == "p_value":
        attr = np.ones((n)) * p

        attr_param = p

    return g, attr, attr_param


def compute_topol(g):
    density = nx.density(g)

    if nx.is_connected(g):
        diameter = nx.diameter(g)
    else:
        diameter = -1

    cluster_coef = nx.average_clustering(g)

    # if g.number_of_edges() > 2 and len(g) > 2:
    #    assort = nx.degree_assortativity_coefficient(g, x='out', y='in')
    # else:
    #    assort = 0

    edges = g.number_of_edges()
    avg_degree = sum(i for i in nx.degree_centrality(g).values()) / len(nx.degree_centrality(g).keys())

    topol = [density, diameter, cluster_coef, edges, avg_degree]

    return topol


def generate_data(dataArgs):
    A = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], dataArgs["max_n_node"], 1)) ## graph data
    Attr = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], 1)) ## graph data
    Param = np.zeros((dataArgs["n_graph"], 3)) ## generative parameters
    Topol = np.zeros((dataArgs["n_graph"], 5)) ## topological properties

    print("\n============= Generating Data ===========================")
    for i in tqdm(range(0, dataArgs["n_graph"]), leave=True, position=0):

        n = np.random.randint(1, dataArgs["max_n_node"])    ## generate number of nodes n between 1 and max_n_node and
        p = np.random.uniform(dataArgs["p_range"][0], dataArgs["p_range"][1]) ## floating p from range

        g, a = generate_graph(n, p)
        g, attr, attr_param = generate_attr(g, n, p, dataArgs)

        g, a, attr = sort_adjacency(g, a, attr) ## extended BOSAM sorting algorithm
        a, attr = pad_data(a, attr, dataArgs["max_n_node"]) ## pad adjacency matrix to allow less nodes than max_n_node and fill diagonal

        A[i] = a
        Attr[i] = attr
        Param[i] = [n,p,attr_param]
        Topol[i] = compute_topol(g)
    print('done')
    return A, Attr, Param, Topol


def generate_batch(data, batch_size=512):
    total = data.shape[0]
    batched_data = []
    total_batches = data.shape[0] // batch_size if data.shape[0] % batch_size == 0 else data.shape[0] // batch_size + 1
    for i in range(int(total_batches)):
        index_low = i * batch_size
        index_high = None if (i + 1) * batch_size >= total else (i + 1) * batch_size
        batched_data.append(data[index_low:index_high])
    return batched_data

# def train_test_split(all_data, test_proportion = 0.1):
