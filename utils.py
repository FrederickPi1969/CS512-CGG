import networkx as nx
import torch
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
from graph_operations import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from networkx.generators import random_graphs
from discretize import *
from model import *

"""
reshape current matrix to only include existing nodes
"""
def reshapeMatrix(matrixList):
    matrices = []
    single_matrix = False
    if len(np.array(matrices).shape) == 2:
        matrices = [matrices]
        single_matrix = True
    for graphIdx,matrix in enumerate(matrixList):
        number_of_nodes = 0
        for i in range(len(matrix)):
            if matrix[i][i].any() == 1:
                number_of_nodes += 1
            else:
                break
        matrices.append(matrix[:number_of_nodes,:number_of_nodes])
    if single_matrix:
        matrices = matrices[0]
    return np.array(matrices)

"""
add padding to matrix to desire shape
"""
def padMatrix(matrixInput, max_n_node):
    n = len(matrixInput)
    return np.pad(matrixInput, [(0, max_n_node-n), (0, max_n_node-n)], mode='constant', constant_values=0)

def computeNP(A, A_hat):
    batch_size = len(A)
    n, n_hat, p, p_hat = [],[],[],[]
    A = A.numpy().squeeze(-1)
    A_hat = A_hat.numpy().squeeze(-1)
    discretizer = Discretizer(A, A_hat)
    A_hat = discretizer.discretize('hard_threshold')

    for a, a_hat in zip(A, A_hat):
        p_a = 0.0
        n_a = int(np.sum([a[i,i] for i in range(len(a))]))
        if n_a > 1:
            p_a = (np.sum(a[:n_a,:n_a])-n_a)/2
            p_a /= (n_a*(n_a - 1)) / 2

        n_a_hat = int(sum([a_hat[i,i] for i in range(len(a_hat))]))
        p_a_hat = 0.0
        if n_a_hat > 1:
            p_a_hat = (np.sum(a_hat[:n_a_hat,:n_a_hat])-n_a_hat)/2
            p_a_hat /= (n_a_hat*(n_a_hat - 1)) / 2

        # normalize
        n_a /= A.shape[1]
        n_a_hat /= A.shape[1]
        n.append(n_a)
        p.append(p_a)
        n_hat.append(n_a_hat)
        p_hat.append(p_a_hat)

    n = torch.unsqueeze(torch.from_numpy(np.array(n)), -1)
    p = torch.unsqueeze(torch.from_numpy(np.array(p)), -1)
    n_hat = torch.unsqueeze(torch.from_numpy(np.array(n_hat)), -1)
    p_hat = torch.unsqueeze(torch.from_numpy(np.array(p_hat)), -1)
    mse = nn.MSELoss(reduction="mean")
    n_loss = mse(n.flatten(), n_hat.flatten()) * A.shape[1]*A.shape[2]
    p_loss = mse(p.flatten(), p_hat.flatten()) * A.shape[1]*A.shape[2]
    loss = n_loss + p_loss
    return n, n_hat, p, p_hat, loss
    
def graph_mapping(A, dataArgs, modelArgs):
    batch = []
    A = A.detach().numpy()
    if len(A.shape) > 2:
        for graph in A:
            batch.append(graph)
    else:
        batch = [A]
    batch = np.array(batch)
    for i in range(len(batch)):
        graph = batch[i].reshape(dataArgs['max_n_node'], dataArgs['max_n_node'])
        new_graph = np.zeros((dataArgs['max_n_node'], dataArgs['max_n_node']))
        for j,row in enumerate(graph):
            maxval = max(graph[j])
            minval = min(graph[j])
            threshold = (maxval + minval) / 2

            for k in range(j, len(row)):
                colmaxval = max(graph[:][k])
                colminval = min(graph[:][k])
                colthreshold = (colmaxval + colminval) / 2
                inverse_maxval = max(graph[k])
                inverse_minval = min(graph[k])
                inverse_threshold = (inverse_maxval + inverse_minval) / 2
                inverse_colmaxval = max(graph[:][j])
                inverse_colminval = min(graph[:][j])
                inverse_colthreshold = (inverse_colmaxval + inverse_colminval) / 2

                vote = []
                vote.append(graph[j][k] >= threshold)
                vote.append(graph[j][k] >= colthreshold)
                vote.append(graph[k][j] >= inverse_threshold)
                vote.append(graph[k][j] >= inverse_colthreshold)
                if sum(vote) >= 2:
                    new_graph[j][k] = 1
                    new_graph[k][j] = 1
                else:
                    new_graph[j][k] = 0
                    new_graph[k][j] = 0

        graph = new_graph

        real_num_of_node = int(dataArgs['max_n_node'])
        for row in range(dataArgs['max_n_node']):
            if graph[row][row] != 1:
                real_num_of_node = row
                break
        valid_subgraph = graph[:real_num_of_node, :real_num_of_node]
        graph = np.zeros((dataArgs['max_n_node'], dataArgs['max_n_node']))
        graph[:real_num_of_node, :real_num_of_node] = valid_subgraph
        graph = graph.reshape((dataArgs['max_n_node'], dataArgs['max_n_node'], 1))
        batch[i] = graph
    batch = np.array(batch)
    batch = torch.from_numpy(batch)
    # assert batch.shape == A.shape
    return batch

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
    else:
        graph = A.numpy().reshape(dataArgs['max_n_node'], dataArgs['max_n_node'])
        edited_list = [nx.from_numpy_matrix(graph)]

    for i,graph in enumerate(edited_list):
        if modelArgs['edit_method'] == 'densify':
            increase_density = 0.1
            if args.get('increase_density') != None:
                increase_density = args.get('increase_density')
            densify(graph, increase_density)
        elif modelArgs['edit_method'] == 'sparsify':
            decrease_density = 0.1
            if args.get('decrease_density') != None:
                decrease_density = args.get('decrease_density')
            sparsify(graph, decrease_density)
        elif modelArgs['edit_method'] == 'self_multiply':
            n = 1
            if args.get('n') != None:
                n = args.get('n')
            self_multiply(graph, n)
        elif modelArgs['edit_method'] == 'self_repetition':
            n = 1
            linknode = 0
            if args.get('n') != None:
                n = args.get('n')
            if args.get('linknode') != None:
                linknode = args.get('linknode')
            self_multiply(graph, n, linknode)
        elif modelArgs['edit_method'] == 'densify_to':
            target_density = 1
            if args.get('target_density') != None:
                target_density = args.get('target_density')
            densify_to(graph, target_density)
        elif modelArgs['edit_method'] == 'sparsify_to':
            target_density = 0
            if args.get('target_density') != None:
                target_density = args.get('target_density')
            sparsify_to(graph, target_density)
        elif modelArgs['edit_method'] == 'add_edge_coherent':
            n = 1
            descending = True
            if args.get('n') != None:
                n = args.get('n')
            if args.get('descending') != None:
                descending = args.get('descending')
            add_edge_coherent(graph, n, descending)
        elif modelArgs['edit_method'] == 'remove_edge_coherent':
            n = 1
            descending = False
            if args.get('n') != None:
                n = args.get('n')
            if args.get('descending') != None:
                descending = args.get('descending')
            remove_edge_coherent(graph, n, descending)
        elif modelArgs['edit_method'] == 'remove_edge_difference':
            n = 1
            if args.get('n') != None:
                n = args.get('n')
            remove_edge_difference(graph, n)
        elif modelArgs['edit_method'] == 'add_node':
            namelist = None
            n = 1
            m = 1
            descending = True
            if args.get('n') != None:
                n = args.get('n')
            if args.get('m') != None:
                m = args.get('m')
            if args.get('descending') != None:
                descending = args.get('descending')
            if args.get('namelist') != None:
                namelist = args.get('namelist')
            add_node(graph, namelist, n, m, descending)
        elif modelArgs['edit_method'] == 'remove_node':
            n = 1
            descending = False
            if args.get('n') != None:
                n = args.get('n')
            if args.get('descending') != None:
                descending = args.get('descending')
            remove_node(graph, n, descending)
        else:
            raise ValueError
        graph = nx.to_numpy_matrix(graph)
        edited_list[i] = graph
    edited_list = np.array(edited_list).reshape(shape_list)
    edit_A = torch.from_numpy(edited_list).float()
    A = A.unsqueeze(-1)
    # assert A.shape == edit_A.shape
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

def pad_data_v2(a, attr, max_n_node):
    np.fill_diagonal(a, 1.0)  ## fill the diagonal with fill_diag

    max_a = np.zeros([max_n_node, max_n_node])
    max_a[:a.shape[0], :a.shape[1]] = a
    max_a = np.expand_dims(max_a, axis=2)

    max_attr = np.zeros([max_n_node, max_n_node])
    max_attr[:attr.shape[0], :attr.shape[1]] = attr
    return max_a, max_attr


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

def unpad_data_v2(max_a, attr):
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

    sum = np.sum(attr, dim=1)
    keep = list(sum).index(0)
    attr = attr[:keep, :keep]  ## shorten
    g = nx.from_numpy_matrix(a)

    return g, a, attr



# @title Graph Generation Methods
def generate_graph(n, p):
    g = random_graphs.erdos_renyi_graph(n, p, seed=None, directed=False)
    a = nx.adjacency_matrix(g)

    return g, a

def generate_attr_v2(g, n, p, dataArgs):
    # We take as default that embedding dimension is same as node_num n!!!!!!
    # Notice we don't allow self loop so each node has degree range [0, n-1], which can be directly treated as attr index. We do one-hot on degree when using "degree" mode.
    attr, attr_param = None, None
    if dataArgs["node_attr"] == "none":
        attr = np.ones((n)) * 0.5

        attr_param = 0

    if dataArgs["node_attr"] == "random":
        attr = np.random.rand(n, n)
        attr_param = np.random.rand(1)

    if dataArgs["node_attr"] == "degree":
        a = nx.adjacency_matrix(g).toarray()
        idx = np.asarray([int(x[0]) for x in sorted(g.degree())])
        degree = np.asarray([int(x[1]) for x in sorted(g.degree())])

        attr = np.zeros([n, n])
        attr[idx, degree] = 1

        attr_param = np.random.rand(1)

    if dataArgs["node_attr"] == "uniform":
        uniform_attr = np.random.rand(1)
        attr = torch.empty([n,n]).uniform_().numpy()
        attr_param = uniform_attr


    return g, attr, attr_param


def generate_attr(g, n, p, dataArgs):
    attr, attr_param = None, None
    if dataArgs["node_attr"] == "none":
        attr = np.ones((n)) * 0.5
        attr_param = 0

    if dataArgs["node_attr"] == "random":
        attr = np.random.rand(n,n)
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
    Attr = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], 1)) ## graph data, shape: (n_graph, n_node, 1)
    Param = np.zeros((dataArgs["n_graph"], 3)) ## generative parameters
    Topol = np.zeros((dataArgs["n_graph"], 5)) ## topological properties

    print("\n============= Generating Data ===========================")
    for i in tqdm(range(0, dataArgs["n_graph"]), leave=True, position=0):

        n = np.random.randint(4, dataArgs["max_n_node"])    ## generate number of nodes n between 1 and max_n_node and
        p = np.random.uniform(dataArgs["p_range"][0], dataArgs["p_range"][1]) ## floating p from range

        g, a = generate_graph(n, p)
        g, attr, attr_param = generate_attr(g, n, p, dataArgs)

        g, a, attr = sort_adjacency(g, a, attr) ## extended BOSAM sorting algorithm
        a, attr = pad_data(a, attr, dataArgs["max_n_node"]) ## pad adjacency matrix to allow less nodes than max_n_node and fill diagonal

        if dataArgs["upper_triangular"]:
            A[i] = np.triu(a.reshape(dataArgs["max_n_node"], dataArgs["max_n_node"])).reshape(dataArgs["max_n_node"], dataArgs["max_n_node"], 1)
        else:
            A[i] = a
        Attr[i] = attr
        Param[i] = [n,p,attr_param]
        Topol[i] = compute_topol(g)
    print('done')

    return A, Attr, Param, Topol


def generate_data_v2(dataArgs):
    A = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], dataArgs["max_n_node"], 1)) ## graph data
    Attr = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], dataArgs["max_n_node"])) ## graph data, shape: (n_graph, n_node, n_node)
    Param = np.zeros((dataArgs["n_graph"], 3)) ## generative parameters
    Topol = np.zeros((dataArgs["n_graph"], 5)) ## topological properties

    print("\n============= Generating Data ===========================")
    for i in tqdm(range(0, dataArgs["n_graph"]), leave=True, position=0):

        n = np.random.randint(4, dataArgs["max_n_node"])    ## generate number of nodes n between 1 and max_n_node and
        p = np.random.uniform(dataArgs["p_range"][0], dataArgs["p_range"][1]) ## floating p from range

        g, a = generate_graph(n, p)
        g, attr, attr_param = generate_attr_v2(g, n, p, dataArgs)

        g, a, attr = sort_adjacency(g, a, attr) ## extended BOSAM sorting algorithm
        a, attr = pad_data_v2(a, attr, dataArgs["max_n_node"]) ## pad adjacency matrix to allow less nodes than max_n_node and fill diagonal

        if dataArgs["upper_triangular"]:
            A[i] = np.triu(a.reshape(dataArgs["max_n_node"], dataArgs["max_n_node"])).reshape(dataArgs["max_n_node"], dataArgs["max_n_node"], 1)
        else:
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

## returns graphs and graph dimensions 
def torch_tensor_to_graphs(tensors, graph_node_counts):
    graphs_adj_matrices = list(np.squeeze(tensors.numpy()))
    #print([np.sum(np.diag(g)) - h for g, h in zip(graphs_adj_matrices, graph_node_counts)])
    graphs_adj_matrices = [g[0:h, 0:h] for g, h in zip(graphs_adj_matrices, graph_node_counts)]
    for graph in graphs_adj_matrices:
        graph = np.fill_diagonal(graph, 0)
    return [nx.convert_matrix.from_numpy_matrix(g) for g in graphs_adj_matrices]

def graphs_to_torch_tensor(graphs, original_size):
    edited_adj_matrices = [nx.adjacency_matrix(g).todense() for g in graphs]
    for matrix in edited_adj_matrices:
        matrix = np.fill_diagonal(matrix, 1)
    edited_adj_matrices = [padMatrix(g, original_size) for g in edited_adj_matrices]
    #print(edited_adj_matrices)
    return torch.unsqueeze(torch.from_numpy(np.asarray(edited_adj_matrices).astype(float)), -1)
    

# def generate_dblp_data(dataArgs, load = False, save = True):
#     if load:
#         return pickle.load(open("dblp_subgraphs", 'rb'))
#
#     dblp_graph = dblp()
#
#     A = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], dataArgs["max_n_node"], 1)) ## graph data
#     Attr = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], 1)) ## graph data
#     Param = np.zeros((dataArgs["n_graph"], 3)) ## generative parameters
#     Topol = np.zeros((dataArgs["n_graph"], 5)) ## topological properties
#
#     print("\n============= Generating Data ===========================")
#     for i in tqdm(range(0, dataArgs["n_graph"]), leave=True, position=0):
#
#         n = np.random.randint(1, dataArgs["max_n_node"])    ## generate number of nodes n between 1 and max_n_node and
#
#         g, a = sample_subgraph(dblp_graph, target_size = n, max_size = dataArgs["max_n_node"], start = None)
#         n = len(g)
#         p = nx.transitivity(g)
#         g, attr, attr_param = generate_attr(g, n, p, dataArgs)
#
#         g, a, attr = sort_adjacency(g, a, attr) ## extended BOSAM sorting algorithm
#         a, attr = pad_data(a, attr, dataArgs["max_n_node"]) ## pad adjacency matrix to allow less nodes than max_n_node and fill diagonal
#
#         A[i] = a
#         Attr[i] = attr
#         Param[i] = [n,p,attr_param]
#         Topol[i] = compute_topol(g)
#
#     print('done')
#     if save:
#         pickle.dump((A, Attr, Param, Topol), open("dblp_subgraphs", 'wb'))
#     return A, Attr, Param, Topol


def generate_batched_mask(batched_A_hat, batched_param):
    batch_size, max_node, _ = batched_A_hat.shape
    mask = torch.zeros(batch_size, max_node, max_node)
    for i in range(batch_size):
        # print(batched_param[i][-1])
        n = int(batched_param[i][-1])  # dim -1 is the predicted node num
        mask[i][:n,:n] = torch.ones([n, n])
    return mask  # -> (b, n, n)

def masked_normalization(batched_A_hat, batched_param):
    """
    Input a RAW score matrix A_hat -> (b, n, n), batched_param (n, p, param) tuple.
    """
    batch_size, max_node, _ = batched_A_hat.shape
    batched_mask = generate_batched_mask(batched_A_hat, batched_param)
    temp_min = batched_A_hat.masked_fill(1 - batched_mask, 1e9)  #first change all entries without nodes to 1e9 to get the global min size: (b, n, n)
    temp_max = batched_A_hat.masked_fill(1 - batched_mask, 1e-9)  #change all entries without nodes to 1e-9 to extract the global max
    global_min = torch.min(temp_min.view(batch_size, -1), dim=-1)[0].view(-1,1,1) # the minimum in each (n,n) maxtrix in batch
    global_max = torch.max(temp_max.view(batch_size, -1), dim=-1)[0].view(-1,1,1)

    normalized = ((batched_A_hat - global_min) / global_max) * batched_mask ## need adjustment to get the scores distributed around 0.5
    return normalized # this need discretization
