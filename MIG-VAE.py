
# What GNN cannot learn [iclr20]
#  https://arxiv.org/abs/1911.08795
#  https://openreview.net/forum?id=B1l2bp4YwS 
#  fspool [iclr20] 

#@title Libraries and Support Functions
## Basic
from tqdm import tqdm
import argparse
import os
import random
import itertools
import sys
import math

## Computation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.stats.stats import pearsonr 
from scipy.stats import norm

## Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

## Network Processing
import networkx as nx
from networkx.generators import random_graphs


def sort_adjacency(g, a, attr):
	
	node_k1 = dict(g.degree())    ## sort by degree
	node_k2 = nx.average_neighbor_degree(g)    ## sort by neighbor degree
	node_closeness = nx.closeness_centrality(g)
	node_betweenness = nx.betweenness_centrality(g)

	node_sorting = list()

	for node_id in g.nodes():
		node_sorting.append((node_id, node_k1[node_id], node_k2[node_id], node_closeness[node_id], node_betweenness[node_id]))
		
	node_descending = sorted(node_sorting, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
	mapping = dict()

	for i, node in enumerate(node_descending):
		mapping[node[0]] = i
		
		temp = attr[node[0]]     ## switch node attributes according to sorting
		attr[node[0]] = attr[i]
		attr[i] = temp
		
	a = nx.adjacency_matrix(g, nodelist=mapping.keys()).todense()    ## switch graph node ids according to sorting
	
	return g, a, attr

	
	
	
def pad_data(a, attr, max_n_node):

	np.fill_diagonal(a, 1.0)    ## fill the diagonal with fill_diag

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

	## delete rows and columns
	a = max_a
	a = a[:, keep]    # keep columns
	a = a[keep, :]    # keep rows         
	
	attr = np.reshape(attr, (attr.shape[0]))
	
	attr = attr[:len(keep)]    ## shorten
	g = nx.from_numpy_matrix(a)

	return g, a, attr
	
	
	
	
def plot_graph(g, a, attr, draw):
	
	orig_cmap = plt.cm.PuBu
	fixed_cmap = shiftedColorMap(orig_cmap, start=min(attr), midpoint=0.5, stop=max(attr), name='fixed')
	
	## adjust colour reconstructed_a_padded according to features
	a = np.reshape(a, (a.shape[0], a.shape[1]))
	a_channel = np.copy(a)
	a_channel = np.tile(a_channel[:, :, None], [1, 1, 3])    ## broadcast 1 channel to 3

	for node in range(0, len(g)):
		color = fixed_cmap(attr[node])[:3]
		a_channel[node, :node + 1] = a_channel[node, :node + 1] * color
		a_channel[:node, node] = a_channel[:node, node] * color
		
	if draw == True:
		fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
		#plt.axis('off')
		plt.sca(axes[0])
		nx.draw_kamada_kawai(g, node_color=attr, font_color='white', cmap = fixed_cmap)
		axes[1].set_axis_off()
		axes[1].imshow(a_channel)
		fig.tight_layout()
	
	return fixed_cmap, a_channel
	
				
	
	
	
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	
	cdict = {'red': [],'green': [],'blue': [],'alpha': []}

	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)

	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint=False),
		np.linspace(midpoint, 1.0, 129, endpoint=True)])

	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)

		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)

	return newcmap
	


# # _1 Generate Graph Data

#@title Graph Generation Methods
def generate_graph(n,p): 

	g = random_graphs.barabasi_albert_graph(n, p, seed=None) 
	a = nx.adjacency_matrix(g)
	
	return g, a

	

def compute_topol(g):
	
	density = nx.density(g)

	if nx.is_connected(g):
		diameter = nx.diameter(g)
	else:
		diameter = -1

	cluster_coef = nx.average_clustering(g)

	#if g.number_of_edges() > 2 and len(g) > 2:
	#        assort = nx.degree_assortativity_coefficient(g, x='out', y='in')
	#else:
	#        assort = 0

	edges = g.number_of_edges()
	avg_degree = sum(i for i in nx.degree_centrality(g).values()) / len(nx.degree_centrality(g).keys())
	
	topol = [density, diameter, cluster_coef, edges, avg_degree]
	
	return topol

	
	
	
def generate_attr(g, n, p, dataArgs):
	
		if dataArgs["node_attr"] == "none":
			attr = np.ones((n)) * 0.5
			
			attr_param = 0
			
	
		if dataArgs["node_attr"] == "random":                
			attr = np.random.rand(n)
			
			attr_param = np.random.rand(1)
			

		if dataArgs["node_attr"] == "degree":
			attr = np.asarray([int(x[1]) for x in sorted(g.degree())])    
			attr = (attr+1) / (dataArgs["max_n_node"]+1)
			
			attr_param = np.random.rand(1)
			

		if dataArgs["node_attr"] == "uniform":
			uniform_attr = np.random.rand(1)
			attr = np.ones((n)) * uniform_attr
			
			attr_param = uniform_attr


		if dataArgs["node_attr"] == "p_value":    
			attr = np.ones((n)) * p
			
			attr_param = p
				
		return g, attr, attr_param

				


def generate_data(dataArgs): 

	A = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], dataArgs["max_n_node"], 1)) ## graph data
	Attr = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], 1)) ## graph data
	Param = np.zeros((dataArgs["n_graph"], 3)) ## generative parameters
	Topol = np.zeros((dataArgs["n_graph"], 5)) ## topological properties
	

	# for i in tqdm(range(0,dataArgs["n_graph"])):
	for i in range(0,dataArgs["n_graph"]):
		
		n = random.randint(2, dataArgs["max_n_node"])        ## generate number of nodes n between 1 and max_n_node and
		p = random.randint(1, n - 1) ## floating p from range
		
		g, a = generate_graph(n, p)
		g, attr, attr_param = generate_attr(g, n, p, dataArgs)
		
		g, a, attr = sort_adjacency(g, a, attr) ## extended BOSAM sorting algorithm
		a, attr = pad_data(a, attr, dataArgs["max_n_node"]) ## pad adjacency matrix to allow less nodes than max_n_node and fill diagonal
		
		
		A[i] = a
		Attr[i] = attr
		Param[i] = [n,p,attr_param]
		Topol[i] = compute_topol(g)
		
	return A, Attr, Param, Topol


#@title    ER Random Graph Data Specifications

dataArgs = dict()

#@markdown select the maximum number of nodes per graph 

maximum_number_of_nodes_n = "24" #@param [12, 24, 30, 48]
dataArgs["max_n_node"] = int(maximum_number_of_nodes_n)

#@markdown select the range of p

range_of_linkage_probability_p = "0,1" #@param [[0.0,1.0], [0.2,0.8], [0.5,0.5]]
dataArgs["p_range"] = [float(range_of_linkage_probability_p.split(",")[0]), float(range_of_linkage_probability_p.split(",")[1])]


#@markdown specify the generation process of node attributes

node_attributes = "uniform" #@param ["none", "uniform", "degree", "p_value", "random"]
dataArgs["node_attr"] = node_attributes


#@markdown specify the number of graphs generated for training and validation
# number_of_graph_instances = "10000" #@param [1, 100, 1000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000]
number_of_graph_instances = "1000" #@param [1, 100, 1000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000]
dataArgs["n_graph"] = int(number_of_graph_instances) 
		
A, Attr, Param, Topol = generate_data(dataArgs)

g, a, attr = unpad_data(A[0], Attr[0])
# fixed_cmap, a_channel = plot_graph(g, A[0], attr, draw = True)


# 

# # _2 Train Model

#@title Model Support Functions
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

## Keras
# %tensorflow_version 1.4.0
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Dropout, Activation, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine.topology import Layer

import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
	return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
	"""Load citation network dataset (cora only for now)"""
	print('Loading {} dataset...'.format(dataset))

	idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
	features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
	labels = encode_onehot(idx_features_labels[:, -1])

	# build graph
	idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
	idx_map = {j: i for i, j in enumerate(idx)}
	edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
	edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
					 dtype=np.int32).reshape(edges_unordered.shape)
	adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
						shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

	# build symmetric adjacency matrix
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

	# features = normalize_features(features)
	# adj = normalize_adj(adj + sp.eye(adj.shape[0]))

	print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

	return features.todense(), adj, labels


def load_data_attention(path="data/cora/", dataset="cora"):
	"""Load citation network dataset (cora only for now)"""
	print('Loading {} dataset...'.format(dataset))

	idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
	features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
	labels = encode_onehot(idx_features_labels[:, -1])

	# build graph
	idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
	idx_map = {j: i for i, j in enumerate(idx)}
	edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
	edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
					 dtype=np.int32).reshape(edges_unordered.shape)
	adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
						shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

	# build symmetric adjacency matrix
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

	features = normalize_features(features)
	adj = normalize_adj(adj + sp.eye(adj.shape[0]))

	print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

	return features.todense(), adj, labels


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


def normalize_adj_numpy(adj, symmetric=True):
	if symmetric:
		d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
		a_norm = adj.dot(d).transpose().dot(d)
	else:
		d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
		a_norm = d.dot(adj)
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
	adj_out_tensor = []
	for i in range(adj_tensor.shape[0]):
		adj = adj_tensor[i]
		adj = adj + np.eye(adj.shape[0])
		adj = normalize_adj_numpy(adj, symmetric)
		adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
		adj_out_tensor.append(adj)
	adj_out_tensor = np.array(adj_out_tensor)
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


def sample_mask(idx, l):
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)


def get_splits(y):
	idx_train = range(140)
	idx_val = range(200, 500)
	idx_test = range(500, 1500)
	y_train = np.zeros(y.shape, dtype=np.int32)
	y_val = np.zeros(y.shape, dtype=np.int32)
	y_test = np.zeros(y.shape, dtype=np.int32)
	y_train[idx_train] = y[idx_train]
	y_val[idx_val] = y[idx_val]
	y_test[idx_test] = y[idx_test]
	train_mask = sample_mask(idx_train, y.shape[0])
	return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def get_splits_v2(y):
	idx_train = range(1708)
	idx_val = range(1708, 1708 + 500)
	idx_test = range(1708 + 500, 2708)
	y_train = np.zeros(y.shape, dtype=np.int32)
	y_val = np.zeros(y.shape, dtype=np.int32)
	y_test = np.zeros(y.shape, dtype=np.int32)
	y_train[idx_train] = y[idx_train]
	y_val[idx_val] = y[idx_val]
	y_test[idx_test] = y[idx_test]
	train_mask = sample_mask(idx_train, y.shape[0])
	return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


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


def rescale_laplacian(laplacian):
	try:
		print('Calculating largest eigenvalue of normalized graph Laplacian...')
		largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
	except ArpackNoConvergence:
		print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
		largest_eigval = 2

	scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
	return scaled_laplacian


def chebyshev_polynomial(X, k):
	"""Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
	print("Calculating Chebyshev polynomials up to order {}...".format(k))

	T_k = list()
	T_k.append(sp.eye(X.shape[0]).tocsr())
	T_k.append(X)

	def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
		X_ = sp.csr_matrix(X, copy=True)
		return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

	for i in range(2, k + 1):
		T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

	return T_k


def sparse_to_tuple(sparse_mx):
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = shapeparse_mx.shape
	return coords, values, shape
	
	
	
	
	
def graph_conv_op(x, num_filters, graph_conv_filters, kernel):

	if len(x.get_shape()) == 2:
		conv_op = K.dot(graph_conv_filters, x)
		conv_op = tf.split(conv_op, num_filters, axis=0)
		conv_op = K.concatenate(conv_op, axis=1)
	elif len(x.get_shape()) == 3:
		conv_op = K.batch_dot(graph_conv_filters, x)
		conv_op = tf.split(conv_op, num_filters, axis=1)
		conv_op = K.concatenate(conv_op, axis=2)
	else:
		raise ValueError('x must be either 2 or 3 dimension tensor'
						 'Got input shape: ' + str(x.get_shape()))

	conv_out = K.dot(conv_op, kernel)
	return conv_out



class GCN(Layer):

	def __init__(self,
				 output_dim,
				 num_filters,
				 activation=None,
				 use_bias=True,
				 kernel_initializer='glorot_uniform',
				 bias_initializer='zeros',
				 kernel_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 bias_constraint=None,
				 **kwargs):
		super(GCN, self).__init__(**kwargs)

		self.output_dim = output_dim
		self.num_filters = num_filters

		self.activation = activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.kernel_initializer.__name__ = kernel_initializer
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)

	def build(self, input_shape):

		if self.num_filters != int(input_shape[1][-2]/input_shape[1][-1]):
			raise ValueError('num_filters does not match with graph_conv_filters dimensions.')

		self.input_dim = input_shape[0][-1]
		kernel_shape = (self.num_filters * self.input_dim, self.output_dim)

		self.kernel = self.add_weight(shape=kernel_shape,
										initializer=self.kernel_initializer,
										name='kernel',
										regularizer=self.kernel_regularizer,
										constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.output_dim,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None

		self.built = True

	def call(self, inputs):

		output = graph_conv_op(inputs[0], self.num_filters, inputs[1], self.kernel)
		if self.use_bias:
			output = K.bias_add(output, self.bias)
		if self.activation is not None:
			output = self.activation(output)
		return output

	def compute_output_shape(self, input_shape):
		output_shape = (input_shape[0][0], input_shape[0][1], self.output_dim)
		return output_shape

	def get_config(self):
		config = {
			'output_dim': self.output_dim,
			'num_filters': self.num_filters,
			'activation': activations.serialize(self.activation),
			'use_bias': self.use_bias,
			'kernel_initializer': initializers.serialize(self.kernel_initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint)
		}
		base_config = super(GCN, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

#@title    Model Specifications

#@markdown model build specifications

modelArgs = {"gnn_filters": 2, "conv_filters": 16, "kernel_size": 3}

number_of_latent_variables= "3" #@param [1, 2, 3, 4, 5]
modelArgs["latent_dim"] = int(number_of_latent_variables)

#@markdown training specifications

trainArgs = dict()

weight_graph_reconstruction_loss = "5" #@param [0, 1, 2, 3, 5, 10, 20]
weight_attribute_reconstruction_loss = "2" #@param [0, 1, 2, 3, 5, 10, 20]
beta_value = "10" #@param [0, 1, 2, 3, 5, 10, 20]
trainArgs["loss_weights"] = [int(weight_graph_reconstruction_loss), int(weight_attribute_reconstruction_loss), int(beta_value)]

epochs = "20" #@param [10, 20, 50]
trainArgs["epochs"] = int(epochs)
batch_size = "1024" #@param [2, 4, 8, 16, 32, 128, 512, 1024]
trainArgs["batch_size"] = int(batch_size)
early_stop = "10" #@param [1, 2, 3, 4, 10]
trainArgs["early_stop"] = int(early_stop)
train_test_split = "0.1" #@param [0.1, 0.2, 0.3, 0.5]
trainArgs["data_split"] = float(train_test_split)


## Train and Test Split _______________________________________________

A_train = A[:int((1-trainArgs["data_split"])*A.shape[0])]
Attr_train = Attr[:int((1-trainArgs["data_split"])*Attr.shape[0])]
Param_train = Param[:int((1-trainArgs["data_split"])*Param.shape[0])]
Topol_train = Topol[:int((1-trainArgs["data_split"])*Topol.shape[0])]

A_test = A[int((1-trainArgs["data_split"])*A.shape[0]):]
Attr_test = Attr[int((1-trainArgs["data_split"])*Attr.shape[0]):]
Param_test = Param[int((1-trainArgs["data_split"])*Param.shape[0]):]
Topol_test = Topol[int((1-trainArgs["data_split"])*Topol.shape[0]):]

## build graph_conv_filters

SYM_NORM = True
A_train_mod = preprocess_adj_tensor_with_identity(np.squeeze(A_train), SYM_NORM)
A_test_mod = preprocess_adj_tensor_with_identity(np.squeeze(A_test), SYM_NORM)

train_data = (Attr_train, A_train_mod, Param_train, Topol_train)
test_data = (Attr_test, A_test_mod, Param_test, Topol_test)

modelArgs["input_shape"], modelArgs["output_shape"] = ((Attr_train.shape[1], 1), (A_train.shape[1], A_train.shape[2], 1)), ((Attr_test.shape[1], 1), (A_test.shape[1], A_test.shape[2], 1))

#@title Build and Train Model
class VAE():

	# reparameterization trick
	# instead of sampling from Q(z|X), sample eps = N(0,I)
	# then z = z_mean + sqrt(var)*eps

	def sampling(self, args):
		"""Reparameterization trick by sampling fr an isotropic unit Gaussian.
		# Arguments
			args (tensor): mean and log of variance of Q(z|X)
		# Returns
			z (tensor): sampled latent vector
		"""

		z_mean, z_log_var = args
		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]
		epsilon = K.random_normal(shape=(batch, dim))
		return z_mean + K.exp(0.5 * z_log_var) * epsilon

	
	
	def __init__(self, modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test):

		## MODEL ______________________________________________________________             
			
		## Graph Neural Network Architecture __________________________________
			
		## 1) build encoder model____________________________________

		# build graph_conv_filters
		SYM_NORM = True         # inv(sqrt(D)) dot A
		num_filters = modelArgs['gnn_filters']
		graph_conv_filters = preprocess_adj_tensor_with_identity(np.squeeze(A_train), SYM_NORM)

		# build model
		X_input = Input(shape=(Attr_train.shape[1], Attr_train.shape[2]), name = "node_attributes")
		graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]), name = "adjacency_matrix")
		
		
		# define inputs of features and graph topologies
		inputs = [X_input, graph_conv_filters_input]

		x = GCN(100, num_filters, activation='elu')([X_input, graph_conv_filters_input])
		x = Dropout(0.1)(x)
		x = GCN(100, num_filters, activation='elu')([x, graph_conv_filters_input])
		x = Dropout(0.1)(x)
		x = Lambda(lambda x: K.mean(x, axis=1))(x)    # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
		x = Dense(8, activation='relu')(x)
		x = Dense(6, activation='relu')(x)

		
		z_mean = Dense(modelArgs["latent_dim"], name='z_mean')(x)
		z_log_var = Dense(modelArgs["latent_dim"], name='z_log_var')(x)

		# use reparameterization trick to push the sampling out as input
		# note that "output_shape" isn't necessary with the TensorFlow backend
		z = Lambda(self.sampling, output_shape=(modelArgs["latent_dim"],), name='z')([z_mean, z_log_var])
		
		latent_inputs = Input(shape=(modelArgs["latent_dim"],), name='z_sampling')

		
		## 2.1) build attribute decoder model __________________________

		y = Dense(4, activation='relu')(latent_inputs)
		y = Dense(6, activation='relu')(y)
		y = Dense(10, activation='relu')(y)
		y = Dense(modelArgs["output_shape"][0][0], activation='sigmoid')(y)
		attr_output = Reshape(modelArgs["output_shape"][0], name='node_attributes')(y)
		
		
		
		## 2.2) build adjacency decoder model __________________________

		## shape info needed to build decoder model                
		x_2D = Input(shape=modelArgs["input_shape"][1], name='adjacency_decoder')
		
		for i in range(2):
			modelArgs['conv_filters'] *= 2
			x_2D = Conv2D(filters=modelArgs['conv_filters'], kernel_size=modelArgs['kernel_size'], activation='relu',strides=2, padding='same')(x_2D)
		shape_2D = K.int_shape(x_2D)

		x_2D = Dense(shape_2D[1] * shape_2D[2] * shape_2D[3], activation='relu')(latent_inputs)
		x_2D = Reshape((shape_2D[1], shape_2D[2], shape_2D[3]))(x_2D)

		for i in range(2):
			x_2D = Conv2DTranspose(filters=modelArgs['conv_filters'], kernel_size=modelArgs['kernel_size'],activation='relu', strides=2, padding='same')(x_2D)
			modelArgs['conv_filters'] //= 2

		a_output = Conv2DTranspose(filters=1, kernel_size=modelArgs['kernel_size'], activation='sigmoid',padding='same', name='adjacency_matrix')(x_2D)

	
	
		## INSTANTIATE___________________________________

		## 1) instantiate encoder model
		encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
		#encoder.summary()

		## 2) instantiate decoder model
		decoder = Model(latent_inputs, [attr_output, a_output], name='reconstruction')
		#decoder.summary()
		
		## 3) instantiate VAE model
		attr_a_outputs = decoder(encoder(inputs)[2])
		vae = Model(inputs, attr_a_outputs, name='vae')


		## LOSS FUNCTIONS ______________________________________
		
		def loss_func(y_true, y_pred):
			y_true_attr = y_true[0]
			y_pred_attr = y_pred[0]

			y_true_a = y_true[1]
			y_pred_a = y_pred[1]

			## ATTR RECONSTRUCTION LOSS_______________________
			## mean squared error
			attr_reconstruction_loss = mse(K.flatten(y_true_attr), K.flatten(y_pred_attr))
			attr_reconstruction_loss *= modelArgs["input_shape"][0][0]

			## A RECONSTRUCTION LOSS_______________________
			## binary cross-entropy
			a_reconstruction_loss = binary_crossentropy(K.flatten(y_true_a), K.flatten(y_pred_a))
			a_reconstruction_loss *= (modelArgs["input_shape"][1][0] * modelArgs["input_shape"][1][1])

			## KL LOSS _____________________________________________
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5

			## COMPLETE LOSS __________________________________________________
			# attr_reconstruction_loss = tf.Print(attr_reconstruction_loss, [attr_reconstruction_loss], message="attr_reconstruction_loss: ")
			# a_reconstruction_loss = tf.Print(a_reconstruction_loss, [a_reconstruction_loss], message="a_reconstruction_loss: ")
			# kl_loss = tf.Print(kl_loss, [kl_loss], message="kl_loss: ")
			# tf.Print('weight_attribute_reconstruction_loss:', attr_reconstruction_loss, 'a_reconstruction_loss:', a_reconstruction_loss, 'kl_loss:', kl_loss)

			# loss = K.mean(trainArgs["loss_weights"][0] * a_reconstruction_loss + trainArgs["loss_weights"][1] * attr_reconstruction_loss + trainArgs["loss_weights"][2] * kl_loss)
			loss = trainArgs["loss_weights"][0] * a_reconstruction_loss

			return loss
		
		
	
		## MODEL COMPILE______________________________________________
		
		vae.compile(optimizer='adam', loss=loss_func)
		#vae.summary()
		
		 
		## TRAIN______________________________________________

		# Set callback functions to early stop training and save the best model so far
		callbacks = [EarlyStopping(monitor='val_loss', patience=trainArgs["early_stop"])]

		vae.fit([Attr_train, A_train_mod], [Attr_train, A_train], epochs=trainArgs["epochs"], batch_size=trainArgs["batch_size"], callbacks=callbacks, validation_data=([Attr_test, A_test_mod], [Attr_test, A_test]))

		self.model = (encoder, decoder)
	
	
if dataArgs["node_attr"] == "none":
	trainArgs["loss_weights"] = [int(weight_graph_reconstruction_loss), int(0), int(beta_value)]
		
vae = VAE(modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test)
model = vae.model 


# # _3 Latent Space Analysis using the Decoder


#@title Decoder Analysis Support Functions


## DECODER - Latent Space Interpolation____________________________

def generate_manifold(analyzeArgs, modelArgs, trainArgs, model, data):

	Attr, A_mod, Param, Topol = data

	encoder, decoder = model    ## trained model parts
	z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
	z_var = K.exp(0.5 * z_log_var)

			 
	if len(analyzeArgs["z"]) >= 2:
		
		z_sample = np.zeros(modelArgs["latent_dim"])
		z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

		## fill unobserved dimensions with mean of latent variable dimension
		for dim in range(0, len(z_sample[0])):
			z_sample[0][dim] = np.mean(z_mean[:, dim])

		grid_x = np.linspace(analyzeArgs["range"][0], analyzeArgs["range"][1], analyzeArgs["size_of_manifold"])
		grid_y = np.linspace(analyzeArgs["range"][0], analyzeArgs["range"][1], analyzeArgs["size_of_manifold"])[::-1]    ## revert

		figure = np.zeros((analyzeArgs["size_of_manifold"] * dataArgs["max_n_node"], analyzeArgs["size_of_manifold"] * dataArgs["max_n_node"], 3))
		fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(8, 8))

		## Set common labels
		fig.text(0.5, 0.04, "z_" + str(analyzeArgs["z"][0]), ha='center')
		fig.text(0.04, 0.5, "z_" + str(analyzeArgs["z"][1]), va='center', rotation='vertical')

		for i, yi in enumerate(grid_y):
			for j, xi in enumerate(grid_x):

				xi_value = xi ** 1
				yi_value = yi ** 1

				try: 
					z_sample[0][analyzeArgs["z"][0]] = xi ** 1
					z_sample[0][analyzeArgs["z"][1]] = yi ** 1
				except:
					print("please select correct latent variables")
					print("number of latent variables to choose from: z_" + str(np.arange(modelArgs["latent_dim"])))
					sys.exit()
					
				[attr, max_a] = decoder.predict(z_sample)
				
				g, a, attr = unpad_data(max_a[0], attr[0])
				attr = np.clip(attr, 0.0, 1.0)
				# fixed_cmap, a_channel = plot_graph(g, max_a[0], attr, draw = False)

				figure[i * dataArgs["max_n_node"]: (i + 1) * dataArgs["max_n_node"], j * dataArgs["max_n_node"]: (j + 1) *    dataArgs["max_n_node"], :] = a_channel         

				plt.sca(axs[i, j])
				nx.draw_kamada_kawai(g, node_size=12, node_color=attr, width=0.2, font_color='white', cmap=fixed_cmap)
				axs[i, j].set_axis_off()

		start_range = dataArgs["max_n_node"] // 2
		end_range = (analyzeArgs["size_of_manifold"] - 1) * dataArgs["max_n_node"] + start_range + 1
		pixel_range = np.arange(start_range, end_range, dataArgs["max_n_node"])
		sample_range_x = np.round(grid_x, 1)
		sample_range_y = np.round(grid_y, 1)


		plt.figure(figsize=(10, 10))
		plt.xticks([])
		plt.yticks([])
		plt.xlabel("z_" + str(analyzeArgs["z"][0]), fontweight='bold')
		plt.ylabel("z_" + str(analyzeArgs["z"][1]), fontweight='bold')
		plt.imshow(figure, cmap='Greys_r')
		
		
		
		
		
	elif len(analyzeArgs["z"]) == 1 or modelArgs["latent_dim"] == 1:
		
		z_sample = np.zeros(modelArgs["latent_dim"])
		z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

		## fill unobserved dimensions with mean of latent variable dimension
		for dim in range(0, len(z_sample[0])):
			z_sample[0][dim] = np.mean(z_mean[:, dim])

			
		grid_x = np.linspace(analyzeArgs["range"][0], analyzeArgs["range"][1], analyzeArgs["size_of_manifold"])

		figure = np.zeros((1 * dataArgs["max_n_node"], analyzeArgs["size_of_manifold"] * dataArgs["max_n_node"], 3))
		fig, axs = plt.subplots(1, analyzeArgs["size_of_manifold"], figsize=(8, 2))

		## Set common labels
		fig.text(0.5, 0.04, "z_" + str(analyzeArgs["z"][0]), ha='center')
						 
		axs = axs.ravel()
		for j, xi in enumerate(grid_x):

			xi_value = xi ** 1

			try: 
				z_sample[0][analyzeArgs["z"][0]] = xi ** 1
			except:
				print("please select correct latent variables")
				print("number of latent variables to choose from: z_" + str(np.arange(modelArgs["latent_dim"])))
				sys.exit()

			[attr, max_a] = decoder.predict(z_sample)

			g, a, attr = unpad_data(max_a[0], attr[0])
			attr = np.clip(attr, 0.0, 1.0)
			# fixed_cmap, a_channel = plot_graph(g, max_a[0], attr, draw = False)
					
			figure[0:dataArgs["max_n_node"], j * dataArgs["max_n_node"]: (j + 1) * dataArgs["max_n_node"]] = a_channel

			jx = np.unravel_index(j, axs.shape)
			plt.sca(axs[jx])

			nx.draw_kamada_kawai(g, node_size=12, node_color=attr, width=0.2, font_color='white', cmap=fixed_cmap)
			axs[jx].set_axis_off()
			axs[jx].set(ylabel='z_0')

		start_range = dataArgs["max_n_node"] // 2
		end_range = (analyzeArgs["size_of_manifold"] - 1) * dataArgs["max_n_node"] + start_range + 1
		pixel_range = np.arange(start_range, end_range, dataArgs["max_n_node"])
		sample_range_x = np.round(grid_x, 1)


		plt.figure(figsize=(10, 10))
		#plt.axis('off')
		plt.xticks([])
		plt.yticks([])
		plt.xlabel("z_" + str(analyzeArgs["z"][0]), fontweight='bold')
		plt.imshow(figure, cmap='Greys_r')


#@title    Interpolation Manifold Specifications

analyzeArgs = {"z": [0,1], "act_range": [-4, 4], "act_scale": 1, "size_of_manifold": 7}

#@markdown select one or two latent variables to visualize (limited by number of variables used in model)

z_vars = list()
z_0 = True #@param {type:"boolean"}
if z_0:
	z_vars.append(0)
z_1 = False #@param {type:"boolean"}
if z_1:
	z_vars.append(1)
z_2 = True #@param {type:"boolean"}
if z_2:
	z_vars.append(2)
z_3 = False #@param {type:"boolean"}
if z_3:
	z_vars.append(3)
z_4 = False #@param {type:"boolean"}
if z_4:
	z_vars.append(4)
z_5 = False #@param {type:"boolean"}
if z_5:
	z_vars.append(5)

analyzeArgs["z"] = np.asarray(z_vars)

print("the trained model comprises " + str(modelArgs["latent_dim"]) + " latent variables from which z_" + str(analyzeArgs["z"]) + " are visualized.\n\n")


#@markdown manifold settings
interpolation_range =    "-6,6" #@param [[-2,2], [-4, 4], [-6, 6]]
size_of_manifold = "15" #@param [5, 7, 10, 15]

analyzeArgs["range"] = [int(interpolation_range.split(",")[0]),int(interpolation_range.split(",")[1])]
analyzeArgs["size_of_manifold"] = int(size_of_manifold)

generate_manifold(analyzeArgs, modelArgs, trainArgs, model, test_data)


# # _4 Latent Space Analysis using the Encoder

# In[ ]:


#@title Mutual Information Gap Support Functions

def compute_mig(z, v):
	
	if z.shape[0] > 1:
	
		## normalize data
		z,z_mean,z_std = normalize_data(z)
		v,v_mean,v_std = normalize_data(v)

		## discretize data
		z = discretize_data(z)
		v = discretize_data(v)

		m = discrete_mutual_info(z, v)
		assert m.shape[0] == z.shape[0]
		assert m.shape[1] == v.shape[0]
		# m is [num_latents, num_factors]
		entropy = discrete_entropy(v)
		sorted_m = np.sort(m, axis=0)[::-1]

		mig_score = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
	
	else:
	
		mig_score = "MIG not defined for one latent variable"
	
	return mig_score



## Utilities_______________________________

"""Utility functions that are useful for the different metrics."""
import sklearn

def discrete_mutual_info(z, v):
	"""Compute discrete mutual information."""
	num_codes = z.shape[0]
	num_factors = v.shape[0]
	m = np.zeros([num_codes, num_factors])
	for i in range(num_codes):
		for j in range(num_factors):
			
			if num_factors > 1:
				m[i, j] = sklearn.metrics.mutual_info_score(v[j, :], z[i, :])
			elif num_factors == 1:
				m[i, j] = sklearn.metrics.mutual_info_score(np.squeeze(v), z[i, :])
		
	return m


def discrete_entropy(ys):
	"""Compute discrete mutual information."""
	num_factors = ys.shape[0]
	h = np.zeros(num_factors)
	for j in range(num_factors):
		h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
	return h


def normalize_data(data, mean=None, stddev=None):
	if mean is None:
		mean = np.mean(data, axis=1)
	if stddev is None:
		stddev = np.std(data, axis=1)
	return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


def discretize_data(target, num_bins=10):
	"""Discretization based on histograms."""
	target = np.nan_to_num(target)
	discretized = np.zeros_like(target)
	for i in range(target.shape[0]):
		discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
	return discretized


#@title Correlation between generative parameters v_k and latent variables z_j
def latent_space_encoder(modelArgs, trainArgs, dataArgs, model, data):
	
	n_samples = 1000
	Attr, A_mod, Param, Topol = data
	Attr, A_mod, Param, Topol = Attr[:n_samples], A_mod[:n_samples], Param[:n_samples], Topol[:n_samples]

	encoder, decoder = model    ## trained model parts
	z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
	z_var = K.exp(0.5 * z_log_var)
		
	param_txt = ["n", "p", "attr_param"]
	topol_txt = ["density", "diameter", "assort", "#edges", "avg_degree"]
	 
	
	if dataArgs["node_attr"] == "none":
		param_txt = param_txt[:2]
		Param = Param[:,:2]
		

	## Measuring the Mutual Information Gap ____________________________________________
	
	v = np.reshape(Param, (Param.shape[1], Param.shape[0]))
	z = np.reshape(z_mean, (z_mean.shape[1], z_mean.shape[0])) 
		
	mig_score = compute_mig(z, v)
	
	
	
	if z_sample.shape[-1] >= 2:
	
		## (1) Generative Parameters________________________________________________________

		fig, ax = plt.subplots(nrows= z_sample.shape[-1] , ncols= len(param_txt), figsize = (15,9))
		fig.suptitle('Generative Parameters v' + " – Mutual Information Gap (MIG) score:" + str(round(mig_score, 7)), fontweight = "bold")
		
		for z, row in enumerate(ax):                
			for v, col in enumerate(row):

				plt.ylim(-4, 4)
				y = z_sample[:,z]
				x = Param[:,v]
				sns.regplot(x, y, color="steelblue", ax=col, scatter_kws={'alpha':0.3}, order=2)

				corr = round(pearsonr(x,y)[0],3)
				cov = round(np.cov(x, y)[0][1]/max(x),3)
				col.annotate("corr:"+str(corr)+", cov:"+str(cov), xy=(0, 1), xytext=(12, -12), va='top',xycoords='axes fraction', textcoords='offset points')

		## add row and column titles
		rows = ['z_{}'.format(row) for row in range(z_sample.shape[-1])]
		cols = [t for t in param_txt]

		for axis, col in zip(ax[0], cols):
			axis.set_title(col, fontweight='bold')

		for axis, row in zip(ax[:,0], rows):
			axis.set_ylabel(row, rotation=0, size='large', fontweight='bold')



		## (2) Graph Topology_______________________________________________________

		fig, ax = plt.subplots(nrows= z_sample.shape[-1] , ncols= len(topol_txt), figsize = (20,6))
		fig.suptitle('Graph Topology', fontweight = "bold")
		
		for z, row in enumerate(ax):                
			for v, col in enumerate(row):

				plt.ylim(-4, 4)
				y = z_sample[:,z]
				x = Topol[:,v]
				sns.regplot(x, y, color="steelblue", ax=col, scatter_kws={'alpha':0.3})

				corr = round(pearsonr(x,y)[0],3)
				cov = round(np.cov(x, y)[0][1]/max(x),3)
				col.annotate("corr:"+str(corr)+", cov:"+str(cov), xy=(0, 1), xytext=(12, -12), va='top',xycoords='axes fraction', textcoords='offset points')

		## add row and column titles
		rows = ['z_{}'.format(row) for row in range(z_sample.shape[-1])]
		cols = [t for t in topol_txt]

		for axis, col in zip(ax[0], cols):
			axis.set_title(col, fontweight='bold')

		for axis, row in zip(ax[:,0], rows):
			axis.set_ylabel(row, rotation=0, size='large', fontweight='bold')
			
			
			
			

	elif z_sample.shape[-1] == 1:
	
		## (1) Generative Parameters________________________________________________________

		fig, ax = plt.subplots(nrows= z_sample.shape[-1] , ncols= len(param_txt), figsize = (20,5))
		fig.suptitle('Generative Parameters v' + " – Mutual Information Gap (MIG) score: " + str(round(mig_score, 7)), fontweight = "bold")

		
		for v, col in enumerate(range(len(param_txt))):

			plt.sca(ax[v])
			plt.ylim(-4, 4)    
			y = z_sample[:,0]
			x = Param[:,v]
			sns.regplot(x, y, color="steelblue", scatter_kws={'alpha':0.3}, order=2)

			corr = round(pearsonr(x,y)[0],3)
			cov = round(np.cov(x, y)[0][1]/max(x),3)
			plt.annotate("corr:"+str(corr)+", cov:"+str(cov), xy=(0, 1), xytext=(12, -12), va='top',xycoords='axes fraction', textcoords='offset points')

		## add row and column titles
		cols = [t for t in param_txt]
		fig.text(0.1, 0.5, 'z_0', fontweight = "bold", ha='center', va='center', rotation='vertical')

		for axis, col in zip(ax[:,], cols):
			axis.set_title(col, fontweight='bold')



		## (2) Graph Topology_______________________________________________________

		fig, ax = plt.subplots(nrows= z_sample.shape[-1] , ncols= len(topol_txt), figsize = (20,6))
		
		for v, col in enumerate(range(len(topol_txt))):

		
			plt.sca(ax[v])
			plt.ylim(-4, 4)
			y = z_sample[:,0]
			x = Topol[:,v]
			sns.regplot(x, y, color="steelblue", scatter_kws={'alpha':0.3}, order=2)

			corr = round(pearsonr(x,y)[0],3)
			cov = round(np.cov(x, y)[0][1]/max(x),3)
			plt.annotate("corr:"+str(corr)+", cov:"+str(cov), xy=(0, 1), xytext=(12, -12), va='top',xycoords='axes fraction', textcoords='offset points')

			## add row and column titles
			cols = [t for t in topol_txt]
			fig.text(0.1, 0.5, 'z_0', fontweight = "bold", ha='center', va='center', rotation='vertical')

			for axis, col in zip(ax[:,], cols):
				axis.set_title(col, fontweight='bold')




## PLOT RESULTS ________________________________________

latent_space_encoder(modelArgs, trainArgs, dataArgs, model, test_data)

#@title Posterior Distributions


def visualize_distributions(modelArgs, trainArgs, model, data):

	n_samples = 1000
	Attr, A_mod, Param, Topol = data
	Attr, A_mod, Param, Topol = Attr[:n_samples], A_mod[:n_samples], Param[:n_samples], Topol[:n_samples]

	encoder, decoder = model    ## trained model parts
	z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
	z_var = K.exp(0.5 * z_log_var)     
	
	col_titles = ['z_{}'.format(col) for col in range(z_mean.shape[1])]
	
	fig, ax = plt.subplots(nrows= 1, ncols= z_sample.shape[-1], figsize = (10,3))

	if z_sample.shape[-1] > 1:     
		for z, col in enumerate(ax): 

			plt.sca(ax[z])
			col.yaxis.set_visible(False)
			plt.xlabel('z_'+str(z), fontweight = "bold")
			grid = np.linspace(-4, 4, 1000)
			kde_z = scipy.stats.gaussian_kde(z_sample[:, z])

			plt.plot(grid, norm.pdf(grid, 0.0, 1.0), label="Gaussian prior", color='steelblue', linestyle=':',markerfacecolor='blue', linewidth=6)
			plt.plot(grid, kde_z(grid), label="z", color='midnightblue', markerfacecolor='blue', linewidth=6)

	else:
			plt.yticks([])
			plt.xlabel('z_0', fontweight = "bold")
			grid = np.linspace(-4, 4, 1000)
			kde_z = scipy.stats.gaussian_kde(z_sample[:, 0])

			plt.plot(grid, norm.pdf(grid, 0.0, 1.0), label="Gaussian prior", color='steelblue', linestyle=':',markerfacecolor='blue', linewidth=6)
			plt.plot(grid, kde_z(grid), label="z", color='midnightblue', markerfacecolor='blue', linewidth=6)
	
		


visualize_distributions(modelArgs, trainArgs, model, test_data)


#@title Dependency between Node Attributes and Graph Topology


analyzeArgs = dict()

#@markdown select type of node attribute randomization (note: shuffling will not have any effects if all nodes attributes in a graph are the same)

attribute_randomization =    "assign random attributes" #@param ["shuffle attributes", "assign random attributes"]
analyzeArgs["randomization"] = attribute_randomization


def attr_topol_correlation(analyzeArgs, modelArgs, trainArgs, dataArgs, model, data):

	n_samples = 1000
	Attr, A_mod, Param, Topol = data
	Attr, A_mod, Param, Topol = Attr[:n_samples], A_mod[:n_samples], Param[:n_samples], Topol[:n_samples]
		
	param_txt = ["n", "p", "attr_param"]     

				 
	## Randomize Attributes ___________________________________________
				 
	Attr_rand = np.copy(Attr)
	Rand_degree = np.zeros((Attr_rand.shape[0]))
	
	for i in range(Attr_rand.shape[0]):
		
		rand_degree = random.uniform(0.0, 1.0)
		attr_rand = Attr_rand[i]
		
		## reshape attr and unpad
		attr_rand = np.reshape(attr_rand, (attr_rand.shape[0]))
		nodes_n = attr_rand[attr_rand>0.0].shape[0]
		attr_rand = attr_rand[:nodes_n]    ## shorten
		
		if analyzeArgs["randomization"] == "shuffle attributes":
			#math.ceil
			for m in range(0, int(rand_degree * nodes_n)):
				swap = np.random.randint(low = 0, high = attr_rand.shape[0], size = 2)
				temp = attr_rand[swap[0]]
				attr_rand[swap[0]] = attr_rand[swap[1]]
				attr_rand[swap[1]] = temp

		elif analyzeArgs["randomization"] == "assign random attributes":
			
			rand_n = np.random.choice(nodes_n, int(rand_degree * nodes_n), replace=False)
			rand_value = np.random.uniform(0,1, rand_n.shape[0])
			 
			attr_rand[rand_n] = rand_value

			
		## pad features with zeroes
		
		zeroes = np.zeros((dataArgs["max_n_node"] - attr_rand.shape[0]))
		attr_rand = np.concatenate((attr_rand, zeroes))
		attr_rand = np.reshape(attr_rand, (attr_rand.shape[-1],1))
		
		Attr_rand[i] = attr_rand 
		Rand_degree[i] = rand_degree
			 

	## Encoder ______________________________________________
	
	encoder, decoder = model    ## trained model parts
	
	## 1) Original Attributes
	z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
	z_var = K.exp(0.5 * z_log_var)
	
	## 2) Randomized Attributes
	z_mean_rand, z_log_var_rand, z_sample_rand = encoder.predict([Attr_rand, A_mod], trainArgs["batch_size"])
	z_var_rand = K.exp(0.5 * z_log_var_rand)     
	
	z_shift = np.abs(z_sample - z_sample_rand)    
	v_rand = Rand_degree
		
	## Measuring the Mutual Information Gap ____________________________________________
		
	v_rand_reshaped = np.reshape(v_rand, (1, v_rand.shape[0]))
	z_shift_reshaped = np.reshape(z_shift, (z_shift.shape[1], z_shift.shape[0]))
			
	mig_score = compute_mig(z_shift_reshaped, v_rand_reshaped)
	#print("Mutual Information Gap (MIG) score:", round(mig_score, 7))
	
	#sys.exit()
	
	## Latent Variables and Attribute Shift ____________________________

	fig, ax = plt.subplots(nrows= 1, ncols= z_shift.shape[-1], figsize=(15,6))
	fig.suptitle('Correlation between Node Attributes and Latent Variables' + " – Mutual Information Gap (MIG) score: " + str(round(mig_score, 7)), fontweight = "bold")

	for latent_z, col in enumerate(ax):                

		plt.sca(ax[latent_z])
		plt.ylim(0, 3)
		y = z_shift[:,latent_z]
		x = v_rand
		sns.regplot(x, y, color="steelblue", ax=col, ci = None, x_ci='sd', scatter_kws={'alpha':0.3}, order=2)

		## compute correlation and standardized covariance
		corr = round(pearsonr(x,y)[0],3)
		cov = round(np.cov(x, y)[0][1]/max(x),3)
		std = round(np.std(y),3)
		col.annotate("corr:"+str(corr)+", std:"+str(std), xy=(0, 1), xytext=(12, -12), va='top',xycoords='axes fraction', textcoords='offset points', fontweight='bold')
		col.set_xlabel('attribute randomization degree', fontweight = "bold")
		col.set_ylabel('shift in z_' + str(latent_z), rotation=90, fontweight = "bold")
			

if dataArgs["node_attr"] == "none":
	print("generated graphs do not feature node attributes")
else:
	attr_topol_correlation(analyzeArgs, modelArgs, trainArgs, dataArgs, model, test_data)


# **Some pieces of this code are inspired by or modified versions of existing code bases.**
# 
# Variational Autoencoder – https://keras.io/examples/variational\_autoencoder/ <br>
# Graph Sampling Algorithms – https://github.com/Ashish7129/Graph\_Sampling<br>
# Graph Neural Network Library – https://vermamachinelearning.github.io<br>
# Mutual Information Gap (MIG) – https://github.com/google-research/disentanglement_lib