import numpy as np
import networkx as nx
from utils import *
from matplotlib import pyplot as plt

def showLoss(modelName, train_losses, validation_losses=None):
	plt.figure()
	plt.title(modelName + " Loss")
	plt.plot(np.arange(len(train_losses)), np.array(train_losses), label = "train loss")
	if validation_losses:
		plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), label = "test loss")
	plt.legend()
	plt.show()

def drawGraph(A_train, batched_A_hat, edit_train=None, gen_A_train=None, w_alpha=None, showGraph=True, sample_size=2):
	# sampling
	edit_A, gen_A, edit_A_sample, gen_A_sample, w_alpha, col_size = None, None, None, None, None, 2
	if edit_train:
		col_size = 4
	has_edit_graph = False
	max_n_node = A_train[0].shape[1]
	if type(A_train) != list:
		A_train = [A_train]
		batched_A_hat = [batched_A_hat]
	a = reshapeMatrix(A_train[0].cpu().numpy().squeeze(-1))
	a_hat = reshapeMatrix(batched_A_hat[0].cpu().numpy().squeeze(-1))
	if edit_train != None:
		edit_A = reshapeMatrix(edit_train[0].cpu().numpy().squeeze(-1))
		has_edit_graph = True
	if gen_A_train != None:
		gen_A = gen_A_train[0].detach().numpy().squeeze(-1)
		discretizer_gen_A = Discretizer(gen_A, gen_A)
		gen_A = discretizer_gen_A.discretize('hard_threshold')
		gen_A = reshapeMatrix(gen_A)

	if len(a.shape) == 2:
		a_sample = [a]
		a_hat_sample = [a_hat]
		A_str = " A hat"
		A_hat_str = "Edit A hat"
	else:
		assert a.shape[0] == a_hat.shape[0]
		sample_space_size = a_hat.shape[0]
		sample_idx = np.random.choice(range(sample_space_size), sample_size, replace=False)
		a_sample = a[sample_idx]
		a_hat_sample = a_hat[sample_idx]
		if has_edit_graph:
			edit_A_sample = edit_A[sample_idx]
		if has_edit_graph:
			gen_A_sample = gen_A[sample_idx]
		A_str = " A"
		A_hat_str = "A hat"

	# visualize using networkx
	if showGraph:
		sample_size = len(a_sample)
		G = nx.grid_2d_graph(sample_size,col_size)  #4x4 grid
		pos = nx.spring_layout(G,iterations=100)
		fig = plt.figure()

		for i in range(sample_size):
			edit_graph, gen_graph = None, None
			a_graph = nx.from_numpy_matrix(a_sample[i])
			a_hat_graph = nx.from_numpy_matrix(a_hat_sample[i])
			if has_edit_graph:
				edit_graph = nx.from_numpy_matrix(edit_A_sample[i])
			if has_edit_graph:
				gen_graph = nx.from_numpy_matrix(gen_A_sample[i])
			ax = fig.add_subplot(sample_size,col_size,i*col_size+1)
			if i == 0:
				ax.title.set_text(A_str)
			nx.draw(a_graph)

			ax = fig.add_subplot(sample_size,col_size,i*col_size+2)
			if i == 0:
				ax.title.set_text(A_hat_str)
			nx.draw(a_hat_graph)

			if has_edit_graph:
				ax = fig.add_subplot(sample_size,col_size,i*col_size+3)
				if i == 0:
					ax.title.set_text("Edit A hat")
				nx.draw(edit_graph)

			if has_edit_graph:
				ax = fig.add_subplot(sample_size,col_size,i*col_size+4)
				if i == 0:
					ax.title.set_text("Gen A hat")
				nx.draw(gen_graph)
			
		if has_edit_graph:
			plt.subplots_adjust(left=0.1,right=0.9,top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
		else:
			plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
		plt.show()

	fig = plt.figure()
	for i in range(sample_size):
		edit_a, gen_a = None, None
		a = padMatrix(a_sample[i], max_n_node)
		a_hat = padMatrix(a_hat_sample[i], max_n_node)
		if has_edit_graph:
			edit_a = padMatrix(edit_A_sample[i], max_n_node)
			gen_a = padMatrix(gen_A_sample[i], max_n_node)


		ax = fig.add_subplot(sample_size,col_size,i*col_size+1)
		if i == 0:
			ax.title.set_text(A_str)
		plt.imshow(a, cmap='binary')

		ax = fig.add_subplot(sample_size,col_size,i*col_size+2)
		if i == 0:
			ax.title.set_text(A_hat_str)
		plt.imshow(a_hat, cmap='binary')

		if has_edit_graph:
			ax = fig.add_subplot(sample_size,col_size,i*col_size+3)
			if i == 0:
				ax.title.set_text("Edit A hat")
			plt.imshow(edit_a, cmap='binary')

		if has_edit_graph:
			ax = fig.add_subplot(sample_size,col_size,i*col_size+4)
			if i == 0:
				ax.title.set_text("Gen A hat")
			plt.imshow(gen_a, cmap='binary')

	if has_edit_graph:
		plt.subplots_adjust(left=0.1,right=0.9,top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
	else:
		plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
	plt.show()
	
def debugDecoder(A_train, A_validate, batched_A_hat, batched_A_hat_test, discretize_method, printMatrix):
	A = []
	A_hat = []
	for i in range(len(A_train)):
		A.extend(A_train[i])
		A_hat.extend(batched_A_hat[i])
	for i in range(len(A_validate)):
		A.extend(A_validate[i])
		A_hat.extend(batched_A_hat_test[i])
	A = torch.stack(A).squeeze().numpy()
	A_hat = torch.stack(A_hat).squeeze().numpy()
	A_hat_raw = copy.deepcopy(A_hat)

	discretizer = Discretizer(A, A_hat)
	A_hat = discretizer.discretize(discretize_method)
	recall = 0
	accuracy = 0
	precision = 0
	for i,a in enumerate(A):
		a_hat = A_hat[i]
		a_hat_raw = A_hat_raw[i]
		num_of_node = 0
		accuracy += 1 - (np.sum(abs(a - a_hat)) / (a.shape[0] * a.shape[0]))
		true_positive = 0
		false_negative = 0
		for r in range(a.shape[0]):
			for c in range(a.shape[0]):
				if a_hat[r][c] == a[r][c] and a_hat[r][c] == 1:
					true_positive += 1 
				if a_hat[r][c] == 0 and a[r][c] == 1:
					false_negative += 1
		if np.sum(a_hat) != 0:
			precision += true_positive / np.sum(a_hat)
		if true_positive + false_negative > 0:
			recall += true_positive / (true_positive + false_negative)
		if printMatrix:
			print('=====')
			print('A:')
			print(a)
			print('A_hat_raw:')
			print(a_hat_raw)
			print('A_hat_discretized:')
			print(a_hat)
			print('=====')
	accuracy /= len(A)
	precision /= len(A)
	recall /= len(A)
	f1_score = (2*precision*recall) / (precision+recall)
	print("accuracy:", accuracy)
	print("precision:", precision)
	print("recall:", recall)
	print("f1 score:", f1_score)
