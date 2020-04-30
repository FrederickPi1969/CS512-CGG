import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

def showLoss(modelName, train_losses, validation_losses=None):
	plt.figure()
	plt.title(modelName + " Loss")
	plt.plot(np.arange(len(train_losses)), np.array(train_losses), label = "train loss")
	if validation_losses:
		plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), label = "test loss")
	plt.legend()
	plt.show()

def drawGraph(A_train, batched_A_hat, edit_train=None, gen_A_train=None, sample_size=2):
	# sampling
	edit_A, gen_A, edit_A_sample, gen_A_sample, col_size = None, None, None, None, 2
	if edit_train:
		col_size = 4
	has_edit_graph = False
	a = A_train[0].cpu().numpy().squeeze(-1)        
	a_hat = batched_A_hat[0].cpu().numpy().squeeze(-1)
	if edit_train != None:
		edit_A = edit_train[0].cpu().numpy().squeeze(-1) 
		has_edit_graph = True
	if gen_A_train != None:
		gen_A = gen_A_train[0].detach().numpy().squeeze(-1)

	if len(a.shape) == 2:
		a_sample = [a]
		a_hat_sample = [a_hat]
		A_str = " A hat"
		A_hat_str = "Edit A hat"
	else:
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
		ax.title.set_text('Sample ' + str(i+1) + A_str)
		nx.draw(a_graph)

		ax = fig.add_subplot(sample_size,col_size,i*col_size+2)
		ax.title.set_text(A_hat_str)
		nx.draw(a_hat_graph)

		if has_edit_graph:
			ax = fig.add_subplot(sample_size,col_size,i*col_size+3)
			ax.title.set_text("Edit A hat")
			nx.draw(edit_graph)

		if has_edit_graph:
			ax = fig.add_subplot(sample_size,col_size,i*col_size+4)
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
		a = a_sample[i]
		a_hat = a_hat_sample[i]
		edit_a = edit_A_sample[i]
		gen_a = gen_A_sample[i]


		ax = fig.add_subplot(sample_size,col_size,i*col_size+1)
		ax.title.set_text('Sample ' + str(i+1) + A_str)
		plt.imshow(a, cmap='binary')

		ax = fig.add_subplot(sample_size,col_size,i*col_size+2)
		ax.title.set_text(A_hat_str)
		plt.imshow(a_hat, cmap='binary')

		if has_edit_graph:
			ax = fig.add_subplot(sample_size,col_size,i*col_size+3)
			ax.title.set_text("Edit A hat")
			plt.imshow(edit_a, cmap='binary')

		if has_edit_graph:
			ax = fig.add_subplot(sample_size,col_size,i*col_size+4)
			ax.title.set_text("Gen A hat")
			plt.imshow(gen_a, cmap='binary')

	if has_edit_graph:
		plt.subplots_adjust(left=0.1,right=0.9,top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
	else:
		plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
	plt.show()
	

