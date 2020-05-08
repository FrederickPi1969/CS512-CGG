import numpy as np
import networkx as nx
import pickle,os,glob
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
	has_edit_graph, max_n_node = False, A_train[0].shape[1]
	if type(A_train) != list:
		A_train = [A_train]
		batched_A_hat = [batched_A_hat]
	a = reshapeMatrix(A_train[0].cpu().numpy().squeeze(-1))
	a_hat = reshapeMatrix(batched_A_hat[0].cpu().numpy().squeeze(-1))
	if edit_train != None:
		edit_A = reshapeMatrix(edit_train[0].cpu().numpy().squeeze(-1))
		has_edit_graph = True
	if gen_A_train != None:
		gen_A = gen_A_train[0].detach().cpu().numpy().squeeze(-1)
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




def drawGraphSaveFigure(A_train, batched_A_hat, edit_train=None, gen_A_train=None, w_alpha=None, showGraph=True, sample_size=30, clearImage=False):
	# sampling
	if not os.path.exists('Image'):
		os.mkdir('Image')
	if clearImage:
		files = glob.glob('Image/*.png')
		for f in files:
		    os.remove(f)

	edit_A, gen_A, edit_A_sample, gen_A_sample, w_alpha, col_size = None, None, None, None, None, 2
	has_edit_graph, max_n_node = False, A_train[0].shape[1]
	
	if edit_train:
		col_size = 4
	if type(A_train) != list:
		A_train = [A_train]
		batched_A_hat = [batched_A_hat]

	a = reshapeMatrix(A_train[0].cpu().numpy().squeeze(-1))
	a_hat = reshapeMatrix(batched_A_hat[0].cpu().numpy().squeeze(-1))

	if edit_train != None:
		edit_A = reshapeMatrix(edit_train[0].cpu().numpy().squeeze(-1))
		has_edit_graph = True
	if gen_A_train != None:
		gen_A = gen_A_train[0].detach().cpu().numpy().squeeze(-1)

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
	if showGraph:
		sample_size = len(a_sample)
		
		for i in range(sample_size):
			
			G = nx.grid_2d_graph(2,col_size)  #2x4 grid
			pos = nx.spring_layout(G,iterations=100)
			fig = plt.figure()

			edit_a, gen_a, edit_graph, gen_graph = None, None, None, None
			a = padMatrix(a_sample[i], max_n_node)
			a_hat = padMatrix(a_hat_sample[i], max_n_node)
			a_graph = nx.from_numpy_matrix(a_sample[i])
			a_hat_graph = nx.from_numpy_matrix(a_hat_sample[i])
		
			if has_edit_graph:
				edit_graph = nx.from_numpy_matrix(edit_A_sample[i])
				gen_graph = nx.from_numpy_matrix(gen_A_sample[i])
			if has_edit_graph:
				edit_a = padMatrix(edit_A_sample[i], max_n_node)
				gen_a = padMatrix(gen_A_sample[i], max_n_node)

			ax = fig.add_subplot(2,col_size,1)
			ax.title.set_text(A_str)
			nx.draw(a_graph)

			ax = fig.add_subplot(2,col_size,5)
			plt.imshow(a, cmap='binary')

			ax = fig.add_subplot(2,col_size,2)
			ax.title.set_text(A_hat_str)
			nx.draw(a_hat_graph)

			ax = fig.add_subplot(2,col_size,6)
			plt.imshow(a_hat, cmap='binary')

			if has_edit_graph:
				ax = fig.add_subplot(2,col_size,3)
				ax.title.set_text("Edit A hat")
				nx.draw(edit_graph)

				ax = fig.add_subplot(2,col_size,7)
				plt.imshow(edit_a, cmap='binary')

			if has_edit_graph:
				ax = fig.add_subplot(2,col_size,4)
				ax.title.set_text("Gen A hat")
				nx.draw(gen_graph)

				ax = fig.add_subplot(2,col_size,8)
				plt.imshow(gen_a, cmap='binary')
			
			if has_edit_graph:
				plt.subplots_adjust(left=0.1,right=0.9,top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
			else:
				plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)

			plt.savefig('Image/sample_'+str(i+1)+'.png')
			plt.close()





def debugDiscretizer(gew_edit_A_hat_train, gen_A_raw_train, gen_A_max_train, gen_A_min_train, w_gen_A_hat_train, discretize_method="hard_threshold", printMatrix=True, abortPickle=False):
	# check pickle
	gen_A, edit_A, gen_A_max, gen_A_min, gen_A_normal, gen_A_discretize = [], [], [], [], [], None
	dump_list = [gen_A, edit_A, gen_A_max, gen_A_min, gen_A_normal, gen_A_discretize]
	file_list = ['gen_A', 'edit_A', 'gen_A_max', 'gen_A_min', 'gen_A_normal', 'gen_A_discretize']
	if not os.path.exists('pickles'):
		os.mkdir('pickles')
	if os.path.exists('pickles/gen_A.pickle') and not abortPickle:
		# load the pickle
		for i,file in enumerate(file_list):
			f = open('pickles/'+file+'.pickle', 'rb')
			dump_list[i] = pickle.load(f)
			f.close()

		gen_A, edit_A, gen_A_max, gen_A_min, gen_A_normal, gen_A_discretize = dump_list
	else:
		for i in range(len(gen_A_raw_train)):
			gen_A.extend(gen_A_raw_train[i])
			edit_A.extend(gew_edit_A_hat_train[i])
			gen_A_max.extend(gen_A_max_train[i])
			gen_A_min.extend(gen_A_min_train[i])
			gen_A_normal.extend(w_gen_A_hat_train[i])

		gen_A = torch.stack(gen_A).squeeze().cpu().numpy()
		edit_A = torch.stack(edit_A).squeeze().numpy()
		gen_A_max = torch.stack(gen_A_max).cpu().squeeze().numpy()
		gen_A_min = torch.stack(gen_A_min).cpu().squeeze().numpy()
		gen_A_normal = torch.stack(gen_A_normal).cpu().squeeze().numpy()

		discretizer = Discretizer(gen_A_normal, gen_A_normal)
		gen_A_discretize = discretizer.discretize(discretize_method)
		dump_list[-1] = gen_A_discretize

		# store the pickle
		for i,file in enumerate(file_list):
			f = open('pickles/'+file+'.pickle', 'wb')
			pickle.dump(dump_list[i], f)
			f.close()

	
	for i,a in enumerate(gen_A_discretize):
		a_raw = gen_A[i]
		edit_a = edit_A[i]
		max_A = gen_A_max[i]
		min_A = gen_A_min[i]
		a_normal = gen_A_normal[i]
		if printMatrix:	
			print('=====')
			print('edit_a:')
			print(edit_a)
			print('gen_A_AAT_raw:')
			print(a_raw)
			print('max_A:')
			print(max_A)
			print('min_A:')
			print(min_A)
			print('gen_A_normalized:')
			print(a_normal)
			print('gen_A_discretize:')
			print(a)
			print('=====')




def computeScore(a, a_hat):
	recall = 0
	accuracy = 0
	precision = 0
	true_positive = 0
	false_negative = 0

	accuracy += 1 - (np.sum(abs(a - a_hat)) / (a.shape[0] * a.shape[0]))

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
	return recall,accuracy,precision




def debugDecoder(A_train, A_validate, batched_A_hat, batched_A_hat_test, batched_A_hat_raw_train=None, batched_A_hat_max_train=None, batched_A_hat_min_train=None, discretize_method='hard_threshold', printMatrix=True):
	A = []
	A_hat = []
	for i in range(len(A_train)):
		A.extend(A_train[i])
		A_hat.extend(batched_A_hat[i])
	for i in range(len(A_validate)):
		A.extend(A_validate[i])
		A_hat.extend(batched_A_hat_test[i])
	A = torch.stack(A).squeeze().numpy()
	A_hat = torch.stack(A_hat).squeeze().cpu().numpy()
	A_hat_raw = copy.deepcopy(A_hat)

	discretizer = Discretizer(A, A_hat)
	A_hat = discretizer.discretize(discretize_method)

	recall = 0
	accuracy = 0
	precision = 0
	for i,a in enumerate(A):
		a_hat = A_hat[i]
		a_hat_raw = A_hat_raw[i]
		recall_result,accuracy_result,precision_result = computeScore(a, a_hat)
		recall += recall_result
		accuracy += accuracy_result
		precision += precision_result

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





def debugDecoderAAT(A_tuple, A_hat_tuple, A_hat_raw_tuple, A_max_tuple, A_min_tuple, discretize_method="hard_threshold", printMatrix=True):
	A = []
	A_hat = []
	A_hat_raw = []
	A_max = []
	A_min = []

	A_train, A_validate = A_tuple
	batched_A_hat, batched_A_hat_test = A_hat_tuple
	batched_A_hat_raw_train, batched_A_hat_raw_test = A_hat_raw_tuple 
	batched_A_hat_max_train, batched_A_hat_max_test = A_max_tuple
	batched_A_hat_min_train, batched_A_hat_min_test = A_min_tuple

	for i in range(len(A_train)):
		A.extend(A_train[i])
		A_hat.extend(batched_A_hat[i])
		A_max.extend(batched_A_hat_max_train[i])
		A_min.extend(batched_A_hat_min_train[i])
		A_hat_raw.extend(batched_A_hat_raw_train[i])

	for i in range(len(A_validate)):
		A.extend(A_validate[i])
		A_hat.extend(batched_A_hat_test[i])
		A_max.extend(batched_A_hat_max_test[i])
		A_min.extend(batched_A_hat_min_test[i])
		A_hat_raw.extend(batched_A_hat_raw_test[i])

	A = torch.stack(A).squeeze().numpy()
	A_hat = torch.stack(A_hat).squeeze().numpy()
	A_max = torch.stack(A_max).squeeze().numpy()
	A_min = torch.stack(A_min).squeeze().numpy()
	A_hat_raw = torch.stack(A_hat_raw).squeeze().numpy()
	discretizer = Discretizer(A_hat, A_hat)
	A_hat_discretized = discretizer.discretize(discretize_method)

	recall = 0
	accuracy = 0
	precision = 0
	for i,a in enumerate(A):
		a_hat = A_hat[i]
		a_max = A_max[i]
		a_min = A_min[i]
		a_raw = A_hat_raw[i]
		a_hat_discretized = A_hat_discretized[i]

		recall_result,accuracy_result,precision_result = computeScore(a, a_hat_discretized)
		recall += recall_result
		accuracy += accuracy_result
		precision += precision_result

		if printMatrix:
			print('=====')
			print('A:')
			print(a)
			print('AA.T:')
			print(a_raw)
			print('A_max:')
			print(a_max)
			print('A_min:')
			print(a_min)
			print('A_hat:')
			print(a_hat)
			print('A_hat_discretize:')
			print(a_hat_discretized)
			print('=====')

	accuracy /= len(A)
	precision /= len(A)
	recall /= len(A)
	f1_score = (2*precision*recall) / (precision+recall)
	print("accuracy:", accuracy)
	print("precision:", precision)
	print("recall:", recall)
	print("f1 score:", f1_score)