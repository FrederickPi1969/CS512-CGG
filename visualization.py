import numpy as np
import networkx as nx
import pickle,os,glob
from utils import *
from matplotlib import pyplot as plt

"""
Visualize the loss function given the model name and the array of losses

Param:
'modelName': name of the model (VAE/Discriminator/Steering GAN)
'train_losses': 1D array of training loss for each epoch
'validation_losses': 1D array of validation loss for each epoch
"""
def showLoss(modelName, train_losses, validation_losses=None):
	plt.figure()
	plt.title(modelName + " Loss")
	plt.plot(np.arange(len(train_losses)), np.array(train_losses), label = "train loss")
	if validation_losses:
		plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), label = "test loss")
	plt.legend()
	plt.show()

"""
Visualize the networkx graph and the adjacency matrix heatmap given 
arrays of different batched adjacency matrix tensor with shape: 
(batch_size, max_n_node, max_n_node, 1)

Param:
'A_train': 1D array of batched ground truth adjacency matrix tensor with above shape
'batched_A_hat': 1D array of discretized A hat adjacency matrix tensor with above shape
'edit_train': 1D array of batched edited A hat adjacency matrix tensor with above shape
'gen_A_train': 1D array of batched predicted raw gen_A adjacency matrix tensor with above shape
'w_alpha': 1D array of alpha for each epoch, shape: (epoch num, 1) [unfinished]
'showGraph': a boolean parameter to decide whether we should draw the networkx graph during the visualization
'sample_size': a integer determine how many sample we draw during the visualization 
				default value is 2
"""
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
		gen_A = discretizer_gen_A.discretize('gen_hard_threshold')
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



"""
Similar function with the previous one with additional saving plot figure feature

Visualize the networkx graph and the adjacency matrix heatmap given 
arrays of different batched adjacency matrix tensor with shape: 
(batch_size, max_n_node, max_n_node, 1)

Param:
'A_train': 1D array of batched ground truth adjacency matrix tensor with above shape
'batched_A_hat': 1D array of discretized A hat adjacency matrix tensor with above shape
'edit_train': 1D array of batched edited A hat adjacency matrix tensor with above shape
'gen_A_train': 1D array of batched predicted raw gen_A adjacency matrix tensor with above shape
'w_alpha': 1D array of alpha for each epoch, shape: (epoch num, 1) [unfinished]
'showGraph': a boolean parameter to decide whether we should draw the networkx graph during the visualization
'sample_size': a integer determine how many sample we draw during the visualization 
				default value is 30, each sample will be an individual .png file. Thus,
				by default, it will generate 30 png files under the directory 'currPath/Image/'
'clearImage': a boolean value determines whether we clear all the image in the directory 'currPath/Image/'
			  before we run this function
"""
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
	a_hat = batched_A_hat[0].cpu().numpy().squeeze(-1)
	discretizer_A_hat = Discretizer(a_hat, a_hat)
	a_hat = discretizer_A_hat.discretize('hard_threshold')
	a_hat = reshapeMatrix(a_hat)
	if edit_train != None:
		edit_A = reshapeMatrix(edit_train[0].cpu().numpy().squeeze(-1))
		has_edit_graph = True
	if gen_A_train != None:
		gen_A = gen_A_train[0].detach().cpu().numpy().squeeze(-1)
		discretizer_gen_A = Discretizer(gen_A, gen_A)
		gen_A = discretizer_gen_A.discretize('gen_hard_threshold')
		gen_A = reshapeMatrix(gen_A)
		
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
		A_str = " Ground Truth Matrix A"
		A_hat_str = "Decoded Matrix A hat"

	# visualize using networkx
	sample_size = len(a_sample)
	row_size = 1
	if showGraph:
		row_size = 2

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
			edit_a = padMatrix(edit_A_sample[i], max_n_node)
			gen_a = padMatrix(gen_A_sample[i], max_n_node)

		ax = fig.add_subplot(row_size,col_size,1)
		ax.title.set_text(A_str)
		plt.imshow(a, cmap='binary')

		if showGraph:
			ax = fig.add_subplot(row_size,col_size,5)
			nx.draw(a_graph)

		ax = fig.add_subplot(row_size,col_size,2)
		ax.title.set_text(A_hat_str)
		plt.imshow(a_hat, cmap='binary')

		if showGraph:
			ax = fig.add_subplot(row_size,col_size,6)
			nx.draw(a_hat_graph)

		if has_edit_graph:
			ax = fig.add_subplot(row_size,col_size,3)
			ax.title.set_text("Edit A hat")
			plt.imshow(edit_a, cmap='binary')

			if showGraph:
				ax = fig.add_subplot(row_size,col_size,7)
				nx.draw(edit_graph)

		if has_edit_graph:
			ax = fig.add_subplot(row_size,col_size,8)
			ax.title.set_text("Gen A hat")
			plt.imshow(gen_a, cmap='binary')

			if showGraph:
				ax = fig.add_subplot(row_size,col_size,4)
				nx.draw(gen_graph)
			
		if has_edit_graph:
			plt.subplots_adjust(left=0.1,right=0.9,top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)
		else:
			plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=0.3, wspace=0.3)

		plt.savefig('Image/sample_'+str(i+1)+'.png')
		plt.close()



"""
This function is used to compute the element-wise accuracy, precision, recall, 
and F-1 score between two matrix

matrix has the shape: (max_n_node, max_n_node)
"""
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



"""
A function used to debug the Steering GAN portion of the model.

[shape]: (batch_size, max_n_node, max_n_node, 1)

Param:
gen_edit_A_hat_train: 1D array of batched ground truth edit A hat tensor with above shape
gen_A_raw_train: 1D array of batched Decoder(z + alpha*w) output raw tensor with above shape
gen_A_max_train: 1D array of maximal value of each row of gen_A_raw_train
gen_A_min_train: 1D array of minimal value of each row of gen_A_raw_train
w_gen_A_hat_train: 1D array of batched Decoder(z + alpha*w) output normalized tensor with above shape
					normalized by (x - row_min) / row_max
discretize_method: a string to determine the discretize method for the normalized gen A matrix tensors
					default is hard threshold
printMatrix: a boolean value determines whether we print above debug information to the stdout 
abortPickle: a boolean value determines whether we clear all the pickles that store the previous array
computePerformance: a boolean value determines whether we print the accuracy/precision/recall/f-1score of the generated matrix

To improve scalability, this function will store all the above array into corrospounding pickle
files under the path "currPath/pickles/*.pickle".

To generate useful log information, set the 'printMatrix' flag to True and run the following
command:
	python3 main.py > somePath/steering_gan_log.txt
"""
def debugDiscretizer(original_A, gen_edit_A_hat_train, gen_A_raw_train, gen_A_max_train, gen_A_min_train, w_gen_A_hat_train,
					 masked_norm_A_hats, discretize_method="hard_threshold", printMatrix=True, abortPickle=False, computePerformance=True):
	# check pickle
	gen_A, edit_A, gen_A_max, gen_A_min, gen_A_normal,\
	gen_A_discretize, masked_normalized_A, original_As = [], [], [], [], [], None,[],[]
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
			edit_A.extend(gen_edit_A_hat_train[i])
			gen_A_max.extend(gen_A_max_train[i])
			gen_A_min.extend(gen_A_min_train[i])
			gen_A_normal.extend(w_gen_A_hat_train[i])
			masked_normalized_A.extend(masked_norm_A_hats[i])
			original_As.extend(original_A[i])



		gen_A = torch.stack(gen_A).squeeze().cpu().numpy()
		edit_A = torch.stack(edit_A).squeeze().numpy()
		gen_A_max = torch.stack(gen_A_max).cpu().squeeze().numpy()
		gen_A_min = torch.stack(gen_A_min).cpu().squeeze().numpy()
		gen_A_normal = torch.stack(gen_A_normal).cpu().squeeze().numpy()
		masked_normalized_A = torch.stack(masked_normalized_A).squeeze().numpy()
		original_As = torch.stack(original_As).squeeze().cpu().numpy()


		# discretizer = Discretizer(gen_A_normal, gen_A_normal)
		discretizer = Discretizer(masked_normalized_A, masked_normalized_A) ## Changed this to masked_normalized_A_hat
		gen_A_discretize = discretizer.discretize(discretize_method)
		dump_list[-1] = gen_A_discretize

		# store the pickle
		for i,file in enumerate(file_list):
			f = open('pickles/'+file+'.pickle', 'wb')
			pickle.dump(dump_list[i], f)
			f.close()

	recall,accuracy,precision = 0,0,0
	for i,a in enumerate(gen_A_discretize):
		a_raw = gen_A[i]
		edit_a = edit_A[i]
		max_A = gen_A_max[i]
		min_A = gen_A_min[i]
		masked_norm_A = masked_normalized_A[i]
		a_normal = gen_A_normal[i]
		ori_a= original_As[i]
		if computePerformance:
			recall_result,accuracy_result,precision_result = computeScore(edit_a, a)
			recall += recall_result
			accuracy += accuracy_result
			precision += precision_result

		if printMatrix:
			print("=============================")
			print("Original A:")
			print(ori_a)
			print('edit_a:')
			print(edit_a)
			print('gen_A_AAT_raw:')
			# print(a_raw)
			# print('max_A:')
			# print(max_A)
			# print('min_A:')
			# print(min_A)
			print("masked_normalized_A:")
			print(masked_norm_A)
			# print('gen_A_normalized:')
			# print(a_normal)
			print('gen_A_discretize:')
			print(a)
			print("=============================")
	if computePerformance:
		accuracy /= len(gen_A_discretize)
		precision /= len(gen_A_discretize)
		recall /= len(gen_A_discretize)
		f1_score = (2*precision*recall) / (precision+recall)
		print("accuracy:", accuracy)
		print("precision:", precision)
		print("recall:", recall)
		print("f1 score:", f1_score)



"""
A function used to debug the MIG decoder portion of the model.

[shape]: (batch_size, max_n_node, max_n_node, 1)

Param:
A_train: 1D array of batched ground truth A train matrix tensor with above shape
A_validate: 1D array of batched ground truth A validate matrix tensor with above shape
batched_A_hat: 1D array of decoded A hat matrix tensor with above shape
batched_A_hat_test: 1D array of decoded A hat validation matrix tensor with above shape
discretize_method: a string to determine the discretize method for the normalized A hat matrix tensors
					default is hard threshold
printMatrix: a boolean value determines whether we print above debug information to the stdout 

To generate useful log information, set the 'printMatrix' flag to True and run the following
command:
	python3 main.py > somePath/decoder_log.txt
"""
def debugDecoder(A_train, A_validate, batched_A_hat, batched_A_hat_test, discretize_method='hard_threshold', printMatrix=True):
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



"""
A function used to debug the MIG decoder portion of the model with additional debug
information: the dot product of the AA.T in the decoding process

[shape]: (batch_size, max_n_node, max_n_node, 1)

Param:
A_tuple: a tuple of 1D arrays of batched ground truth A train/validate matrix tensor with above shape
		 A_tuple = (A_train, A_validate)
A_hat_tuple: a tuple of 1D arrays of batched normalized A hat train/validate matrix tensor with above shape
		 A_hat_tuple = (batched_A_hat, batched_A_hat_test)
		 normalized by (x-row_min) / row_max
A_hat_raw_tuple: a tuple of 1D arrays of batched A hat raw train/validate matrix tensor with above shape
		 A_hat_tuple = (batched_A_hat_raw_train, batched_A_hat_raw_test)
		 A_hat_raw means the matrix generated directly from the dot product
A_max_tuple: a tuple of 1D arrays of batched row maxmal of A hat raw train/validate matrix tensor with above shape
		 A_max_tuple = (batched_A_hat_max_train, batched_A_hat_max_test)
A_min_tuple: a tuple of 1D arrays of batched row minimal of A hat raw train/validate matrix tensor with above shape
		 A_min_tuple = (batched_A_hat_min_train, batched_A_hat_min_test)
discretize_method: a string to determine the discretize method for the normalized A hat matrix tensors
					default is hard threshold
printMatrix: a boolean value determines whether we print above debug information to the stdout 

To generate useful log information, set the 'printMatrix' flag to True and run the following
command:
	python3 main.py > somePath/decoder_aat_log.txt
"""

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