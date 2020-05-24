import numpy as np
import pickle, copy, random
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

"""
A class to disscretize a batched matrix numpy array with the shape
(batch_size, max_n_node, max_n_node)

Usage:
# convert batched tensor to numpy array
ground_truth = ground_truth.cpu().numpy().squeeze(-1)
predicted = predicted.cpu().numpy().squeeze(-1)

# create discretizer object
discretizer = Discretizer(ground_truth, predicted)

# call the discretize method to discretized the predicted array
predicted = discretizer.discretize('hard_threshold')

# restore the numpy array back to tensor original shape
ground_truth = torch.unsqueeze(torch.from_numpy(A), -1)
predicted = torch.unsqueeze(torch.from_numpy(A_hat), -1)
"""
class Discretizer(object):
    """
    Discretizer constructor

    Param:
    A:  the ground truth matrix numpy array with the shape (batch_size, max_n_node, max_n_node)
    A_hat: the predicted matrix numpy array with the shape (batch_size, max_n_node, max_n_node)
    pretrain: a boolean value to determine whether we want to use the pretrained random 
              forest classifer as the discretizer
    filename: if pretrain = True, filename stores the path to the pretrained random forest
              model picke file
    """
    def __init__(self, A, A_hat, pretrain=False, filename=None): ## A_hat is continuous adi matrix
        assert A_hat.shape == A.shape
        assert len(A.shape) == 3 and A.shape[1] == A.shape[2]
        self.A = A
        self.A_hat = A_hat
        self.pretrain = pretrain
        self.filename = filename


    """
    Discretize interface

    Param:
    method: the method we want to use to discretize the array
            it includes:
                1. hard_threshold
                2. random_sampling (non-deterministic)
                3. random_forest
                4. vote_sampling
    args: additional arguments for specific discetize method
    """
    def discretize(self, method, **args):
        if method == "hard_threshold":
            threshold = 0.5
            if args.get('threshold') != None:
                threshold = args.get('threshold')
            return self.hard_threshold(threshold)
        elif method == "random_sampling":
            return self.random_sampling()
        elif method == "kmeans":
            return self.kmeans_test()
        elif method == "random_forest":
            if not self.pretrain:
                assert args.get('rf_A') != None and args.get('rf_A_hat') != None

                rf_A = copy.deepcopy(args.get('rf_A'))
                rf_A_hat = copy.deepcopy(args.get('rf_A_hat'))
 
                assert len(rf_A) == len(rf_A_hat)
                
                for i,batch in enumerate(rf_A):
                    rf_A[i] = rf_A[i].cpu().numpy().squeeze(-1)
                    rf_A_hat[i] = rf_A_hat[i].cpu().numpy().squeeze(-1)


                rf_A = np.concatenate(rf_A, axis=0)
                rf_A_hat = np.concatenate(rf_A_hat, axis=0)

                assert rf_A.shape == rf_A_hat.shape
                assert rf_A_hat.shape[-2:] == self.A_hat.shape[-2:]
                
                n_estimators = 1000
                if args.get('n_estimators') != None:
                    n_estimators = args.get('n_estimators')

                return self.random_forest(n_estimators, rf_A, rf_A_hat)
            else:
                assert self.filename != None
                return self.random_forest()
        elif method == "vote_mapping":
            return self.vote_mapping()
        elif method == 'gen_hard_threshold':
            threshold = 0.5
            if args.get('threshold') != None:
                threshold = args.get('threshold')
            return self.gen_hard_threshold(threshold)
        else:
            raise ValueError('Error! Invalid discretize method! Please input one of the following methods:\n \
                              [hard_threshold, random_sampling, random_forest, vote_mapping]')


    """
    discretize the array by a hard threshold with default value equal to 0.5
    to enforce the discretized matrix to be symmetric, for each entry matrix[i][j],
    I set matrix[i][j] = max(matrix[j][i], matrix[i][j]) >= threshold
    """
    def hard_threshold(self, threshold=0.5):
        res = copy.deepcopy(self.A_hat)
        batch_size = self.A_hat.shape[0]
        for i in range(batch_size):
            for j in range(len(self.A_hat[i])):
                for k in range(j, len(self.A_hat[i])):
                    # val = (self.A_hat[i][j][k] + self.A_hat[i][k][j]) / 2
                    val = max(self.A_hat[i][j][k], self.A_hat[i][k][j])
                    if val > threshold:
                        res[i][j][k] = 1
                        res[i][k][j] = 1
                    else:
                        res[i][j][k] = 0
                        res[i][k][j] = 0
        assert res.shape == self.A.shape
        return res

    """
    discretize method used by discretizing matrix outputed by Decoder(z+alpha*w)
    discretize the array by a hard threshold with default value equal to 0.5.
    First, normalize each matrix by (A-A.min())/A.max()
    to enforce the discretized matrix to be symmetric, for each entry matrix[i][j],
    I set matrix[i][j] = max(matrix[j][i], matrix[i][j]) >= threshold
    """
    def gen_hard_threshold(self, threshold=0.5):
        res = copy.deepcopy(self.A_hat)
        batch_size = self.A_hat.shape[0]
        for i,graph in enumerate(self.A_hat):
            temp = graph - graph.min()
            graph = ((temp - temp.min()) / temp.max() >= 0.50).astype(int)
            res[i] = graph
        assert res.shape == self.A.shape
        return res

    """
    for each entry matrix[i][j], if matrix[i][j] = p, then matrix[i][j] will have a probaility
    of p to be equal to 1
    """
    def random_sampling(self):
        res = copy.deepcopy(self.A_hat)
        batch_size = self.A_hat.shape[0]
        for i in range(batch_size):
            res[i] = [[random.random() < x for x in row] for row in self.A_hat[i]]
        assert res.shape == self.A.shape
        return res

    """
    Train a random forest classifier for this discretizing task.

    Param:
    'n_estimators': number of estinmator of the random forest classifier
    'rf_A': the training label for the random forest classifier (can be batched_A_hat)
    'rf_A_hat': the training data for the random forest classifier (can be A_train)
    """
    def random_forest(self, n_estimators=1000, rf_A=None, rf_A_hat=None):
        res = []
        if self.pretrain == True:
            clf = pickle.load(open(self.filename, 'rb'))
        else:
            clf = RandomForestClassifier(n_estimators=n_estimators)
            s = np.shape(rf_A_hat)
            x_train = [[rf_A_hat[z][y][x], rf_A_hat[z][x][y], *[rf_A_hat[z][y][c] for c in range(s[2])], *[rf_A_hat[z][c][y] for c in range(s[1])], *[rf_A_hat[z][x][c] for c in range(s[2])], *[rf_A_hat[z][c][x] for c in range(s[1])]] for x in range(s[2]) for y in range(s[1]) for z in range(s[0])]
            y_train = np.matrix.flatten(rf_A)
            clf.fit(x_train,y_train)
            pickle.dump(clf, open("pretrain_random_forest_model.sav", 'wb'))
            self.pretrain = True
    
        s = self.A_hat.shape
        x_predict = [[self.A_hat[z][y][x], self.A_hat[z][x][y], *[self.A_hat[z][y][c] for c in range(s[2])], *[self.A_hat[z][c][y] for c in range(s[1])], *[self.A_hat[z][x][c] for c in range(s[2])], *[self.A_hat[z][c][x] for c in range(s[1])]] for x in range(s[2]) for y in range(s[1]) for z in range(s[0])]

        res = clf.predict(x_predict)
        res = np.reshape(res, s)

        assert res.shape == self.A.shape
        return res

    """
    A deterministic discretizing method

    for each entry matrix[i][j], we obtain 4 thresholds:
        1.  maxval: max value at row i
            minval: min value at row i
            threshold1 = avg(maxval, minval) 
        2.  colmaxval: max value at col j
            colminval: min value at col j
            threshold2 = avg(colmaxval, colminval) 
        3.  inverse_maxval: max value at row j
            inverse_minval: min value at row j
            threshold3 = avg(inverse_maxval, inverse_minval) 
        4.  inverse_colmaxval: max value at col i
            inverse_colminval: min value at col i
            threshold4 = avg(inverse_colmaxval, inverse_colminval)
    for each threshold, we can get 1 result whether matrix[i][j] should be 1

    for the 4 results we get for the current entry, we take a vote. If # of 1 > # of 0
    matrix[i][j] = matrix[j][i] = 1 and vise versa
    """
    def vote_mapping(self):
        res = np.zeros(self.A_hat.shape)
        batch_size = self.A_hat.shape[0]

        for batch_num in range(batch_size):
            for i,row in enumerate(self.A_hat[batch_num]):
                maxval = max(row)
                minval = min(row)
                threshold = (maxval + minval) / 2

                for j in range(i, len(row)):
                    colmaxval = max(self.A_hat[batch_num][:][j])
                    colminval = min(self.A_hat[batch_num][:][j])
                    colthreshold = (colmaxval + colminval) / 2
                    inverse_maxval = max(self.A_hat[batch_num][j])
                    inverse_minval = min(self.A_hat[batch_num][j])
                    inverse_threshold = (inverse_maxval + inverse_minval) / 2
                    inverse_colmaxval = max(self.A_hat[batch_num][:][i])
                    inverse_colminval = min(self.A_hat[batch_num][:][i])
                    inverse_colthreshold = (inverse_colmaxval + inverse_colminval) / 2

                    vote = []
                    vote.append(self.A_hat[batch_num][i][j] >= threshold)
                    vote.append(self.A_hat[batch_num][i][j] >= colthreshold)
                    vote.append(self.A_hat[batch_num][j][i] >= inverse_threshold)
                    vote.append(self.A_hat[batch_num][j][i] >= inverse_colthreshold)
                    if sum(vote) >= 2:
                        res[batch_num][i][j] = 1
                        res[batch_num][j][i] = 1
                    else:
                        res[batch_num][i][j] = 0
                        res[batch_num][j][i] = 0

            real_num_of_node = int(self.A_hat.shape[1])
            for i in range(real_num_of_node):
                if res[batch_num][i][i] != 1:
                    real_num_of_node = i
                    break
            valid_subgraph = copy.deepcopy(res[batch_num][:real_num_of_node, :real_num_of_node])
            res[batch_num] = np.zeros(self.A_hat.shape[1:])
            res[batch_num][:real_num_of_node, :real_num_of_node] = valid_subgraph
        assert res.shape == self.A.shape
        return res

    def kmeans_test(self):
        res = np.zeros(self.A_hat.shape)
        batch_size = self.A_hat.shape[0]
        node_num = self.A_hat.shape[1]
        
        for i in range(batch_size):
            #print(self.A_hat[i])
            for j in range(node_num):
                
                row_size = len(self.A_hat[i][j][self.A_hat[i][j] != np.float32(0.01)])
                if row_size <= 1:
                    continue
                kmeans = KMeans(n_clusters = 2).fit(self.A_hat[i][j][self.A_hat[i][j] != np.float32(0.01)].reshape(-1, 1))
                ## 1 cluster
                if kmeans.cluster_centers_.shape[0] == 1:
                    if kmeans.cluster_centers_[0] > 0.25:
                        res[i][j][self.A_hat[i][j] != np.float32(0.01)] = np.ones(row_size)
                    else:
                        res[i][j][self.A_hat[i][j] != np.float32(0.01)] = np.zeros(row_size)
                elif abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) < 0.15:
                    if (kmeans.cluster_centers_[0] + kmeans.cluster_centers_[1]) / 2 > 0.25:
                        res[i][j][self.A_hat[i][j] != np.float32(0.01)] = np.ones(row_size)
                    else:
                        res[i][j][self.A_hat[i][j] != np.float32(0.01)] = np.zeros(row_size)
                else:
                    if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
                        #print(np.ones(row_size) - kmeans.predict(self.A_hat[i][j][self.A_hat[i][j] != np.float32(0.01)].reshape(-1, 1)))
                        res[i][j][self.A_hat[i][j] != np.float32(0.01)] = np.ones(row_size) - kmeans.predict(self.A_hat[i][j][self.A_hat[i][j] != np.float32(0.01)].reshape(-1, 1))
                    else:
                        res[i][j][self.A_hat[i][j] != np.float32(0.01)] = kmeans.predict(self.A_hat[i][j][self.A_hat[i][j] != np.float32(0.01)].reshape(-1, 1))
            #print(res[i])
        assert res.shape == self.A.shape
        return res

    def kmeans(self):
        res = np.zeros(self.A_hat.shape)
        batch_size = self.A_hat.shape[0]
        node_num = self.A_hat.shape[1]
    
        for i in range(batch_size):
            graph_size = node_num
            #print(self.A_hat[i])
            try:
                graph_size = list(self.A_hat[i][0]).index(np.float32(0.01))
            except:
                pass

            if graph_size == 1:
                res[i][0][0] = 1
                continue
            elif graph_size == 0:
                continue

        #print(graph_size)
            for j in range(graph_size):
                kmeans = KMeans(n_clusters = 2).fit(self.A_hat[i][j][0:graph_size].reshape(-1, 1))
            ## 1 cluster
                if kmeans.cluster_centers_.shape[0] == 1:
                    if kmeans.cluster_centers_[0] > 0.5:
                        res[i][j][0:graph_size] = np.ones(graph_size)
                    else:
                        res[i][j][0:graph_size] = np.zeros(graph_size)
                elif abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) < 0.15:
                    if (kmeans.cluster_centers_[0] + kmeans.cluster_centers_[1]) / 2 > 0.5:
                        res[i][j][0:graph_size] = np.ones(graph_size)
                    else:
                        res[i][j][0:graph_size] = np.zeros(graph_size)
                else:
                    if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
                        res[i][j][0:graph_size] = np.ones(graph_size) - kmeans.predict(self.A_hat[i][j][0:graph_size].reshape(-1, 1))
                    else:
                        kmeans.predict(self.A_hat[i][j][0:graph_size].reshape(-1, 1))
        #print(res[i])
        assert res.shape == self.A.shape
        return res
       


