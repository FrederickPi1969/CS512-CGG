import numpy as np
import pickle, copy, random
from sklearn.ensemble import RandomForestClassifier

class Discretizer(object):
    def __init__(self, A, A_hat, pretrain=False, filename=None): ## A_hat is continuous adi matrix
        assert A_hat.shape == A.shape
        assert len(A.shape) == 3 and A.shape[1] == A.shape[2]
        self.A = A
        self.A_hat = A_hat
        self.pretrain = pretrain
        self.filename = filename

    def discretize(self, method, **args):
        if method == "hard_threshold":
            threshold = 0.4
            if args.get('threshold') != None:
                threshold = args.get('threshold')
            return self.hard_threshold(threshold)
        elif method == "random_sampling":
            return self.random_sampling()
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
        else:
            raise ValueError('Error! Invalid discretize method! Please input one of the following methods:\n \
                              [hard_threshold, random_sampling, random_forest, vote_mapping]')

    def hard_threshold(self, threshold=0.4):
        res = copy.deepcopy(self.A_hat)
        batch_size = self.A_hat[0].shape[0]
        for i in range(batch_size):
            res[i][self.A_hat[i] > threshold] = 1
            res[i][self.A_hat[i] <= threshold] = 0
        assert res.shape == self.A.shape
        return res

    def random_sampling(self):
        res = copy.deepcopy(self.A_hat)
        batch_size = self.A_hat[0].shape[0]
        for i in range(batch_size):
            res[i] = [[random.random() < x for x in row] for row in self.A_hat[i]]
        assert res.shape == self.A.shape
        return res

    def random_forest(self, n_estimators=1000, rf_A=None, rf_A_hat=None):
        res = []
        batch_size = self.A_hat[0].shape[0]
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
       

