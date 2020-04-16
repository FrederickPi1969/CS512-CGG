class Discretizer(object):
    def __init__(self, A, A_hat, pretrain=False): ## A_hat is continuous adj matrix
        self.A = A
        self.A_hat = A_hat
        self.pretrain = pretrain

    def discretize(self, method):
        if method == "hard_threshold": pass
        if method == "random_sampling": pass
        if method == "random_forest": pass

    def hard_threshold(self, threshold=0.4): pass

    def random_sampling(self): pass

    def random_forest(self):
        if self.pretrain: pass
        else : pass



