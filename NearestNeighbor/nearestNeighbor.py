#Reference Andrej Karpathy

#Date 5/23/2023

#My interpretation/summary

#    Nearest neighbors saves a set of training values
#    subsequently, it compares a test value to each and every stored training
#    value. (In this implementation) It uses the 'L1' comparison or "Manhattan
#    distance" to select the closest training value as compared to the test
#    value. In this way, the algorithm scales linearly as more training values
#    are added


import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeroes(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]