import numpy as np
from scipy.stats import mode
from sklearn.base import ClassifierMixin, BaseEstimator

class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1, 1)

        return self
    
    def predict(self, X):
        dists = np.sum(X**2, axis=1).reshape(-1, 1) - 2 *(X @ self.X.T) + np.sum(self.X**2, axis=1).reshape(1, -1)
        dists = np.sqrt(dists)

        k_index = np.argsort(dists, axis=1)[:, :self.k]
        y_k_index = self.y.ravel()[k_index]
        result, count = mode(y_k_index, axis=1)

        return result





