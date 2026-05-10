import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator

class LinearRegression(RegressorMixin, BaseEstimator):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        N, D = X.shape
        b = np.ones((N, 1))
        X_b = np.hstack([X, b])

        w = np.linalg.pinv(X_b) @ y

        self.coef_ = w[:-1]
        self.intercept_ = w[-1]

        return self
    
    def predict(self, X):

        return X @ self.coef_ + self.intercept_


