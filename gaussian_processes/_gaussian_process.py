from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.base import BaseEstimator, MetaEstimatorMixin

import numpy as np
import sklearn

class GaussianProcess(BaseEstimator):
    def __init__(self, sigma_f: float=1.0, sigma_n: float=1e-2):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
    
    def kernel(self, X1: np.ndarray, X2: np.ndarray, l: float=1.0):
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1)

            pred_result = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            result = np.exp(-(pred_result)/ (2 * l**2))
            return result
    
    def fit(self, X, y):
        
        K_train_train = self.kernel(X, X)
        self._K = np.linalg.inv((K_train_train + np.eye(len(X)) * self.sigma_n))
        self._X = X
        self._y = y

    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, ["_K", "_X", "_y"])
        K_test_train = self.kernel(X, self._X)
        K_test_test = self.kernel(X, X)

        mu = K_test_train @ self._K @ self._y
        std = np.diag(K_test_test) - np.sum((K_test_train @ self._K) * K_test_train, axis=1)
        return [mu, std]