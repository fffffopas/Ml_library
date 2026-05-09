import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator

class SVC(RegressorMixin, BaseEstimator):
    def __init__(self, max_iter=1000, batch_size=64, lr=0.1, C=1.0):

        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.C = C

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        N, D = X.shape
        self.coef_ = np.random.randn(D,1) * 0.01
        self.intercept_ = 1.0

        for i in range(self.max_iter):
            random_shuffle = np.random.permutation(N)
            X = X[random_shuffle]
            y = y[random_shuffle]

            pred_coef = self.coef_.copy()

            for j in range(0, N, self.batch_size):
                X_batch = X[j: j + self.batch_size]
                y_batch = y[j: j + self.batch_size]

                mask = ((y_batch *(X_batch @ self.coef_ + self.intercept_)) < 1).ravel()

                self.coef_ = self.coef_ - self.lr * (self.coef_ - self.C * X_batch[mask].T @ y_batch[mask])
                self.intercept_ = self.intercept_ + self.lr * (self.C * np.sum(y_batch[mask]))

            if np.linalg.norm(self.coef_ - pred_coef) < 1e-15:
                break

        return self

    def predict(self, X):

        return np.sign(X @ self.coef_ + self.intercept_)
