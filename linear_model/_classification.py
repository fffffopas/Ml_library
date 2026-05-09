import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator

class LogisticRegression(RegressorMixin, BaseEstimator):
    def __init__(self, max_iter=1000, batch_size=64, lr=0.1):
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr

    def _sigmoid(self, X: np.ndarray):
        Z = np.clip(X, -100, 100)
        return 1/(1 + np.exp(-Z))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        N, D = X.shape

        w = np.random.randn(D + 1, 1) * 0.01
        b_to_x = np.ones((N,1))
        X_b = np.hstack([X, b_to_x])

        for i in range(self.max_iter):
            random_shuffle = np.random.permutation(N)
            X_b = X_b[random_shuffle]
            y = y[random_shuffle]
            pred_coef = w.copy()

            for j in range(0, N, self.batch_size):
                X_batch = X_b[j: j + self.batch_size]
                y_batch = y[j: j + self.batch_size]

                w = w - self.lr * ((1/len(X_batch)) * X_batch.T @(self._sigmoid(X_batch @ w) - y_batch))

            if (np.linalg.norm(w - pred_coef) <= 1e-15):
                break

        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        
        return self
    
    def predict_proba(self, X):
        pred = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.hstack([1 - pred, pred])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class OvRLogisticRegression(RegressorMixin, BaseEstimator):
    def __init__(self, max_iter=1000, batch_size=64, lr=0.1):
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y):
        self.models = list()
        self.unique = np.unique(y)

        for unique_object in self.unique:
            y_object = (y == unique_object).astype(int)
            log_model = LogisticRegression()

            log_model.fit(X, y_object)
            self.models.append(log_model)

        return self

    def predict_proba(self, X):
        return np.array([model.predict_proba(X)[:, 1] for model in self.models]).T
    
    def predict(self, X):
        return self.unique[(np.argmax(self.predict_proba(X), axis=1)).astype(int)]

