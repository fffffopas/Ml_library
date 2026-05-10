import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator

class SVC_BASE(ClassifierMixin, BaseEstimator):
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

class SVC(ClassifierMixin, BaseEstimator):
    def __init__(self, max_iter=1000, C=10.0, tol=1e-3, kernel='rbf', gamma=5):

        self.alpha = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.C = C
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma

    def _rbf_kernel(self, X, X_):
        dists = np.sum(X**2, axis=1).reshape(-1, 1) - 2 *(X @ X_.T) + np.sum(X_**2, axis=1).reshape(1, -1)
        return np.exp(-self.gamma * dists)

    def _get_kernel(self, X, X_):
        if self.kernel == "linear":
            return X @ X_.T

        if self.kernel == "rbf":
            return self._rbf_kernel(X,  X_)


    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).ravel()

        N, D = self.X.shape
        self.alpha = np.zeros(N)
        self.intercept_ = 0.0
        K = self._get_kernel(self.X, self.X)

        def select_random_j(i, N):
            j = i
            while j == i:
                j = np.random.randint(0, N)
            return j

        for _ in range(self.max_iter):
            for i in range(N):
                E = np.sum(self.alpha * y.ravel() * K[:, i]) + self.intercept_ - self.y[i]

                if ((E * self.y[i] < -self.tol) and (self.alpha[i] < self.C)) or ((E * self.y[i] > self.tol) and (self.alpha[i] > 0)):
                    
                    j = select_random_j(i, N)
                    eta = 2.0 * K[i, j] - K[i,i] - K[j, j]
                    if eta >= 0: continue

                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H: continue 

                    E_j = np.sum(self.alpha * y.ravel() * K[:, j]) + self.intercept_ - self.y[j]
                    alpha_j_old = self.alpha[j]
                    self.alpha[j] = self.alpha[j] - self.y[j] * (E - E_j)/eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    alpha_i_old = self.alpha[i]
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.intercept_ - E - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - self.y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.intercept_ - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - self.y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.intercept_ = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.intercept_ = b2
                    else:
                        self.intercept_ = (b1 + b2) / 2

        return self

    def predict(self, X):
        
        return np.sign(self._get_kernel(X, self.X) @ (self.alpha * self.y.ravel()) + self.intercept_)
