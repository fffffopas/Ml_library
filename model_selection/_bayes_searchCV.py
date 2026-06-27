from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
from gaussian_processes import GaussianProcess
from joblib import delayed, Parallel

import scipy.stats
import numpy as np
import sklearn

class BayesSearchCV(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator: BaseEstimator, params_space: dict, cv: int=5, scoring: str="r2", random_state: int=101, n_iter: int=10, start_iter: int=5, n_jobs: int=1):
        
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.params_space = params_space
        self.random_state = random_state
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.n_jobs = n_jobs

    def preprocess(self, X: np.ndarray, y: np.ndarray):

        def k_train(kf, X: np.ndarray, y :np.ndarray, estimator: BaseEstimator, value: dict, metric, sign: int):
            list_metrics = []
            for index_train, index_val in kf.split(X):
                X_train, X_val = X[index_train], X[index_val]
                y_train, y_val = y[index_train], y[index_val]

                model = clone(estimator)
                model.set_params(**value)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                metric_k = sign * metric(y_val, y_pred)

                list_metrics.append(metric_k)

            mean_metric_k = sum(list_metrics)/len(list_metrics)
            return mean_metric_k, value

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.list_param = list()
        self.list_metric = list()
        self.best_score_ = -float("inf")
        self.best_params_ = None
        self.int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'max_leaves']

        grid = []
        for i in range(self.start_iter):
            value = dict()

            for key, dist in self.params_space.items():
                if hasattr(dist, "rvs"):
                    value[key] = float(dist.rvs(random_state=(self.random_state + i)))
                else:
                    np.random.seed(self.random_state + i)
                    value[key] = np.random.choice(dist)

                if key in self.int_params:
                    value[key] = int(np.round(value[key]))

            grid.append(value)

            
        list_metrics_params = Parallel(n_jobs=self.n_jobs)(delayed(k_train)(kf, X, y, self.estimator, value, self.metric, self.sign) 
                                                           for value in grid)

        for mean_metric_k, value in list_metrics_params:
            if mean_metric_k > self.best_score_:
                self.best_score_ = mean_metric_k
                self.best_params_ = value

            self.list_param.append(list(value.values()))
            self.list_metric.append(mean_metric_k)
            
    def _expected_improvement(self, M_pred: np.ndarray, xi: float=0.01):
        mu = np.array(M_pred[0]).ravel()
        std = np.array(M_pred[1]).ravel()

        ei = np.zeros_like(mu)
    
        mask = std > 1e-9 
        imp = mu[mask] - self.best_score_ - xi
        z = imp / std[mask]
    
        ei[mask] = imp * scipy.stats.norm.cdf(z) + std[mask] * scipy.stats.norm.pdf(z)
        return ei
    
    def fit(self, X: np.ndarray, y:np.ndarray):
        X = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        y = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y не преобразуется в np.ndarray")
        
        dict_scoring = {"r2": r2_score,
                        "neg_mae" : mean_absolute_error,
                        "neg_mse" : mean_squared_error,
                        "neg_rmse" : root_mean_squared_error}
        dict_sign = {"r2": 1,
                        "neg_mae" : -1,
                        "neg_mse" : -1,
                        "neg_rmse" : -1}
        
        self.metric = dict_scoring[self.scoring]
        self.sign = dict_sign[self.scoring]
        self.preprocess(X, y)

        def fit_k(estimator: BaseEstimator, index_train: np.ndarray, index_val: np.ndarray, X: np.ndarray, y: np.ndarray, value: dict):
            X_train, X_val = X[index_train], X[index_val]
            y_train, y_val = y[index_train], y[index_val]

            model = clone(estimator)
            model.set_params(**value)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            metric_k = self.sign * self.metric(y_val, y_pred)

            return metric_k

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for i in range(self.n_iter):
            surr_model = GaussianProcess()
            scaler = StandardScaler()
            P_train = np.array(self.list_param)
            M_train = np.array(self.list_metric)

            P_train = scaler.fit_transform(P_train)
            surr_model.fit(P_train, M_train)

            P_test = list()
            for j in range(1000):
                one_item = list()
                for key, dist in self.params_space.items():
                    if hasattr(dist, "rvs"):
                        one_item.append(float(dist.rvs(random_state=(self.random_state + j + 1000 * i))))
                    else:
                        np.random.seed(self.random_state + iter)
                        one_item.append(np.random.choice(dist))

                    if key in self.int_params:
                        one_item[-1] = int(np.round(one_item[-1]))
                
                P_test.append(one_item)
            #print(P_test)
            P_test_ = scaler.transform(np.array(P_test))
            M_pred = surr_model.predict(P_test_)
            #print(M_pred[0].shape, M_pred[1].shape)
            #print(M_pred)
            ei = self._expected_improvement(M_pred)

            best_param = P_test[np.argmax(ei)]
            value = dict(zip(self.params_space.keys(), list(best_param)))

            list_metrics = Parallel(n_jobs=self.n_jobs)(delayed(fit_k)(self.estimator, index_train, index_val, X, y, value)
                                                        for index_train, index_val in kf.split(X))
            
            mean_metric_k = sum(list_metrics)/len(list_metrics)
            if mean_metric_k > self.best_score_:
                self.best_score_ = mean_metric_k
                self.best_params_ = value

            self.list_param.append(list(value.values()))
            self.list_metric.append(mean_metric_k)

        self._refit(X, y)

    def _refit(self, X: np.ndarray, y: np.ndarray):
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)

        self.best_estimator_.fit(X, y)

    def predict(self, X: np.ndarray):
        sklearn.utils.validation.check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict(X)