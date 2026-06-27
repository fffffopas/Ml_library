from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from itertools import product
from sklearn.model_selection import KFold
from joblib import delayed, Parallel

import numpy as np
import sklearn

class GridSearchCV(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator: BaseEstimator, params_grid: dict, cv: int=5, scoring: str="r2", random_state: int=101, n_jobs: int=1):
        
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.params_grid = params_grid
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        y = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y не преобразутеся в np.ndarray")

        dict_scoring = {"r2": r2_score,
                        "neg_mae" : mean_absolute_error,
                        "neg_mse" : mean_squared_error,
                        "neg_rmse" : root_mean_squared_error}
        dict_sign = {"r2": 1,
                        "neg_mae" : -1,
                        "neg_mse" : -1,
                        "neg_rmse" : -1}
        
        self.best_score_ = -float("inf")
        self.best_params_ = None

        metric = dict_scoring[self.scoring]
        sign = dict_sign[self.scoring]

        keys = self.params_grid.keys()
        values = self.params_grid.values()
        grid = [dict(zip(keys, value)) for value in product(*values)]
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        def k_train(kf, X, y, estimator, value, metric, sign):
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

        list_metrics_params = Parallel(n_jobs=self.n_jobs)(delayed(k_train)(kf, X, y, self.estimator, value, metric, sign)
                                                           for value in grid)
        for mean_metric_k, value in list_metrics_params:
            if mean_metric_k > self.best_score_:
                self.best_score_ = mean_metric_k
                self.best_params_ = value

        self._refit(X, y)

    def _refit(self, X: np.ndarray, y: np.ndarray):
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)

        self.best_estimator_.fit(X, y)

    def predict(self, X: np.ndarray):
        sklearn.utils.validation.check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict(X)