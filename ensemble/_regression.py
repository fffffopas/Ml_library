import numpy as np
import pandas as pd
from tree import DecisionTreeRegressor
from sklearn.base import RegressorMixin, BaseEstimator
from joblib import delayed, Parallel

class RandomForestRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_estimators=100, bootstrap=True, max_samples=1, random_state=42,
                 max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", min_impurity_decrease=1e-12):
        
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

    def _converter_to_numpy(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.to_numpy()     
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.asarray(X)

    def _fit_tree(self, X, y, tree_params, seed):
        if self.bootstrap:
            _rng = np.random.default_rng(seed)
            bootstrap_index = _rng.choice(X.shape[0], int(X.shape[0] * self.max_samples), replace=True)
            tree = DecisionTreeRegressor(**tree_params, random_state=seed)
            tree.fit(X[bootstrap_index], y[bootstrap_index])
            return tree
        else:
            tree = DecisionTreeRegressor(**tree_params, random_state=seed)
            tree.fit(X, y)
            return tree
    
    def _predict_by_tree(self, X, tree):
        return tree.predict(X)

    def fit(self, X, y):
        X_ = self._converter_to_numpy(X)
        y_ = self._converter_to_numpy(y)
        seeds_list = []

        for i in range(self.n_estimators):
            seeds_list.append(self.random_state + i)

        tree_params = {
        "max_depth": self.max_depth,
        "min_samples_split": self.min_samples_split,
        "min_samples_leaf": self.min_samples_leaf,
        "max_features": self.max_features,
        "min_impurity_decrease": self.min_impurity_decrease,
        }

        self._trees = Parallel(n_jobs=-1)(delayed(self._fit_tree)(X_, y_, tree_params, s) for s in seeds_list)
        return self
    
    def predict(self, X):
        X_ = self._converter_to_numpy(X)
        answers = Parallel(n_jobs=-1)(delayed(self._predict_by_tree)(X_, tree) for tree in self._trees)
        answers = np.array(answers)
        result = answers.mean(axis=0)
        return result

class GradientBoostingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, max_features="sqrt", min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _converter_to_numpy(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.to_numpy()     
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.asarray(X)
    
    def fit(self, X, y):
        X_ = self._converter_to_numpy(X)
        y_ = self._converter_to_numpy(y)
        tree_params = {
            "max_depth" : self.max_depth,
            "max_features" : self.max_features,
            "min_samples_split" : self.min_samples_split,
            "min_samples_leaf" : self.min_samples_leaf,
            "random_state" : self.random_state
        }

        self._estimators = []
        self._F_0 = np.mean(y_)
        _temp_predictions = np.full(y_.shape[0], fill_value=self._F_0)

        for i in range(self.n_estimators):
            _temp_residuals = y_ - _temp_predictions

            tree = DecisionTreeRegressor(**tree_params)
            tree.fit(X_, _temp_residuals)
            self._estimators.append(tree)

            _temp_predictions += self.learning_rate * tree.predict(X_)

        return self
    
    def predict(self, X):
        X_ = self._converter_to_numpy(X)
        result = np.full(X_.shape[0], fill_value=self._F_0)

        for tree in self._estimators:
            result += self.learning_rate * tree.predict(X_)

        return result