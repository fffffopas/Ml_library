import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator
from joblib import delayed, Parallel

class RandomForestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=100, bootstrap=True, max_samples=1, random_state=42,
                 criterion="gini", max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", min_impurity_decrease=1e-12):
        
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self.criterion = criterion
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
            tree = DecisionTreeClassifier(**tree_params, random_state=seed)
            tree.fit(X[bootstrap_index], y[bootstrap_index])
            return tree
        else:
            tree = DecisionTreeClassifier(**tree_params, random_state=seed)
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
        "criterion": self.criterion,
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
        result = np.zeros(X_.shape[0], dtype=np.int32)
        for i in range(X_.shape[0]):
            result[i] = np.argmax(np.bincount(answers[:, i]))
        return result

class GradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, max_features="sqrt", min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def _create_leaf_function(self, _temp_predictions):
        p = self._sigmoid(_temp_predictions)
        def get_leaf_value(y, index_sample):
            return np.sum(y[index_sample])/ np.sum(p[index_sample]*(1-p[index_sample]))
        
        return get_leaf_value

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
        y_mean = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        self._F_0 = np.log(y_mean/(1 - y_mean))
        _temp_predictions = np.full(y_.shape[0], fill_value=self._F_0)

        for i in range(self.n_estimators):
            _temp_residuals = y_ - self._sigmoid(_temp_predictions)
            _temp_function = self._create_leaf_function(_temp_predictions)

            tree = DecisionTreeRegressor(**tree_params, leaf_function=_temp_function)
            tree.fit(X_, _temp_residuals)
            self._estimators.append(tree)

            _temp_predictions += self.learning_rate * tree.predict(X_)

        return self
    
    def predict_proba(self, X):
        X_ = self._converter_to_numpy(X)
        result = np.full(X_.shape[0], fill_value=self._F_0)

        for tree in self._estimators:
            result += self.learning_rate * tree.predict(X_)

        return self._sigmoid(result)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    

def GradientBoostingMutliClassification(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, max_features="sqrt", min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _softmax(self, F):
        F_shifted = F - F.max(axis=1, keepdims=True)
        exp_F = np.exp(F_shifted)
        return exp_F / exp_F.sum(axis=1, keepdims=True)

    def _create_leaf_function(self, p):
        def get_leaf_value(y, index_sample):
            return np.sum(y[index_sample])/ np.sum(p[index_sample]*(1-p[index_sample]))
        
        return get_leaf_value

    def _converter_to_numpy(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.to_numpy()     
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.asarray(X)

    def _fit_one_tree(self, X, y, tree_params, p):
        _temp_function = self._create_leaf_function(p)
        tree = DecisionTreeRegressor(**tree_params, leaf_function=_temp_function)
        tree.fit(X, y)
        return tree
    
    def _predict_one_tree(self, X, tree):
        return tree.predict(X)
    
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
        self.unique_label, counts = np.unique(y_, return_counts=True)
        self._estimators = [[] for i in range(self.unique_label.shape[0])]
        self._F_0 = np.log(counts/len(y))
        Y_onehot = (y_[:, None] == self.unique_label[None, :]).astype(int)
        _temp_predictions = np.tile(self._F_0, (y_.shape[0], 1))

        for i in range(self.n_estimators):
            p = self._softmax(_temp_predictions)
            _temp_residuals = Y_onehot - p
            _trees = Parallel(n_jobs=-1)(delayed(self._fit_one_tree)(X_, _temp_residuals[:, k], tree_params, p[:, k]) for k in range(len(self.unique_label)))
            self._estimators[i] = _trees

            _temp_predictions += self.learning_rate * np.array(Parallel(n_jobs=-1)(delayed(self._predict_one_tree)(X_, _trees[k]) for k in range(len(self.unique_label)))).T

        return self
    
    def predict_proba(self, X):
        X_ = self._converter_to_numpy(X)
        result = np.tile(self._F_0, (X_.shape[0], 1))

        for i in range(self.n_estimators):
            result += self.learning_rate * np.array(Parallel(n_jobs=-1)(delayed(self._predict_one_tree)(X_, self._estimators[i][k]) for k in range(len(self.unique_label)))).T

        return self._softmax(result)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)        
        return self.unique_label[class_indices]   