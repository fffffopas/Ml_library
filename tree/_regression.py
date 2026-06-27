import numpy as np
import pandas as pd
from .__auxiliary_tree import Node
from sklearn.base import RegressorMixin, BaseEstimator

class DecisionTreeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", random_state=42, min_impurity_decrease=1e-12, min_var=1e-12, leaf_function=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_var = min_var
        self.leaf_function = leaf_function

    def _get_leaf_value(self, y, index_sample):
        if self.leaf_function is not None:
            return self.leaf_function(y, index_sample)
        
        return np.mean(y[index_sample])
    
    def _get_criterion_estimate(self, sum_y, sum_sq_y, n):
        return sum_sq_y/n - (sum_y/n) ** 2
    
    def _get_count_features(self, total_count):
        if isinstance(self.max_features, int):
            return self.max_features
        list_func = {"sqrt":np.sqrt, "log2":np.log2}
        return int(list_func[self.max_features](total_count))
    
    def _converter_to_numpy(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.to_numpy()     
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.asarray(X)

    def _build_tree(self, X, y, node, depth, root_criterion, sorted_index, index_sample):
        if(len(y[index_sample]) < self.min_samples_split or depth >= self.max_depth or root_criterion < self.min_var):
            node.val = self._get_leaf_value(y, index_sample)
            return 

        best_split = [-float("inf")] * 5

        for i in self._rng.choice(X.shape[1], size=self._total_features, replace=False):
            sorted_sample_index = sorted_index[:, i][np.isin(sorted_index[:, i], index_sample)] 
            unique_values, index_box = np.unique(X[sorted_sample_index, i], return_index=True)
            tresholds = (unique_values[:-1] + unique_values[1:])/2

            sum_left = 0
            sum_sq_left = 0
            size_left = 0

            sum_right = np.sum(y[index_sample])
            sum_sq_right = np.sum(y[index_sample] ** 2)
            size_right = y[index_sample].shape[0]
            y_sort = y[sorted_sample_index]

            for j in range(len(index_box)-1):
                y_box = y_sort[index_box[j]:index_box[j+1]]
                sum_y_box = np.sum(y_box)
                sum_y_sq_box = np.sum(y_box ** 2)

                sum_left += sum_y_box
                sum_sq_left += sum_y_sq_box
                size_left += y_box.shape[0]

                sum_right -= sum_y_box
                sum_sq_right -= sum_y_sq_box
                size_right -= y_box.shape[0]

                if(size_left < self.min_samples_leaf or size_right < self.min_samples_leaf):
                    continue

                left_criterion = self._get_criterion_estimate(sum_left, sum_sq_left, size_left)
                right_criterion = self._get_criterion_estimate(sum_right, sum_sq_right, size_right)
                IG = root_criterion - size_left/y_sort.shape[0] * left_criterion - size_right/y_sort.shape[0] * right_criterion

                if IG > best_split[0] and IG > self.min_impurity_decrease:
                    best_split = [IG, i, tresholds[j], left_criterion, right_criterion]

        if (best_split[0] == -float("inf")):
            node.val = self._get_leaf_value(y, index_sample)
            return 

        node.tresh = best_split[2]
        node.index_feature = best_split[1]
        left_node = Node()
        right_node = Node()

        mask = X[index_sample, best_split[1]] <= best_split[2]

        node.left = left_node
        node.right = right_node

        self._build_tree(X, y, left_node, depth+1, best_split[3], sorted_index, index_sample[mask])
        self._build_tree(X, y, right_node, depth+1, best_split[4], sorted_index, index_sample[~mask])

    def fit(self, X, y):
        X_ = self._converter_to_numpy(X)
        y_ = self._converter_to_numpy(y)

        self._rng = np.random.default_rng(self.random_state)
        self._total_features = self._get_count_features(X_.shape[1])

        sorted_index = np.argsort(X_, axis=0)
        index_sample = np.arange(0, X_.shape[0])
        root_criterion = self._get_criterion_estimate(np.sum(y_), np.sum(y_ ** 2), y_.shape[0])

        self.node = Node()
        self._build_tree(X_, y_, self.node, 1, root_criterion, sorted_index, index_sample)

        return self

    def _go_by_tree(self, x, node):
        if node._is_leaf():
            return node.val
        if x[node.index_feature] <= node.tresh:
            return self._go_by_tree(x, node.left)
        return self._go_by_tree(x, node.right)


    def predict(self, X):
        X_ = self._converter_to_numpy(X)

        result = np.zeros(len(X_), dtype=np.float64)
        for i in range(len(X_)):
            result[i] = self._go_by_tree(X_[i, :], self.node)

        return result
