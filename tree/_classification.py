import numpy as np
from .__auxiliary_tree import Node
from sklearn.base import ClassifierMixin, BaseEstimator

class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, criterion="gini", max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=1, random_state=42, min_impurity_decrease=1e-12):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    def _gini(self, _count_classes):
        _gini_estimate = 1 - np.sum((_count_classes) ** 2)
        return _gini_estimate

    def _entropy(self, _count_classes, eps=1e-12):
        _entropy_estimate = -1 * np.sum(_count_classes * np.log2(_count_classes + eps))
        return _entropy_estimate

    def _get_criterion_estimate(self, _count_classes):
        if self.criterion == "gini":
            return self._gini(_count_classes)
        
        return self._entropy(_count_classes)
    
    def _build_tree(self, X, y, node, depth, root_criterion, sorted_index, index_sample):
        if(len(y[index_sample]) < self.min_samples_split or depth >= self.max_depth or np.unique(y[index_sample]).shape[0] == 1):
            node.val = np.argmax(np.bincount(y[index_sample]))
            return 

        best_split = [-float("inf")] * 5

        for i in range(X.shape[1]):
            sorted_sample_index = sorted_index[:, i][np.isin(sorted_index[:, i], index_sample)] ## вот тут надо придумать что нибудь
            unique_values, index_box = np.unique(X[sorted_sample_index, i], return_index=True)
            tresholds = (unique_values[:-1] + unique_values[1:])/2

            count_classes_left = np.zeros(self.count_unique_classes)
            size_left = 0

            count_classes_right = np.bincount(y[index_sample], minlength=self.count_unique_classes)
            size_right = y[index_sample].shape[0]
            y_sort = y[sorted_sample_index]

            for j in range(len(index_box)-1):
                box = np.bincount(y_sort[index_box[j] :index_box[j+1]], minlength=self.count_unique_classes)
                count_classes_left += box
                size_left += np.sum(box) 

                count_classes_right -= box
                size_right -= np.sum(box)

                if(size_left < self.min_samples_leaf or size_right < self.min_samples_leaf):
                    continue

                left_criterion = self._get_criterion_estimate(count_classes_left/size_left)
                right_criterion = self._get_criterion_estimate(count_classes_right/size_right)
                IG = root_criterion - size_left/y_sort.shape[0] * left_criterion - size_right/y_sort.shape[0] * right_criterion

                if IG > best_split[0] and IG > self.min_impurity_decrease:
                    best_split = [IG, i, tresholds[j], left_criterion, right_criterion]

        if (best_split[0] == -float("inf")):
            node.val = np.argmax(np.bincount(y[index_sample]))
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
        X_ = X.values  
        y_ = y.values
        self.count_unique_classes = len(np.unique(y_))
        sorted_index = np.argsort(X_, axis=0)
        index_sample = np.arange(0, X_.shape[0])

        root_criterion = self._get_criterion_estimate(np.bincount(y_, minlength=self.count_unique_classes)/y_.shape[0])
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
        X_ = X.values 
        result = np.zeros(len(X_), dtype=np.int32)
        for i in range(len(X_)):
            result[i] = self._go_by_tree(X_[i, :], self.node)

        return result
