import numpy as np


class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        info_gain=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf nodes
        self.value = value


class DecisionTreeModel:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    # fit the model to the data
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    # predict the label for a given example
    def predict(self, X):
        return np.array([self._predict(example, self.root) for example in X])

    # grow the tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # greedily select the best split according to information gain
        best_feature, best_threshold, best_info_gain = self._best_split(
            X, y, n_samples, n_features
        )

        # grow the children that result from the split
        left_X, left_y, right_X, right_y = self._split(
            X, y, best_feature, best_threshold
        )
        left_child = self._grow_tree(left_X, left_y, depth + 1)
        right_child = self._grow_tree(right_X, right_y, depth + 1)
        return Node(
            best_feature, best_threshold, left_child, right_child, best_info_gain
        )

    # find the best split
    def _best_split(self, X, y, n_samples, n_features):
        # best criteria
        best_feature = None
        best_threshold = None
        best_info_gain = -1

        # to avoid the same feature being selected in every node,
        # you could randomly select a subset of features here

        # check all features
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # check every feature value as a candidate threshold
            for threshold in possible_thresholds:
                info_gain = self._information_gain(y, feature_values, threshold)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_info_gain

    # calculate information gain
    def _information_gain(self, y, feature_values, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # generate split
        left_idxs, right_idxs = self._split_idxs(feature_values, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # weighted avg child entropy
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # information gain is entropy before split - weighted entropy after
        info_gain = parent_entropy - child_entropy
        return info_gain

    # calculate entropy of label distribution
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    # split data based on a threshold
    def _split(self, X, y, feature_index, threshold):
        left_idxs = np.argwhere(X[:, feature_index] <= threshold).flatten()
        right_idxs = np.argwhere(X[:, feature_index] > threshold).flatten()
        return X[left_idxs], y[left_idxs], X[right_idxs], y[right_idxs]

    # get the indexes for the split
    def _split_idxs(self, feature_values, threshold):
        left_idxs = np.argwhere(feature_values <= threshold).flatten()
        right_idxs = np.argwhere(feature_values > threshold).flatten()
        return left_idxs, right_idxs

    # get most common label
    def _most_common_label(self, y):
        counter = np.bincount(y)
        most_common = np.argmax(counter)
        return most_common

    # make a prediction for a single example
    def _predict(self, example, node):
        # if we have a leaf node
        if node.value is not None:
            return node.value

        # choose the feature that we will test
        feature_value = example[node.feature_index]

        # determine if we will follow left or right child
        if feature_value <= node.threshold:
            return self._predict(example, node.left)
        else:
            return self._predict(example, node.right)
