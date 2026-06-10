from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Critères d'arrêt
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            counter = Counter(y)
            return Node(value=counter.most_common(1)[0][0] if len(y) > 0 else 0)

        # Trouver le meilleur split
        best_feat, best_thresh = self._best_split(X, y, n_features)

        # Créer les sous-arbres
        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = ~left_idxs

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = 0, 0

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Gini du parent
        parent_gini = self._gini(y)
        # Séparation
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        # Gini moyen pondéré des enfants
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        child_gini = (n_l / n) * self._gini(y[left_idxs]) + (n_r / n) * self._gini(y[right_idxs])
        return parent_gini - child_gini

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum([p ** 2 for p in proportions if p > 0])

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestcustom(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self._estimator_type = "classifier"

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            # Bootstrapping (échantillonnage avec remplacement)
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = np.array(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        proba_1 = []
        for sample_pred in tree_preds:
            votes_1 = np.sum(sample_pred)
            proba_1.append(votes_1 / self.n_estimators)

        proba_1 = np.array(proba_1)
        return np.column_stack((1 - proba_1, proba_1))

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([Counter(sample_pred).most_common(1)[0][0] for sample_pred in tree_preds])