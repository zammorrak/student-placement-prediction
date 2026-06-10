import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegressionCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self._estimator_type = "classifier"

    def get_params(self, deep=True):
        return {"lr": self.lr, "n_iters": self.n_iters}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Conversion en tableaux numpy
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Initialisation des paramètres
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Descente de gradient
        for _ in range(self.n_iters):
            model_linear = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(model_linear)

            # Calcul des gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Mise à jour des poids
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        X = np.array(X)
        model_linear = np.dot(X, self.weights) + self.bias
        proba_1 = self._sigmoid(model_linear)
        return np.column_stack((1 - proba_1, proba_1))

    def predict(self, X):
        X = np.array(X)
        model_linear = np.dot(X, self.weights) + self.bias
        return np.where(self._sigmoid(model_linear) >= 0.5, 1, 0)