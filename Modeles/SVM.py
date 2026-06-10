import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LinearSVMCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param  # Terme de régularisation
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self._estimator_type = "classifier"

    def get_params(self, deep=True):
        return {"lr": self.lr, "lambda_param": self.lambda_param, "n_iters": self.n_iters}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        X = np.array(X)
        y_transformed = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_transformed[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_transformed[idx]))
                    self.b -= self.lr * y_transformed[idx]

        return self

    def predict_proba(self, X):
        X = np.array(X)
        distance = np.dot(X, self.w) - self.b
        proba_1 = 1 / (1 + np.exp(-distance))
        return np.column_stack((1 - proba_1, proba_1))

    def predict(self, X):
        X = np.array(X)
        approx = np.dot(X, self.w) - self.b
        return np.where(np.sign(approx) <= -1, 0, 1)