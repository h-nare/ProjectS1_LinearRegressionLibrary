# main functions 

import json
import math
import random
from .metrics import mse
from .plots import plot_loss


class LinearRegression:
    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        normalize=True,
        early_stopping=True,
        patience=100,
        tol=1e-6,
        random_init=True,
    ):
        """
        A simple Linear Regression model trained with Gradient Descent.

        - learning_rate: step size for gradient descent updates
        - epochs: maximum number of passes over the training data
        - normalize: whether to standardize features (mean 0, std 1)
        - early_stopping: stop if loss stops improving
        - patience: how many epochs to wait before stopping
        - tol: minimum required improvement to reset patience
        - random_init: if True, initialize weights to small random values
                       otherwise initialize to zeros
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.normalize = normalize
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.random_init = random_init

        # model parameters
        self.weights = []        # list of weights (one per feature)
        self.bias = 0.0

        # normalization parameters
        self.feature_means = []
        self.feature_stds = []

        # training history
        self.loss_history = []

    # ---------- internal helpers for normalization ----------

    def _compute_norm_params(self, X):
        """
        Compute mean and std for each feature (column of X).
        """
        n_features = len(X[0])
        self.feature_means = []
        self.feature_stds = []

        for j in range(n_features):
            col = [X[i][j] for i in range(len(X))]
            mean = sum(col) / len(col)
            # variance
            var = sum((x - mean) ** 2 for x in col) / len(col)
            std = math.sqrt(var)
            if std == 0:
                std = 1.0  # avoid division by zero
            self.feature_means.append(mean)
            self.feature_stds.append(std)

    def _normalize_X(self, X):
        """
        Apply stored normalization (mean/std) to given X.
        Assumes _compute_norm_params has already been called.
        """
        X_norm = []
        for row in X:
            new_row = []
            for j in range(len(row)):
                x = row[j]
                x_norm = (x - self.feature_means[j]) / self.feature_stds[j]
                new_row.append(x_norm)
            X_norm.append(new_row)
        return X_norm

    def _init_weights(self, n_features):
        """
        Initialize weights and bias.
        """
        if self.random_init:
            self.weights = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
        else:
            self.weights = [0.0] * n_features
        self.bias = 0.0

    # ---------- FIT (TRAIN) ----------

    def fit(self, X, y):
        """
        Train the model using gradient descent on the given data.

        X: list of samples, each sample is a list of feature values.
        y: list of target values.
        """
        if not X:
            raise ValueError("Empty X passed to fit().")

        n_samples = len(X)
        n_features = len(X[0])

        # normalization
        if self.normalize:
            self._compute_norm_params(X)
            X_train = self._normalize_X(X)
        else:
            X_train = X

        # initialize parameters
        self._init_weights(n_features)
        self.loss_history = []

        # for early stopping
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):

            # ----- predictions -----
            y_pred = []
            for i in range(n_samples):
                pred = self.bias
                for j in range(n_features):
                    pred += self.weights[j] * X_train[i][j]
                y_pred.append(pred)

            # ----- compute gradients -----
            grad_w = [0.0] * n_features
            grad_b = 0.0

            for i in range(n_samples):
                error = y_pred[i] - y[i]
                grad_b += error
                for j in range(n_features):
                    grad_w[j] += error * X_train[i][j]

            # average gradients
            grad_b = (2.0 / n_samples) * grad_b
            for j in range(n_features):
                grad_w[j] = (2.0 / n_samples) * grad_w[j]

            # ----- update parameters -----
            self.bias -= self.learning_rate * grad_b
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * grad_w[j]

            # ----- compute loss & record -----
            current_loss = mse(y, y_pred)
            self.loss_history.append(current_loss)

            # ----- early stopping -----
            if self.early_stopping:
                if current_loss < best_loss - self.tol:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        # print(f"Early stopping at epoch {epoch}")
                        break

    # ---------- PREDICT ----------

    def predict(self, X):
        """
        Predict y values for given X.

        X: list of samples, each sample is a list of feature values.
        """
        if not X:
            return []

        # apply same normalization as during training
        if self.normalize:
            X = self._normalize_X(X)

        preds = []
        for row in X:
            pred = self.bias
            for j in range(len(self.weights)):
                pred += self.weights[j] * row[j]
            preds.append(pred)
        return preds

    # ---------- CONVENIENCE: PLOT LOSS ----------

    def plot_loss(self):
        """
        Plot the training loss curve.
        """
        plot_loss(self.loss_history)

        # -----------------------
    # save model parameters
    # -----------------------
    def save(self, filename):
        # I save weights, bias and normalization params to a JSON file
        data = {
            "weights": self.weights,
            "bias": self.bias,
            "normalize": self.normalize,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds
        }

        with open(filename, "w") as f:
            json.dump(data, f)

    # -----------------------
    # load model parameters
    # -----------------------
    def load(self, filename):
        # I load saved parameters from JSON and restore the model state
        with open(filename, "r") as f:
            data = json.load(f)

        self.weights = data["weights"]
        self.bias = data["bias"]
        self.normalize = data["normalize"]
        self.feature_means = data["feature_means"]
        self.feature_stds = data["feature_stds"]


# class LinearRegression:
#     def __init__(self, learning_rate, epochs):
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.weights = []
#         self.bias = 0

#     # y = m*x + b
#     def fit(self, X, y):
#         n_samples = len(X)
#         n_features = len(X[0])

#         self.weights = [0] * n_features
#         self.bias = 0


#         for _ in range(self.epochs):
#             # compute predictions
#             y_pred = []
#             for i in range(n_samples):
#                 prediction = self.bias
#                 for j in range(n_features):
#                     prediction += self.weights[j]*X[i][j]
#                 y_pred.append(prediction)

#             # compute gradients for m and b
#             grad_w = [0] * n_features
#             grad_b = 0

#             for i in range(n_samples):
#                 error = y_pred[i] - y[i]
#                 grad_b += error
#                 for j in range(n_features):
#                     grad_w[j] += error * X[i][j]

#             grad_b = (2 / n_samples) * grad_b
#             for j in range(n_features):
#                 grad_w[j] = (2/n_samples)*grad_w[j]

#             # update parameters
#             grad_b -= self.learning_rate * grad_b
#             for j in range(n_features):
#                 self.weights[j] -= self.learning_rate*grad_w[j]

#     def predict(self, X):
#         predictions = []
#         for sample in X:
#             prediction = self.bias
#             for j in range(len(self.weights)):
#                 prediction += self.weights[j] * sample[j]
#             predictions.append(prediction)
#         return predictions

#     def mse(self,X,y):
#         y_pred = self.predict(X)
#         error_sum = 0
#         for i in range(len(y)):
#             error_sum += (y_pred[i] - y[i]) ** 2
#         return error_sum / len(y)
