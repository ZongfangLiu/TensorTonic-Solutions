import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _binary_cross_entropy(y, p):
    return - np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / y.shape[0]

def _gradient_w(X, p, y):
    return X.T @ (p - y) / y.shape[0]

def _gradient_b(p, y):
    return np.sum(p - y) / y.shape[0]

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    w, b = np.zeros(X.shape[1]), 0.0
    for i in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        # update
        w -= lr * _gradient_w(X, p, y)
        b -= lr * _gradient_b(p, y)

    return w, b
        
        