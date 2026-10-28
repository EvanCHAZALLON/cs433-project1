"""The six methods required for Project 1.

Every function returns (w, loss): the last weight vector and the loss at that w.
For ridge and regularized logistic regression the loss does not include the penalty.
"""

import numpy as np


def compute_mse(y, tx, w):
    """MSE with the 1/2 factor: L(w) = 1/(2N) * ||y - tx w||^2."""
    e = y - tx @ w
    return e @ e / (2 * len(y))


def sigmoid_function(z):
    """Logistic function 1 / (1 + exp(-z))."""
    return 1 / (1 + np.exp(-z))


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with gradient descent on the MSE.

    Args:
        y: targets, shape (N,).
        tx: features, shape (N, D).
        initial_w: starting weights, shape (D,).
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: last weights, shape (D,).
        loss: MSE at w.
    """
    w = initial_w
    for n in range(max_iters):
        e = y - tx @ w
        grad = -tx.T @ e / len(e)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with stochastic gradient descent (batch size 1) on the MSE.

    At each step one sample is drawn at random and w moves along its gradient.
    Same arguments and outputs as mean_squared_error_gd.
    """
    w = initial_w
    for n in range(max_iters):
        p = np.random.randint(0, len(y))
        e = y[p : p + 1] - tx[p : p + 1] @ w
        sub_grad = -tx[p : p + 1].T @ e / len(e)
        w = w - gamma * sub_grad
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares with the normal equations (tx^T tx) w = tx^T y.

    Returns:
        w: optimal weights, shape (D,).
        loss: MSE at w.
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression with the normal equations.

    Minimizes 1/(2N) ||y - tx w||^2 + lambda_ ||w||^2, i.e. solves
    (tx^T tx + 2 N lambda_ I) w = tx^T y.

    Returns:
        w: optimal weights, shape (D,).
        loss: MSE at w, without the penalty term.
    """
    a = tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w_opt = np.linalg.solve(a, tx.T @ y)
    loss = compute_mse(y, tx, w_opt)
    return w_opt, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with gradient descent (labels y in {0, 1}).

    Args:
        y: labels in {0, 1}, shape (N,).
        tx: features, shape (N, D).
        initial_w: starting weights, shape (D,).
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: last weights, shape (D,).
        loss: mean negative log-likelihood at w.
    """
    w = initial_w
    for n in range(max_iters):
        p = sigmoid_function(tx @ w)
        grad = tx.T @ (p - y) / len(y)
        w = w - gamma * grad
    p = sigmoid_function(tx @ w)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression with gradient descent (y in {0, 1}).

    Minimizes the mean negative log-likelihood + lambda_ ||w||^2, so the
    penalty adds 2 lambda_ w to the gradient.

    Returns:
        w: last weights, shape (D,).
        loss: mean negative log-likelihood at w, without the penalty term.
    """
    w = initial_w
    for n in range(max_iters):
        p = sigmoid_function(tx @ w)
        grad = tx.T @ (p - y) / len(y) + 2 * lambda_ * w
        w = w - gamma * grad
    p = sigmoid_function(tx @ w)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return w, loss
