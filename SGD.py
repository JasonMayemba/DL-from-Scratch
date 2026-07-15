import numpy as np

def stochastic_gradient_descent(x, y, theta, learning_rate=0.01, epochs=1000):
    """
    Performs stochastic gradient descent to learn theta parameters.

    Parameters:
    x : numpy array
        Input features, shape (m, n) where m is the number of samples and n is the number of features.
    y : numpy array
        Target values, shape (m, 1).
    theta : numpy array
        Initial theta parameters, shape (n, 1).
    learning_rate : float
        Learning rate for stochastic gradient descent.
    epochs : int
        Number of epochs for stochastic gradient descent.

    Returns:
    theta : numpy array
        Optimized theta parameters, shape (n, 1).
    cost_history : list
        List of cost values at each iteration.
    """
