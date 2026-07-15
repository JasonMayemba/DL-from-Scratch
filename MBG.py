import numpy as np

def mini_batch_gradient_descent(x, y, theta, learning_rate=0.01, epochs=1000, batch_size=32):
    """
    Performs mini-batch gradient descent to learn theta parameters.

    Parameters:
    x : numpy array
        Input features, shape (m, n) where m is the number of samples and n is the number of features.
    y : numpy array
        Target values, shape (m, 1).
    theta : numpy array
        Initial theta parameters, shape (n, 1).
    learning_rate : float
        Learning rate for mini-batch gradient descent.
    epochs : int
        Number of epochs for mini-batch gradient descent.
    batch_size : int
        Size of each mini-batch.

    Returns:
    theta : numpy array
        Optimized theta parameters, shape (n, 1).
    cost_history : list
        List of cost values at each iteration.
    """
    m = len(y)
    cost_history = []

    for _ in range(epochs):
        # Shuffle the data
        indices = np.arange(m)
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

