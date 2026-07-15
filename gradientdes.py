import numpy as np

def gradient_descent(x, y, theta, learning_rate=0.01, epochs=1000):
    """
    Performs gradient descent to learn theta parameters.

    Parameters:
    x : numpy array
        Input features, shape (m, n) where m is the number of samples and n is the number of features.
    y : numpy array
        Target values, shape (m, 1).
    theta : numpy array
        Initial theta parameters, shape (n, 1).
    learning_rate : float
        Learning rate for gradient descent.
    epochs : int
        Number of epochs for gradient descent.

    Returns:
    theta : numpy array
        Optimized theta parameters, shape (n, 1).
    cost_history : list
        List of cost values at each iteration.
    """
    m = len(y)
    cost_history = []

    for _ in range(epochs):
        # Forward propagation
        predictions = np.dot(x, theta)

        # Compute cost
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        # Compute gradients
        gradients = (1/m) * np.dot(x.T, (predictions - y))

        # Update theta
        theta -= learning_rate * gradients

    return theta, cost_history

x = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.random.rand(100, 1)   # 100 target values

theta = np.random.rand(2, 1)  # Initial theta parameters
optimized_theta, cost_history = gradient_descent(x, y, theta)
print("Optimized theta parameters (W and B):", optimized_theta)
print("Final cost:", cost_history[-1])
