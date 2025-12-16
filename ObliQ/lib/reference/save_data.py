import numpy as np
from itertools import product


def generate_random_qubo(size):
    """Generate a random QUBO matrix of given size."""
    Q = np.random.uniform(-2, 2, (size, size))
    Q = np.triu(Q) + np.triu(Q, 1).T  # Make the matrix symmetric
    np.fill_diagonal(Q, 0)
    return Q


def qubo_objective(Q, x):
    """Compute the QUBO objective function value for a given binary vector x."""
    return np.dot(x, np.dot(Q, x))


def solve_qubo(Q):
    """Solve the QUBO problem by evaluating all possible binary combinations."""
    size = Q.shape[0]
    best_solution = None
    best_value = float("inf")
    for combination in product([0, 1], repeat=size):
        x = np.array(combination)
        value = qubo_objective(Q, x)
        if value < best_value:
            best_value = value
            best_solution = x
    return best_solution, best_value
