import numpy as np

def generate_qubo_matrix(n, k, edges, P):
    Q = np.zeros((n*k, n*k))

    # Add penalties for the equality constraints (each node must be colored exactly one color)
    for i in range(n):
        for p in range(k):
            Q[i*k + p, i*k + p] -= P
        for p in range(k):
            for q in range(p + 1, k):
                Q[i*k + p, i*k + q] += P
                Q[i*k + q, i*k + p] += P

    # Add penalties for the inequality constraints (no two adjacent nodes can share the same color)
    for (i, j) in edges:
        for p in range(k):
            Q[i*k + p, j*k + p] += P
            Q[j*k + p, i*k + p] += P

    constant_term = P * (n + len(edges))  # Add the constant term for the penalties
    return Q, constant_term

def check_solution(n, k, edges, solution):
    equality_violations = 0
    inequality_violations = 0

    # Check the equality constraints
    for i in range(n):
        if np.sum(solution[i*k:(i+1)*k]) != 1:
            equality_violations += 1

    # Check the inequality constraints
    for (i, j) in edges:
        for p in range(k):
            if solution[i*k + p] + solution[j*k + p] > 1:
                inequality_violations += 1

    return equality_violations, inequality_violations
