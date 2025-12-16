import numpy as np
from itertools import product

def qubo_objective(Q, x):
    """Compute the QUBO objective function value for a given binary vector x."""
    return np.dot(x, np.dot(Q, x))

def find_locations(binary_list):
    """Find the indices of ones and zeros in a binary list."""
    ones = [index for index, value in enumerate(binary_list) if value == 1]
    zeros = [index for index, value in enumerate(binary_list) if value == 0]
    return ones, zeros

def string_to_list(s):
    """Convert a string representation of state into a list of integers."""
    cleaned_string = str(s)[1:-1]
    list_representation = cleaned_string.split(',')
    return [int(x) for x in list_representation]

def solve_qubo(Q):
    """Solve the QUBO problem by evaluating all possible binary combinations."""
    size = Q.shape[0]
    best_solution = None
    best_value = float('inf')
    for combination in product([0, 1], repeat=size):
        x = np.array(combination)
        value = qubo_objective(Q, x)
        if value < best_value:
            best_value = value
            best_solution = x
    return best_solution, best_value

def get_sorted_indices(arr):
    """Return a list of sorted indices based on the values in a 1D array."""
    return np.argsort(arr)[::-1].tolist()

def solution_guesses3(avg_num, graph_val = 0):
    """Generate solution guesses based on the sorted indices of average values."""
    if graph_val == 0:
        solution = get_sorted_indices(avg_num)
        return [solution[:i+1] for i in range(len(solution))]
    elif graph_val == 1:
        sorted_indices = get_sorted_indices(avg_num)
        solution = []
        selected_groups = set()

        for idx in sorted_indices:
            group = int(idx / 3)
            # Check if the group has not been selected yet
            if group not in selected_groups:
                solution.append(idx)
                selected_groups.add(group)
        return [solution[:i+1] for i in range(len(solution))]
    else:
        sorted_indices = get_sorted_indices(avg_num)
        size = int(np.sqrt(len(avg_num)))
        solution = []
        selected_rows = set()
        selected_cols = set()
        # print("QQ",sorted_indices)
        
        # Step 3: Iterate over the sorted indices and select elements
        for idx in sorted_indices:
            row = idx // size  # Calculate row index
            col = idx % size   # Calculate column index
            # print(idx,row,col,selected_rows,selected_cols)
            # Check if both the row and column have not been selected yet
            if row not in selected_rows and col not in selected_cols:
                solution.append(idx)
                selected_rows.add(row)
                selected_cols.add(col)
        
            # print(solution)
        return [solution[:i+1] for i in range(len(solution))]
