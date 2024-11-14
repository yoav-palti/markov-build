import numpy as np
from utils import normalize_rows

def construct_transition_matrix_converge(stationary_distribution, self_loops = False, seed=None):
    n = len(stationary_distribution)
    rng = np.random.default_rng(seed)

    # Step 1: Generate a random matrix with off-diagonal positive values and zero diagonal.
    P = rng.rand(n, n)  # Generate random values for all entries
    if not self_loops:
        np.fill_diagonal(P, 0)    # Set diagonal elements to 0

    # Step 2: Normalize columns to satisfy column sum constraint (P @ u = u)
    P = normalize_rows(P)

    # Step 3: Adjust to satisfy y @ P = y
    # We use an iterative adjustment to satisfy the stationary distribution
    for _ in range(1000):  # Iterative adjustment loop
        P = P * stationary_distribution / (stationary_distribution @ P)
        P = normalize_rows(P)
    return P