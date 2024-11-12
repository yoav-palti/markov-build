import numpy as np

from ring_partition import partition_on_ring
from utils import normalize_rows


def construct_transition_matrix_ring(stationary_distribution, N, self_loops=False, anchor=None, anchor_atraction=10):
    """
    Construct a Markov transition matrix with the given stationary distribution.
    see: https://www.tandfonline.com/doi/pdf/10.1080/00031305.1977.10479205?casa_token=xFDNuf82eQIAAAAA:bPXUpy5az3B89yLpKyeL4Xa1srl8oxFP6OJEYsNB3SGy1MeD30Hr-YT5TrdwoABGkrNBmONaCDVua7k
    for a bit more information on the method
    """
    # Normalize the probabilities to integers based on the common denominator
    partition_vec = np.astype(stationary_distribution * N, np.int64)

    # Step 2: Partition states based on the probability vector
    ring_partition = partition_on_ring(partition_vec, recurring_states=self_loops, max_attempts=-1, anchor=anchor, anchor_atraction=anchor_atraction)

    # Step 3: Initialize an empty transition matrix
    P = np.zeros((len(partition_vec), len(partition_vec)))

    # Step 4: Fill in the transition probabilities
    for ring_location in range(len(ring_partition)):
        current_state = ring_partition[ring_location]
        next_state = ring_partition[(ring_location + 1) % N]
        P[current_state, next_state] += 1

    P = normalize_rows(P)
    return P
