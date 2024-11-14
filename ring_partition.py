import numpy as np
import logging

logger = logging.getLogger(__name__)

def _partition_on_ring(partition_vector, recurring_states=False, anchor=None, anchor_atraction=10, rng=None):
    """
    Partition states into subsets S_i based on the probability vector.

    Parameters
    ----------
    partition_vector
    N
    recurring_states: bool
        If False, the subsets will not contain recurring states

    Returns
    -------
    List of subsets S_i

    Raises
    -------
    RuntimeError
        the partition may fail if no_recurring_states is demanded
    """
    if rng is None:
        rng = np.random.default_rng()

    N = np.sum(partition_vector)
    partition_vector = np.asarray(partition_vector).copy()
    num_states = len(partition_vector)

    states = np.arange(num_states)

    if not recurring_states and partition_vector.max() >= N / 2:
        raise ValueError(
            "The partition vector must have a maximum value less than 0.5 if no recurring state is demanded.")

    # start with the most non probable state in order to make it easier to avoid recurring states at the last step
    first_state = np.argmin(partition_vector)
    ring_states = [first_state]
    partition_vector[first_state] -= 1

    for i in range(1,N):
        prev_chosen_state = ring_states[-1]
        if not recurring_states:
            partition_vector_no_last = partition_vector.copy()
            partition_vector_no_last[prev_chosen_state] = 0
            if i == N - 1:
                partition_vector_no_last[first_state] = 0
            partition_porbs = partition_vector_no_last
        else:
            partition_porbs = partition_vector
        if partition_porbs.sum() == 0:
            raise RuntimeError("The partition failed. Try again.")

        # make even states more probable to be chosen after the anchor state
        if prev_chosen_state==anchor:
            partition_porbs[::2]*=anchor_atraction

        # # make the anchor state more probable to be chosen after even states
        # if prev_chosen_state%2==
        #     partition_porbs[anchor]*=anchor_atraction

        partition_porbs = partition_porbs / partition_porbs.sum()
        chosen_state = rng.choice(states, p=partition_porbs)
        ring_states.append(chosen_state)
        partition_vector[chosen_state] -= 1
    return np.asarray(ring_states)

def partition_on_ring(partition_vector, recurring_states=True, max_attempts=100, anchor=None, anchor_atraction=10, seed=None):
    """
    Partition states into subsets S_i based on the probability vector.

    Parameters
    ----------
    partition_vector
    recurring_states: bool
        If False, the subsets will not contain recurring states
    max_attempts: int
        Maximum number of attempts to partition the states
        if -1 the partition will be attempted until it is successful

    Returns
    -------
    List of subsets S_i
    """
    max_attempts = int(max_attempts)
    counter = 0
    rng = np.random.default_rng(seed)  # same rng for all attempts in order to avoid the same attempts
    while counter != max_attempts:
        try:
            logger.info(f"Attempt {counter + 1}")
            return _partition_on_ring(partition_vector, recurring_states, anchor=anchor, anchor_atraction=anchor_atraction, rng=rng)
        except RuntimeError:
            counter += 1
    raise RuntimeError(f"The partition failed after {max_attempts} attempts.")


if __name__ == '__main__':
    partition_vector = [1, 2, 3, 4, 5,14]
    partitions = partition_on_ring(partition_vector,  recurring_states=False, max_attempts=-1)
    print(partitions)