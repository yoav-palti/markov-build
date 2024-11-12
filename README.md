# Markov Chain Ring Construction

This project provides tools for constructing Markov transition matrices with a given stationary distribution using a ring partition method.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

To use this project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt

# if you want to plot the markov chain
pip install matplotlib networkx -U
```

## Usage

To construct a Markov chain ring, use the `markov_chain_ring` module. For example:

```python
from ring_construction import construct_transition_matrix_ring
from markov_chain import MarkovChain

# Define the stationary distribution
N = 100
# notice that the sum of the stationary distribution must be equal to 1
stationary_dist = np.array([26,11,11,9,9,7,7,5,5,4,4,2])/N

# Construct the Markov chain ring
transition_matrix = construct_transition_matrix_ring(stationary_dist,N)
markov_chain = MarkovChain(transition_matrix)

print(transition_matrix)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
