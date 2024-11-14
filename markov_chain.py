import numpy as np

class MarkovChain:
    def __init__(self, transition_matrix, states):
        self.P = transition_matrix
        self.states = states
        self.n = len(states)
        self.state_map = {i: state for i, state in enumerate(states)}
        self._stationary_dist = None

    def sample(self, sequence_length, seed=None):
        rng = np.random.default_rng(seed)
        # get the initial state
        state_k = rng.choice(self.n, p=self.stationary_dist)
        sequence = [self.state_map[state_k]]
        # generate the sequence
        for _ in range(sequence_length - 1):
            state_k = rng.choice(self.n, p=self.P[state_k])
            sequence.append(self.state_map[state_k])
        return sequence

    @property
    def stationary_dist(self):
        if self._stationary_dist is None:
            eig_vals, eig_vec = np.linalg.eig(self.P.T)
            stationary_dist = eig_vec[:, np.argmax(eig_vals)].real
            stationary_dist = stationary_dist / stationary_dist.sum()
            self._stationary_dist = stationary_dist
        return self._stationary_dist

    def occur_after_anchor(self, anchor_idx):
        occurrence_after = self.stationary_dist[anchor_idx]*self.P[anchor_idx,:]
        return occurrence_after

    def occur_before_anchor(self, anchor_idx):
        occurrence_before = self.stationary_dist*self.P[:,anchor_idx]
        return occurrence_before

    def neighbor_to_anchor(self, anchor_idx):
        return self.occur_before_anchor(anchor_idx) + self.occur_after_anchor(anchor_idx)

    def plot(self,th=0):
        import networkx as nx
        import matplotlib.pyplot as plt
        # Convert sympy Matrix to numpy array for easier handling with networkx
        graph = nx.DiGraph()

        # Add nodes and edges with rates
        for i, state_from in enumerate(self.states):
            for j, state_to in enumerate(self.states):
                rate = self.P[i][j]
                if rate > th:
                    graph.add_edge(state_from, state_to, label=f"{rate:.2f}")

        # Draw the graph
        pos = nx.spring_layout(graph)
        edge_labels = {(u, v): data['label'] for u, v, data in graph.edges(data=True)}

        arc_rad = 0.1
        connectionstyle = f"arc3,rad={arc_rad}"
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold',
                arrows=True, connectionstyle=connectionstyle)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, connectionstyle=connectionstyle)
        plt.title("Continuous-Time Markov Chain with Forward and Backward Rates")
        plt.show()