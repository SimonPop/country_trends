import networkx as nx
import numpy as np
import pandas as pd

class CyclingGraph():
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node2station = {} # Contains a link from a gaph node to the corresponding station (multiple node can correspond to the same station, if they are on opposite direction or on a different line)
        self.node2line = {}
        self.n_lines = 0

    def add_line(self, stations, weight: float, cycle: bool):
        """
        Creates an independant line in the disconnected graph.
        Saves the information to join the graph afterwards.
        """
        cycle_stations = stations + stations[::-1]
        offset = self.graph.number_of_nodes()
        nodes = [s + offset for s, _ in enumerate(cycle_stations)]
        for station, node in zip(cycle_stations, nodes):
            self.node2station[node] = station

        for station in nodes:
            self.node2line[station] = self.n_lines

        if cycle:
            base_nodes = nodes
            shifted_nodes = np.roll(nodes, -1)
        else:
            base_nodes = nodes[:-1]
            shifted_nodes = nodes[1:]
        for station_a, station_b in zip(base_nodes, shifted_nodes):
            self.graph.add_edge(station_a, station_b, weight=weight)
            self.graph.add_edge(station_a, station_a, weight=1-weight)

        if not cycle:
            self.graph.add_edge(shifted_nodes[-1], shifted_nodes[-1], weight=1)

        self.n_lines += 1

    def merge_graph(self):
        pass

    def adjacency_matrix(self):
        A = nx.adjacency_matrix(self.graph, weight='weight').T
        return A.todense()

    def simulate(self, init_vector, step_nb: int):
        """Simulates the steps of flow in the graph when initialized with a given vector."""
        A = nx.adjacency_matrix(self.graph, weight='weight').T
        x = init_vector
        X = [x]
        for i in range(step_nb-1):
            x = A@x
            X.append(
                x
            )
        X_merged = self.merge_X(np.array(X))
        return pd.DataFrame(X_merged)

    def num_stations(self):
        return len(set(self.node2station.values()))

    def merge_X(self, X):
        stations = set(self.node2station.values())
        merged_X = {station: np.zeros(X.shape[0]) for station in stations}
        for x in range(X.shape[1]):
            station = self.node2station[x]
            merged_X[station] = merged_X[station] + X[:,x]
        return merged_X
    
    def random_initialization(self, low, high):
        return np.random.randint(low, high, size=len(self.graph.nodes()))