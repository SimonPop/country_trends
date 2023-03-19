import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
class MetroGraph():
    def __init__(self):
        self.graph = nx.DiGraph()
        self.n_lines = 1
        self.interstations = {}

    def create_interstations(self, stations):
        for station in stations:
            if not station in self.interstations:
                offset = self.graph.number_of_nodes()
                node_id = offset
                self.interstations[station] = offset
                self.graph.add_node(node_id, station=station, side=0, line=0)

    def add_line(self, stations, weight: float):
        """
        Creates an independant line in the disconnected graph.
        Saves the information to join the graph afterwards.
        """
        interweight = 0.1

        self.create_interstations(stations)

        way_in = []
        way_back = []

        for station in stations:
            offset = self.graph.number_of_nodes()
            interstation = self.interstations[station]
            # Sens 1  
            node_id_pos = offset
            self.graph.add_node(node_id_pos, station=station, side=1, line=self.n_lines)
            # Sens -1
            node_id_neg = offset + 1 
            self.graph.add_node(node_id_neg, station=station, side=-1, line=self.n_lines)
            # Add link between stations and inter-station
            self.graph.add_edge(interstation, node_id_neg, weight=interweight)
            self.graph.add_edge(interstation, node_id_pos, weight=interweight)
            self.graph.add_edge(node_id_pos, interstation, weight=interweight)
            self.graph.add_edge(node_id_neg, interstation, weight=interweight)
            # Add to cycle
            way_in.append(node_id_pos)
            way_back.insert(0, node_id_neg)
        
        station_order = way_in + way_back
        rolled_order = np.roll(station_order, -1)

        for a, b in zip(station_order, rolled_order):
            self.graph.add_edge(a, b, weight=weight)
            self.graph.add_edge(a, a, weight=1-weight-interweight)

        self.n_lines += 1

    def adjacency_matrix(self):
        A = nx.adjacency_matrix(self.graph, weight='weight')
        A = A / A.sum(axis=1)
        return A # .todense()

    def simulate(self, init_vector, step_nb: int):
        """Simulates the steps of flow in the graph when initialized with a given vector."""
        A = self.adjacency_matrix()
        x = init_vector
        X = [x]
        for _ in range(step_nb-1):
            x = A@x
            X.append(
                x
            )
        return X

    def num_stations(self):
        node2station = nx.get_node_attributes(self.graph, 'station')
        return len(set(node2station.values()))
    
    def num_nodes(self):
        return self.graph.number_of_nodes()
    
    def random_initialization(self, low, high):
        return np.random.randint(low, high, size=len(self.graph.nodes()))
    
    def contract(self, G):
        node2station = nx.get_node_attributes(self.graph, "station")
        station2nodes = defaultdict(list)
        for n, s in node2station.items():
            station2nodes[s].append(n)
        for station in station2nodes.keys():
            nodes = station2nodes[station]
            base = nodes[0]
            for node in nodes[1:]:
                G = nx.contracted_nodes(G, base, node)
        return G
        