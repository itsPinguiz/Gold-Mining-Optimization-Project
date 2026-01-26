import networkx as nx
import numpy as np
import random

class Problem:
    def __init__(self, n, density, alpha, beta, seed=None):
        self.n = n
        self.density = density
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Create Random Graph
        # We attempt to create a connected graph. 
        # If the random graph is disconnected, we force connections.
        connected = False
        while not connected:
            self.graph = nx.gnp_random_graph(n, density, directed=False, seed=seed)
            if nx.is_connected(self.graph):
                connected = True
            elif seed is not None: 
                # Force connection for seeded instances if they fail initial check
                components = list(nx.connected_components(self.graph))
                for i in range(len(components) - 1):
                    u = list(components[i])[0]
                    v = list(components[i+1])[0]
                    self.graph.add_edge(u, v)
                    # Add random weight for this forced edge
                    self.graph[u][v]['dist'] = np.random.random() * 10
                connected = True

        # Assign Weights (Distance) and Gold
        # Note: 'dist' is assigned here for existing edges
        for u, v in self.graph.edges():
            if 'dist' not in self.graph[u][v]:
                self.graph[u][v]['dist'] = np.random.random() * 10 
            
        for node in self.graph.nodes():
            if node == 0:
                self.graph.nodes[node]['gold'] = 0
            else:
                self.graph.nodes[node]['gold'] = np.random.random() * 10

    def evaluate_path(self, path):
        """
        Validates the path and calculates the total cost.
        Returns: (Total Cost, Status String)
        """
        # Check Format
        if not path or path[0] != 0 or path[-1] != 0:
            return float('inf'), "Invalid: Must start and end at Depot (0)"
        
        # Check all cities visited
        visited = set(path)
        all_cities = set(range(self.n))
        if visited != all_cities:
            return float('inf'), f"Invalid: Missed cities {all_cities - visited}"
            
        current_gold = 0
        total_cost = 0
        
        # Simulate Path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Check Edge Existence
            if not self.graph.has_edge(u, v):
                return float('inf'), f"Invalid: No edge between {u} and {v}"
                
            dist = self.graph[u][v]['dist']
            
            # Cost Formula
            step_cost = dist + (dist * current_gold * self.alpha) ** self.beta
            total_cost += step_cost
            
            # Logic: 
            # - If we arrive at Depot (0), we drop off gold (reset to 0).
            # - If we arrive at a City, we pick up its gold.
            #   (Simplified validation: assumes we pick up gold the first time we visit)
            if v == 0:
                current_gold = 0
            else:
                # We simply add the gold of node v. 
                # Note: A smarter validator might track which gold has already been picked up,
                # but for this specific problem definition, we assume full pickup on visit.
                total_cost_check = total_cost # Snapshot for debugging if needed
                gold_at_v = self.graph.nodes[v]['gold']
                current_gold += gold_at_v
        
        return total_cost, "Valid"