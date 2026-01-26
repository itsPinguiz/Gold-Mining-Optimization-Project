import random
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# PARAMETERS
POP_SIZE=30
GENERATIONS=100
MAX_LOAD=None

# HELPER: SPLIT PROCEDURE
def split_procedure(permutation, problem_data):
    """
    Decodes a TSP permutation into a VRP solution using O(1) Matrix lookups.
    """
    dist_matrix = problem_data['dist_matrix']
    golds = problem_data['golds']
    alpha = problem_data['alpha']
    beta = problem_data['beta']
    
    n = len(permutation)
    V = [float('inf')] * (n + 1)
    P = [-1] * (n + 1)
    V[0] = 0
    
    # Lookahead limit
    MAX_TRIP_LENGTH = 32

    for i in range(n):
        if V[i] == float('inf'): continue
            
        current_gold = 0
        trip_cost = 0
        u = 0 # Start at Depot
        
        for j in range(i, min(n, i + MAX_TRIP_LENGTH)):
            v = permutation[j]
            
            # TRAVEL u -> v
            d_uv = dist_matrix[u][v]
            step_cost = d_uv + (d_uv * current_gold * alpha) ** beta
            trip_cost += step_cost
            
            # PICK UP GOLD
            current_gold += golds[v]
            
            # RETURN COST v -> 0
            d_v0 = dist_matrix[v][0]
            return_cost = d_v0 + (d_v0 * current_gold * alpha) ** beta
            
            total_segment_cost = trip_cost + return_cost
            
            if V[i] + total_segment_cost < V[j+1]:
                V[j+1] = V[i] + total_segment_cost
                P[j+1] = i
            
            u = v

    # Reconstruct Virtual Path
    virtual_path = []
    curr = n
    while curr > 0:
        prev = P[curr]
        virtual_path.append(permutation[prev:curr])
        curr = prev
    virtual_path.reverse()
    
    return virtual_path, V[n]

# HELPER: GENETIC OPERATORS
def inver_over_crossover(p1, p2):
    """
    Inver-over crossover operator for TSP permutations.
    """
    current = p1[:]
    if len(current) < 2: return current
    c = random.choice(current)
    
    for _ in range(15): 
        idx_c = current.index(c)
        next_idx = (idx_c + 1) % len(current)
        c_next = current[next_idx]
        
        if random.random() < 0.05:
            c_prime = random.choice(current)
        else:
            try:
                idx_p2 = p2.index(c)
                c_prime = p2[(idx_p2 + 1) % len(p2)]
            except ValueError:
                c_prime = random.choice(current)
            
        if c_prime == c_next:
            break
            
        idx_c_prime = current.index(c_prime)
        s, e = min(idx_c+1, idx_c_prime), max(idx_c+1, idx_c_prime)
        current[s:e+1] = current[s:e+1][::-1]
        c = c_prime
        
    return current

# HELPER: INITIALIZATION 
def greedy_initialization(dist_matrix, cities):
    """
    Creates a greedy TSP path as an initial genotype.
    """
    unvisited = set(cities)
    current = 0 
    path = []
    
    while unvisited:
        best_city = -1
        min_dist = float('inf')
        
        candidates = unvisited
        if len(unvisited) > 100:
             candidates = random.sample(list(unvisited), 50)
        
        for city in candidates:
            d = dist_matrix[current][city]
            if d < min_dist:
                min_dist = d
                best_city = city
        
        if best_city == -1: best_city = list(unvisited)[0]

        path.append(best_city)
        unvisited.remove(best_city)
        current = best_city
        
    return path

# HELPER: PROBLEM EXPANSION
def expand_problem_data(dist_matrix, golds, max_load=None):
    """
    Splits cities ONLY if they are extreme outliers.
    """
    new_golds = []
    original_indices = []
    
    # If max_load is not manually set, we calculate it dynamically.
    if max_load is None:
        # We only split a city if it's in the top 10% of heaviness
        # OR if it's huge relative to the average.
        avg_gold = np.mean(golds[1:]) # Ignore depot
        max_gold_in_graph = np.max(golds)
        
        # Heuristic: Only split if a city is > 150% of the average
        # This prevents exploding the graph size with unnecessary splits
        threshold = max(max_gold_in_graph * 0.8, avg_gold * 1.5)
    else:
        threshold = max_load

    # Create Virtual Cities
    for i, amount in enumerate(golds):
        # Never split depot (0) or "normal" sized cities (<= threshold)
        if i == 0 or amount <= threshold:
            new_golds.append(amount)
            original_indices.append(i)
        else:
            # We don't want 10 tiny chunks. We want 2 or 3 manageable chunks.
            # We target a chunk size roughly equal to the threshold.
            num_chunks = int(np.ceil(amount / threshold))
            chunk_gold = amount / num_chunks
            
            for _ in range(num_chunks):
                new_golds.append(chunk_gold)
                original_indices.append(i)
                
    new_n = len(new_golds)
    new_golds = np.array(new_golds)
    
    # Rebuild Distance Matrix
    new_dist_matrix = np.zeros((new_n, new_n))
    for i in range(new_n):
        for j in range(new_n):
            u_old = original_indices[i]
            v_old = original_indices[j]
            new_dist_matrix[i][j] = dist_matrix[u_old][v_old]
            
    return new_dist_matrix, new_golds, original_indices

# MAIN ALGORITHM
def my_genetic_algorithm(problem_instance):
    """
    The main optimization pipeline.
    """
    # PRE-COMPUTATION & VIRTUAL SPLITTING
    graph_sparse = nx.to_scipy_sparse_array(problem_instance.graph, weight='dist')
    raw_dist_matrix = shortest_path(graph_sparse, method='auto', directed=False)
    raw_golds = np.array([problem_instance.graph.nodes[i]['gold'] for i in range(problem_instance.n)])
    
    # Call the helper function
    dist_matrix, golds, id_map = expand_problem_data(raw_dist_matrix, raw_golds, MAX_LOAD)
    
    problem_data = {
        'dist_matrix': dist_matrix, # Expanded Matrix
        'golds': golds,             # Expanded Gold
        'alpha': problem_instance.alpha,
        'beta': problem_instance.beta
    }
    
    # GA INITIALIZATION (On Virtual Cities)
    n_expanded = len(golds)
    cities = list(range(1, n_expanded)) 
    population = []
    
    # Seeding
    greedy_geno = greedy_initialization(dist_matrix, cities)
    v_path, fit = split_procedure(greedy_geno, problem_data)
    population.append({'g': greedy_geno, 'f': fit, 'vp': v_path})
    
    for _ in range(POP_SIZE - 1):
        geno = cities[:]
        random.shuffle(geno)
        v_path, fit = split_procedure(geno, problem_data)
        population.append({'g': geno, 'f': fit, 'vp': v_path})
        
    population.sort(key=lambda x: x['f'])
    
    # EVOLUTION LOOP
    for gen in range(GENERATIONS):
        p1 = min(random.sample(population, 4), key=lambda x: x['f'])
        p2 = min(random.sample(population, 4), key=lambda x: x['f'])
        
        child_geno = inver_over_crossover(p1['g'], p2['g'])
        
        if random.random() < 0.15:
            i, j = random.sample(range(len(child_geno)), 2)
            child_geno[i], child_geno[j] = child_geno[j], child_geno[i]
            
        v_path, fit = split_procedure(child_geno, problem_data)
        
        if fit < population[-1]['f']:
            population[-1] = {'g': child_geno, 'f': fit, 'vp': v_path}
            population.sort(key=lambda x: x['f'])

    # PATH RECONSTRUCTION (Virtual -> Real)
    best_virtual_segments = population[0]['vp']
    final_physical_path = [0]
    current_real_node = 0
    
    for segment in best_virtual_segments:
        # Map Virtual IDs back to Real IDs
        # Add Depot (0) to end of every trip
        trip_targets_virtual = segment + [0]
        
        for target_virtual in trip_targets_virtual:
            target_real = id_map[target_virtual] # Look up real ID
            
            if target_real == current_real_node: 
                continue # Skip 0-distance moves
            
            path_leg = nx.shortest_path(problem_instance.graph, 
                                        source=current_real_node, 
                                        target=target_real, 
                                        weight='dist')
            final_physical_path.extend(path_leg[1:])
            current_real_node = target_real
            
    return final_physical_path