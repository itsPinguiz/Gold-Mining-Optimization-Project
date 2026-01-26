from Problem import Problem
from src.algorithm import my_genetic_algorithm

def solution(P: Problem):
    """
    Entry point for the assignment.
    Calls the optimization logic defined in src/algorithm.py
    """
    # Run optimization
    city_path = my_genetic_algorithm(P)
    
    # Convert to required format: [(city, gold), ...]
    formatted_path = []
    for city in city_path:
        if city == 0:
            formatted_path.append((0, 0))
        else:
            gold = P.graph.nodes[city]['gold']
            formatted_path.append((city, gold))
    
    return formatted_path

# Optional for testing
if __name__ == "__main__":
    from src.utils import run_tests
    
    # Wrapper to extract cities for evaluation
    def solution_for_eval(P: Problem):
        formatted = solution(P)
        # Extract just cities for evaluate_path
        return [city for city, gold in formatted]
    
    run_tests(solution_for_eval)
