import time
import sys
import logging
from pathlib import Path
from datetime import datetime

# Importa Problem dalla root
sys.path.insert(0, str(Path(__file__).parent.parent))
from Problem import Problem

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "src/logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name="test_runner"):
    """Setup logger that writes only to file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"test_run_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger, log_file

# Professor's Baseline Results
# Format: (N, Density, Alpha, Beta): Cost
BASELINES = {
    # N=100 Baselines
    (100, 0.2, 1, 1): 25266.41,
    (100, 0.2, 2, 1): 50425.31,
    (100, 0.2, 1, 2): 5334401.93,
    (100, 1, 1, 1): 18266.19,
    (100, 1, 2, 1): 36457.92,
    (100, 1, 1, 2): 5404978.09,
    
    # N=1000 Baselines
    (1000, 0.2, 1, 1): 195402.96,
    (1000, 0.2, 2, 1): 390028.72,
    (1000, 0.2, 1, 2): 37545927.70,
    (1000, 1, 1, 1): 192936.23,
    (1000, 1, 2, 1): 385105.64,
    (1000, 1, 1, 2): 57580018.87
}

def normalize_path(path):
    """
    Normalize path to list of cities.
    Handles both formats:
    - List of tuples: [(city, gold), ...]
    - List of cities: [city1, city2, ...]
    """
    if not path:
        return path
    
    if isinstance(path[0], tuple):
        return [city for city, gold in path]
    return path

def run_tests(solution_func, test_cases=None):
    """
    Runs the solution function against the standard test cases 
    and compares with baselines.
    
    Args:
        solution_func: Function that takes a Problem instance and returns a path
        test_cases: Optional list of (n, density, alpha, beta) tuples. 
                   If None, uses default test cases.
    """
    test_cases = [
        # N=100
        (100, 0.2, 1, 1), (100, 0.2, 2, 1), (100, 0.2, 1, 2),
        (100, 1, 1, 1),   (100, 1, 2, 1),   (100, 1, 1, 2),
        
        # N=1000
        (1000, 0.2, 1, 1), (1000, 0.2, 2, 1), (1000, 0.2, 1, 2),
        (1000, 1, 1, 1),   (1000, 1, 2, 1),   (1000, 1, 1, 2),

        # Edge Cases
        # alpha or beta = 0
        (100, 0.2, 0, 1), (100, 0.2, 1, 0),
        (1000, 1, 0, 1), (1000, 1, 1, 0)   
    ]
    
    logger.info(f"Test Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 110)
    logger.info(f"{'N':<6} | {'Dens':<5} | {'Alpha':<5} | {'Beta':<5} | {'Cost':<15} | {'Baseline':<12} | {'vs Base %':<10} | {'Time':<8} | {'Status'}")
    logger.info("-" * 110)
    
    total_diff = 0
    count = 0
    results = []
    
    for n, dens, alpha, beta in test_cases:
        # Create a specific problem instance
        p = Problem(n, density=dens, alpha=alpha, beta=beta, seed=42)
        
        start = time.time()
        try:
            path = solution_func(p)
            elapsed = time.time() - start
            
            # Normalize path format (handle both tuples and plain cities)
            city_path = normalize_path(path)
            
            # Validate and calculate cost
            cost, status = p.evaluate_path(city_path)
            
            # Compare with baseline
            baseline = BASELINES.get((n, dens, alpha, beta), None)
            baseline_str = f"{baseline:.2f}" if baseline else "N/A"
            
            if baseline and cost != float('inf'):
                diff_pct = ((cost - baseline) / baseline) * 100
                diff_str = f"{diff_pct:+.2f}%"
                total_diff += diff_pct
                count += 1
            else:
                diff_pct = None
                diff_str = "N/A"
            
            result = {
                'n': n, 'density': dens, 'alpha': alpha, 'beta': beta,
                'cost': cost, 'baseline': baseline, 'diff_pct': diff_pct,
                'time': elapsed, 'status': status
            }
            results.append(result)
            
            logger.info(f"{n:<6} | {dens:<5} | {alpha:<5} | {beta:<5} | {cost:<15.2f} | {baseline_str:<12} | {diff_str:<10} | {elapsed:<8.2f} | {status}")
            
        except Exception as e:
            result = {
                'n': n, 'density': dens, 'alpha': alpha, 'beta': beta,
                'cost': float('inf'), 'baseline': None, 'diff_pct': None,
                'time': 0, 'status': f"ERROR: {e}"
            }
            results.append(result)
            
            logger.info(f"{n:<6} | {dens:<5} | {alpha:<5} | {beta:<5} | {'ERROR':<15} | {'N/A':<12} | {'N/A':<10} | {0.0:<8} | {e}")
            import traceback
            logger.info(traceback.format_exc())

    logger.info("-" * 110)
    
    if count > 0:
        avg_diff = total_diff / count
        logger.info(f"Average Improvement vs Baseline: {avg_diff:+.2f}% (Negative is better)")
        
        # Summary statistics
        valid_results = [r for r in results if r['cost'] != float('inf')]
        if valid_results:
            total_time = sum(r['time'] for r in valid_results)
            logger.info(f"Total Time: {total_time:.2f}s | Valid Tests: {len(valid_results)}/{len(test_cases)}")
    
    logger.info("=" * 110)
    logger.info(f"Log saved to: {log_file}")
    
    return results, log_file