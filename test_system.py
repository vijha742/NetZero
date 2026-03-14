"""
Quick Test Script for Zero-Carbon City Simulation

This script tests all major components to ensure the system works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 80)
print("ZERO-CARBON CITY SIMULATION - SYSTEM TEST")
print("=" * 80)

# Test 1: Configuration
print("\n[TEST 1] Testing configuration modules...")
try:
    from src.config import building_config, constraints_config, optimization_config
    print(f"✓ Building types loaded: {building_config.NUM_BUILDING_TYPES}")
    print(f"✓ Constraints loaded: {len(constraints_config.CONSTRAINTS)}")
    print(f"✓ Hardware profiles: {list(optimization_config.HARDWARE_PROFILES.keys())}")
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    sys.exit(1)

# Test 2: City Grid
print("\n[TEST 2] Testing city grid model...")
try:
    from src.models.city_grid import CityGrid
    
    city = CityGrid(size=20)
    city.randomize(method='random_weighted', seed=42)
    counts = city.get_building_counts()
    print(f"✓ City grid created: {city.size}x{city.size}")
    print(f"✓ Building diversity: {len(counts)} different types")
except Exception as e:
    print(f"✗ City grid test failed: {e}")
    sys.exit(1)

# Test 3: Simulation Engine
print("\n[TEST 3] Testing simulation engine...")
try:
    from src.engine.simulation import calculate_metrics
    
    metrics = calculate_metrics(city)
    print(f"✓ Metrics calculated")
    print(f"  Population: {metrics.population:,}")
    print(f"  Net Carbon: {metrics.net_carbon:,.0f}")
    print(f"  Carbon Ratio: {metrics.carbon_ratio*100:.2f}%")
    print(f"  Energy Balance: {metrics.energy_balance:+,.0f}")
except Exception as e:
    print(f"✗ Simulation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Spatial Constraints
print("\n[TEST 4] Testing spatial constraints...")
try:
    from src.engine.spatial_constraints import evaluate_all_constraints
    
    constraint_results = evaluate_all_constraints(city)
    print(f"✓ Constraints evaluated: {len(constraint_results)}")
    
    total_violations = sum(r.get('violations', 0) for r in constraint_results.values())
    print(f"  Total violations: {total_violations}")
except Exception as e:
    print(f"✗ Spatial constraints test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Fitness Function
print("\n[TEST 5] Testing fitness function...")
try:
    from src.optimization.fitness import calculate_fitness
    
    fitness_result = calculate_fitness(city, min_population=1000, max_budget=500000)
    print(f"✓ Fitness calculated: {fitness_result.fitness:.2f}")
    print(f"  Viable: {fitness_result.is_viable}")
    print(f"  Success (<5%): {fitness_result.success_achieved}")
except Exception as e:
    print(f"✗ Fitness test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Genetic Algorithm
print("\n[TEST 6] Testing genetic algorithm (mini run)...")
try:
    from src.optimization.genetic_algorithm import GeneticAlgorithm
    
    ga = GeneticAlgorithm(
        grid_size=20,
        population_size=10,
        generations=20,
        min_population=500,
        max_budget=300000
    )
    
    ga.initialize_population(seed=42)
    print(f"✓ Population initialized: {len(ga.population)} cities")
    print(f"  Initial best fitness: {ga.best_fitness.fitness:.2f}")
    
    # Run a few generations
    print("\n  Running 20 generations...")
    best_city, best_fitness, history = ga.run(verbose=False)
    
    print(f"✓ Optimization complete")
    print(f"  Final best fitness: {best_fitness.fitness:.2f}")
    print(f"  Improvement: {best_fitness.fitness - ga.history['best_fitness'][0]:.2f}")
    print(f"  Final carbon ratio: {best_fitness.metrics.carbon_ratio*100:.2f}%")
    
except Exception as e:
    print(f"✗ Genetic algorithm test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Distance Utilities
print("\n[TEST 7] Testing distance utilities...")
try:
    from src.utils.distance import euclidean_distance, get_cells_by_type
    
    dist = euclidean_distance(0, 0, 3, 4)
    print(f"✓ Distance calculation works: {dist:.2f}")
    
    residential_cells = get_cells_by_type(city.grid, [1, 2])
    print(f"✓ Cell finding works: Found {len(residential_cells)} residential cells")
except Exception as e:
    print(f"✗ Distance utilities test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nThe system is ready to use!")
print("\nNext steps:")
print("1. Run 'streamlit run app.py' to launch the interactive dashboard")
print("2. Open 'research_demo.ipynb' for detailed experiments")
print("3. Customize parameters in src/config/ for your research needs")
print("\n" + "=" * 80)
