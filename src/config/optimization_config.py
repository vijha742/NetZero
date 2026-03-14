"""
Optimization Configuration for Zero-Carbon City Simulation

This module defines parameters for the Genetic Algorithm (GA) and
Simulated Annealing (SA) optimization processes. It includes hardware
profiles to adjust computational parameters based on available resources.

Hardware Profiles:
- minimal: For laptops/low-power devices (i3, 8GB RAM)
- standard: For mid-range systems (i5/i7, 12-16GB RAM)
- high_performance: For workstations (i12+, RTX 3090, 16GB+ RAM)
"""

# ============================================================================
# Hardware Performance Profiles
# ============================================================================
HARDWARE_PROFILES = {
    "minimal": {
        "description": "Optimized for Intel i3 7th gen, 12GB RAM",
        "grid_size": 30,
        "population_size": 30,
        "generations": 500,
        "use_gpu": False,
        "parallel_fitness": False,
        "expected_runtime_minutes": "15-30 min",
    },
    "standard": {
        "description": "Balanced performance for mid-range systems",
        "grid_size": 50,
        "population_size": 50,
        "generations": 1000,
        "use_gpu": False,
        "parallel_fitness": True,
        "expected_runtime_minutes": "5-10 min",
    },
    "high_performance": {
        "description": "Optimized for Intel i12 + RTX 3090, 16GB RAM",
        "grid_size": 70,
        "population_size": 100,
        "generations": 2000,
        "use_gpu": True,
        "parallel_fitness": True,
        "expected_runtime_minutes": "2-5 min",
    },
}

# Default hardware profile
DEFAULT_HARDWARE_PROFILE = "standard"

# ============================================================================
# Genetic Algorithm Parameters
# ============================================================================
GA_CONFIG = {
    # Population parameters
    "population_size": 50,           # Number of candidate cities per generation
    "generations": 1000,             # Maximum number of generations
    "convergence_threshold": 100,    # Stop if no improvement for N generations
    
    # Selection parameters
    "selection_method": "tournament", # Options: "tournament", "roulette", "rank"
    "tournament_size": 5,            # Number of candidates in tournament selection
    "elitism_count": 5,              # Top N solutions preserved unchanged
    
    # Crossover parameters
    "crossover_method": "multi_point", # Options: "single_point", "multi_point", "uniform"
    "crossover_probability": 0.7,    # Probability of crossover (vs cloning parent)
    "crossover_points": 2,           # Number of crossover points for multi-point
    
    # Mutation parameters
    "mutation_method": "swap",       # Options: "swap", "random", "adaptive"
    "mutation_rate": 0.05,           # Probability of mutating each cell (5%)
    "adaptive_mutation": True,       # Increase mutation if convergence stalls
    "mutation_rate_min": 0.02,       # Minimum mutation rate (adaptive)
    "mutation_rate_max": 0.15,       # Maximum mutation rate (adaptive)
    
    # Diversity maintenance
    "diversity_threshold": 0.8,      # If >80% of population similar, force diversity
    "diversity_method": "inject_random", # Options: "inject_random", "increase_mutation"
    
    # Performance optimization
    "parallel_evaluation": True,     # Evaluate fitness in parallel
    "cache_fitness": True,           # Cache fitness for identical grids
    "early_stopping": True,          # Stop if target fitness reached
}

# ============================================================================
# Simulated Annealing Parameters
# ============================================================================
SA_CONFIG = {
    "enabled": False,                # Use SA as refinement step after GA
    "initial_temperature": 1000.0,   # Starting temperature
    "cooling_rate": 0.95,            # Temperature multiplier per iteration
    "min_temperature": 1.0,          # Stop when temperature drops below this
    "iterations_per_temp": 100,      # Iterations at each temperature level
    "neighbor_method": "local_swap", # Options: "local_swap", "random_swap", "region_swap"
    "acceptance_criterion": "metropolis", # Options: "metropolis", "threshold"
}

# ============================================================================
# Fitness Function Weights
# ============================================================================
FITNESS_WEIGHTS = {
    # Primary objectives
    "carbon": 1.0,                   # W1: Weight for carbon emissions (maximize negative)
    "happiness": 0.5,                # W2: Weight for happiness score
    "cost": 0.3,                     # W3: Weight for cost (minimize)
    
    # Constraint penalties (applied as multipliers)
    "hard_constraints": 1e6,         # Extreme penalty for invalid cities
    "soft_constraints": 1.0,         # Weight for spatial constraint violations
    
    # Success criteria
    "target_carbon_ratio": 0.05,     # Target: <5% net carbon emissions
}

# Hard constraint definitions
HARD_CONSTRAINTS = {
    "min_population": None,          # Minimum population (set by user)
    "max_budget": None,              # Maximum budget (set by user)
    "energy_balance": True,          # Energy supply must >= demand
}

# ============================================================================
# City Initialization Parameters
# ============================================================================
INIT_CONFIG = {
    "method": "random_weighted",     # Options: "random", "random_weighted", "template"
    
    # Random weighted initialization (more realistic starting cities)
    "building_probabilities": {
        0: 0.30,  # Empty land (30%)
        1: 0.15,  # Res_HD (15%)
        2: 0.10,  # Res_Eco (10%)
        3: 0.10,  # Factory (10%)
        4: 0.05,  # Coal (5%)
        5: 0.05,  # Solar (5%)
        6: 0.20,  # Forest (20%)
        7: 0.05,  # Road (5%)
    },
    
    # Template-based initialization (predefined "bad" city)
    "template_name": "coal_heavy_city",
    
    # Constraints during initialization
    "ensure_basic_viability": True,  # Ensure some population, energy, etc.
}

# ============================================================================
# Grid Configuration
# ============================================================================
GRID_CONFIG = {
    "default_size": 50,              # Default grid size (50x50)
    "min_size": 20,                  # Minimum allowed grid size
    "max_size": 100,                 # Maximum allowed grid size
    "cell_size_meters": 100,         # Each cell represents 100m x 100m
}

# ============================================================================
# Logging and Visualization
# ============================================================================
LOGGING_CONFIG = {
    "log_level": "INFO",             # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    "log_frequency": 10,             # Log every N generations
    "save_checkpoints": True,        # Save best solution periodically
    "checkpoint_frequency": 50,      # Save every N generations
    "visualize_realtime": True,      # Update visualization during optimization
    "visualization_frequency": 20,   # Update every N generations
}

# ============================================================================
# Research Experiment Presets
# ============================================================================
EXPERIMENT_PRESETS = {
    "quick_test": {
        "description": "Fast test run for debugging",
        "population_size": 20,
        "generations": 100,
        "grid_size": 30,
    },
    "standard_run": {
        "description": "Standard parameters for research paper",
        "population_size": 50,
        "generations": 1000,
        "grid_size": 50,
    },
    "high_quality": {
        "description": "High-quality results for publication figures",
        "population_size": 100,
        "generations": 2000,
        "grid_size": 70,
    },
    "sensitivity_analysis": {
        "description": "Multiple runs with varying weights",
        "population_size": 50,
        "generations": 500,
        "grid_size": 50,
        "weight_variations": [
            {"carbon": 1.0, "happiness": 0.5, "cost": 0.3},
            {"carbon": 1.0, "happiness": 0.3, "cost": 0.5},
            {"carbon": 0.8, "happiness": 0.8, "cost": 0.2},
        ],
    },
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_hardware_profile(profile_name):
    """Load a hardware profile and update relevant configs."""
    if profile_name not in HARDWARE_PROFILES:
        raise ValueError(f"Unknown hardware profile: {profile_name}. "
                        f"Available: {list(HARDWARE_PROFILES.keys())}")
    
    profile = HARDWARE_PROFILES[profile_name]
    
    # Update GA config
    GA_CONFIG["population_size"] = profile["population_size"]
    GA_CONFIG["generations"] = profile["generations"]
    GA_CONFIG["parallel_evaluation"] = profile["parallel_fitness"]
    
    # Update grid config
    GRID_CONFIG["default_size"] = profile["grid_size"]
    
    return profile

def load_experiment_preset(preset_name):
    """Load an experiment preset and update relevant configs."""
    if preset_name not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown experiment preset: {preset_name}. "
                        f"Available: {list(EXPERIMENT_PRESETS.keys())}")
    
    preset = EXPERIMENT_PRESETS[preset_name]
    
    # Update GA config
    if "population_size" in preset:
        GA_CONFIG["population_size"] = preset["population_size"]
    if "generations" in preset:
        GA_CONFIG["generations"] = preset["generations"]
    
    # Update grid config
    if "grid_size" in preset:
        GRID_CONFIG["default_size"] = preset["grid_size"]
    
    return preset

def set_fitness_weights(carbon=None, happiness=None, cost=None):
    """Update fitness function weights."""
    if carbon is not None:
        FITNESS_WEIGHTS["carbon"] = carbon
    if happiness is not None:
        FITNESS_WEIGHTS["happiness"] = happiness
    if cost is not None:
        FITNESS_WEIGHTS["cost"] = cost

def set_hard_constraints(min_population=None, max_budget=None, energy_balance=None):
    """Update hard constraints."""
    if min_population is not None:
        HARD_CONSTRAINTS["min_population"] = min_population
    if max_budget is not None:
        HARD_CONSTRAINTS["max_budget"] = max_budget
    if energy_balance is not None:
        HARD_CONSTRAINTS["energy_balance"] = energy_balance

def get_config_summary():
    """Return a formatted summary of current configuration."""
    summary = {
        "GA": {
            "Population": GA_CONFIG["population_size"],
            "Generations": GA_CONFIG["generations"],
            "Mutation Rate": GA_CONFIG["mutation_rate"],
            "Elitism": GA_CONFIG["elitism_count"],
        },
        "Grid": {
            "Size": f"{GRID_CONFIG['default_size']}x{GRID_CONFIG['default_size']}",
            "Cell Size": f"{GRID_CONFIG['cell_size_meters']}m",
        },
        "Fitness Weights": {
            "Carbon": FITNESS_WEIGHTS["carbon"],
            "Happiness": FITNESS_WEIGHTS["happiness"],
            "Cost": FITNESS_WEIGHTS["cost"],
        },
        "Hard Constraints": HARD_CONSTRAINTS,
    }
    return summary

# ============================================================================
# Research Notes
# ============================================================================
OPTIMIZATION_RESEARCH_NOTES = """
Optimization Algorithm Design Choices:

1. **Genetic Algorithm Selection**:
   - Well-suited for multi-objective, discrete optimization problems
   - Naturally handles spatial constraints (evaluated via fitness function)
   - Interpretable results (each solution is a complete city layout)
   - Comparison baseline: GA is widely used in urban planning research

2. **Parameter Tuning**:
   - Population size: Larger = more diversity but slower convergence
   - Mutation rate: Higher = more exploration, lower = exploitation
   - Elitism: Preserves best solutions (prevents regression)
   
3. **Convergence Criteria**:
   - Primary: <5% net carbon emissions achieved
   - Secondary: No improvement for 100 generations (local optimum)
   - Tertiary: Maximum generations reached

4. **Fitness Function Design**:
   Multi-objective function combines:
   - Carbon minimization (primary goal)
   - Happiness maximization (livability)
   - Cost minimization (economic feasibility)
   
   Weights (W1, W2, W3) allow exploring tradeoffs. Sensitivity analysis
   section in paper will vary these weights.

5. **Simulated Annealing (Optional Refinement)**:
   - Used after GA to fine-tune best solution
   - Local search method: may find better arrangement near GA optimum
   - Computationally cheaper for final refinement

6. **Hardware Scalability**:
   System designed to run on modest hardware (i3) for accessibility,
   but can leverage high-performance systems for:
   - Larger city grids (more realistic)
   - More generations (better convergence)
   - Multiple experiment runs (statistical significance)

7. **Research Reproducibility**:
   - All parameters documented and configurable
   - Random seeds can be fixed for reproducible results
   - Hardware profiles enable fair performance comparisons

Performance Benchmarks (Expected):
- i3 (minimal profile): 30x30 grid, 500 gen → 15-30 min
- i12+3090 (high performance): 70x70 grid, 2000 gen → 2-5 min
- Speedup factor: ~10-15x with GPU acceleration

Future Optimization Enhancements:
- Multi-objective evolutionary algorithms (NSGA-II, SPEA2)
- Parallel island GA for diversity
- Hybrid GA+reinforcement learning
- Transfer learning from previously optimized cities
"""

if __name__ == "__main__":
    # Print configuration summary
    print("=" * 80)
    print("OPTIMIZATION CONFIGURATION SUMMARY")
    print("=" * 80)
    
    print("\n--- Hardware Profiles ---")
    for name, profile in HARDWARE_PROFILES.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {profile['description']}")
        print(f"  Grid: {profile['grid_size']}x{profile['grid_size']}")
        print(f"  Population: {profile['population_size']}")
        print(f"  Generations: {profile['generations']}")
        print(f"  GPU: {'Yes' if profile['use_gpu'] else 'No'}")
        print(f"  Expected Runtime: {profile['expected_runtime_minutes']}")
    
    print("\n" + "=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    
    config = get_config_summary()
    for section, values in config.items():
        print(f"\n{section}:")
        for key, value in values.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
