#!/usr/bin/env python3
"""
Smoke-test for all four reviewer fixes.

Avoids triggering the pre-existing spatial_constraints engine bug by
bypassing evaluate_all_constraints() where possible (using patches or
direct calls) so that the four new changes are tested cleanly.

Run from the repo root:
    python test_fixes.py
"""

import sys
import math
from unittest.mock import patch

PASS = "✓ PASS"
FAIL = "✗ FAIL"


# ─── helper: mock out the spatial constraints so the fitness engine works ──────

def _mock_constraints(city_grid):
    """Stub constraint evaluator that returns empty results (no bonuses or penalties)."""
    return {}


# ──────────────────────────────────────────────────────────────────────────────


def test_normalization_scales():
    """Normalized objectives should all land in [0, 1] for any realistic city."""
    from src.models.city_grid import CityGrid
    from src.optimization.fitness import calculate_fitness, CARBON_SCALE, COST_SCALE

    city = CityGrid(size=20)
    city.randomize(method='random_weighted', seed=42)

    with patch('src.optimization.fitness.evaluate_all_constraints', _mock_constraints):
        result = calculate_fitness(city, min_population=0, max_budget=100_000_000)

    ok = True
    ok &= 0.0 <= result.carbon_normalized    <= 1.0
    ok &= 0.0 <= result.happiness_normalized <= 1.0
    ok &= 0.0 <= result.cost_normalized      <= 1.0
    ok &= 0.0 <= result.fitness              <= 3.0  # generous upper bound

    status = PASS if ok else FAIL
    print(f"{status}  Normalization: "
          f"C_norm={result.carbon_normalized:.3f}  "
          f"H_norm={result.happiness_normalized:.3f}  "
          f"$_norm={result.cost_normalized:.3f}  "
          f"fitness={result.fitness:.4f}")
    print(f"       CARBON_SCALE={CARBON_SCALE:,.0f}  COST_SCALE={COST_SCALE:,.0f}")
    return ok


def test_hard_constraint_budget():
    """Cost > MaxBudget must produce fitness = -inf."""
    from src.models.city_grid import CityGrid
    from src.optimization.fitness import calculate_fitness

    city = CityGrid(size=20)
    city.randomize(method='random_weighted', seed=1)

    with patch('src.optimization.fitness.evaluate_all_constraints', _mock_constraints):
        result = calculate_fitness(city, min_population=0, max_budget=1)  # $1 guaranteed to fail

    ok = result.fitness == float('-inf') and not result.is_viable
    status = PASS if ok else FAIL
    print(f"{status}  Hard constraint (budget violated): fitness={result.fitness}  viable={result.is_viable}")
    return ok


def test_hard_constraint_population():
    """Population < MinPop must produce fitness = -inf."""
    from src.models.city_grid import CityGrid
    from src.optimization.fitness import calculate_fitness

    city = CityGrid(size=20)
    city.randomize(method='random_weighted', seed=2)

    with patch('src.optimization.fitness.evaluate_all_constraints', _mock_constraints):
        result = calculate_fitness(city, min_population=999_999_999, max_budget=None)

    ok = result.fitness == float('-inf') and not result.is_viable
    status = PASS if ok else FAIL
    print(f"{status}  Hard constraint (population violated): fitness={result.fitness}  viable={result.is_viable}")
    return ok


def test_hard_constraint_energy():
    """Energy supply < demand must produce fitness = -inf."""
    from src.models.city_grid import CityGrid
    from src.optimization.fitness import calculate_fitness

    city = CityGrid(size=20)
    # Fill entirely with Res_HD (type 1): energy_demand=100, energy_supply=0
    city.grid[:] = 1

    with patch('src.optimization.fitness.evaluate_all_constraints', _mock_constraints):
        result = calculate_fitness(city, min_population=0, max_budget=None,
                                   energy_balance_required=True)

    ok = result.fitness == float('-inf') and not result.is_viable
    status = PASS if ok else FAIL
    print(f"{status}  Hard constraint (energy deficit): fitness={result.fitness}  viable={result.is_viable}")
    return ok


def test_ga_block_crossover_and_point_mutation():
    """GA should run 3 generations with the new operators and emit correct config."""
    from src.optimization.genetic_algorithm import GeneticAlgorithm

    try:
        with patch('src.optimization.fitness.evaluate_all_constraints', _mock_constraints), \
             patch('src.engine.spatial_constraints.evaluate_all_constraints', _mock_constraints):
            ga = GeneticAlgorithm(
                grid_size=20, population_size=8, generations=3,
                min_population=0, max_budget=100_000_000
            )
            ga.initialize_population(seed=77)
            best, best_f, history = ga.run(verbose=False)

        assert ga.config['crossover_method'] == '2d_block', (
            f"Expected crossover_method='2d_block', got '{ga.config['crossover_method']}'")
        assert ga.config['mutation_method'] == 'point', (
            f"Expected mutation_method='point', got '{ga.config['mutation_method']}'")
        assert len(history['generation']) == 3, "Expected 3 history entries"

        print(f"{PASS}  GA operators: crossover=2d_block, mutation=point  "
              f"best_fitness={best_f.fitness}  viable={best_f.is_viable}")
        return True
    except Exception as e:
        print(f"{FAIL}  GA operators failed: {e}")
        return False


def test_nsga2_runs():
    """NSGA-II should produce a non-empty Pareto front in 3 generations."""
    from src.optimization.nsga2 import NSGA2

    try:
        with patch('src.optimization.nsga2.evaluate_all_constraints', _mock_constraints):
            nsga2 = NSGA2(
                grid_size=20, population_size=10, generations=3,
                min_population=0, max_budget=100_000_000
            )
            nsga2.initialize_population(seed=99)
            pareto, history = nsga2.run(verbose=False)

        assert len(pareto) > 0, "Pareto front must not be empty"
        assert len(history['generation']) == 3

        viable = [i for i in pareto if i.is_viable]
        print(f"{PASS}  NSGA-II: Pareto size={len(pareto)}  feasible={len(viable)}")
        return True
    except Exception as e:
        print(f"{FAIL}  NSGA-II failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_compare_utility():
    """compare_ga_vs_nsga2 should produce a string report without error."""
    from src.models.city_grid import CityGrid
    from src.optimization.fitness import calculate_fitness
    from src.optimization.nsga2 import NSGA2, compare_ga_vs_nsga2

    try:
        city = CityGrid(size=20)
        city.randomize(seed=42)

        with patch('src.optimization.fitness.evaluate_all_constraints', _mock_constraints), \
             patch('src.optimization.nsga2.evaluate_all_constraints', _mock_constraints):
            ga_fit = calculate_fitness(city, min_population=0, max_budget=100_000_000)

            nsga2 = NSGA2(grid_size=20, population_size=6, generations=2,
                          min_population=0, max_budget=100_000_000)
            nsga2.initialize_population(seed=42)
            pareto, _ = nsga2.run(verbose=False)

        report = compare_ga_vs_nsga2(ga_fit, pareto)
        assert "NSGA-II" in report and "GA" in report

        print(f"{PASS}  compare_ga_vs_nsga2 produced a {len(report)}-char report")
        return True
    except Exception as e:
        print(f"{FAIL}  compare_ga_vs_nsga2 failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Smoke-test: Reviewer Fix Validation")
    print("=" * 60)
    print()

    results = [
        test_normalization_scales(),
        test_hard_constraint_budget(),
        test_hard_constraint_population(),
        test_hard_constraint_energy(),
        test_ga_block_crossover_and_point_mutation(),
        test_nsga2_runs(),
        test_compare_utility(),
    ]

    print()
    passed = sum(results)
    total  = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll checks passed ✓")
        sys.exit(0)
    else:
        print("\nSome checks FAILED ✗")
        sys.exit(1)
