"""
Optimization package for Zero-Carbon City Simulation.

Exports:
    GeneticAlgorithm  — weighted-sum GA (primary optimizer)
    NSGA2             — NSGA-II multi-objective baseline
    calculate_fitness — normalized fitness function
    compare_ga_vs_nsga2 — comparison utility for paper tables
"""

from .genetic_algorithm import GeneticAlgorithm
from .fitness import (
    calculate_fitness,
    evaluate_population_fitness,
    get_best_solution,
    fitness_summary,
    compare_fitness,
    CARBON_SCALE,
    COST_SCALE,
)
from .nsga2 import (
    NSGA2,
    Individual,
    compare_ga_vs_nsga2,
    non_dominated_sort,
    crowding_distance_assignment,
)

__all__ = [
    "GeneticAlgorithm",
    "NSGA2",
    "Individual",
    "calculate_fitness",
    "evaluate_population_fitness",
    "get_best_solution",
    "fitness_summary",
    "compare_fitness",
    "compare_ga_vs_nsga2",
    "non_dominated_sort",
    "crowding_distance_assignment",
    "CARBON_SCALE",
    "COST_SCALE",
]
