#!/usr/bin/env python3
"""
Quick optimization test to validate the GA achieves <5% carbon target.
"""

import sys
from src.optimization.genetic_algorithm import GeneticAlgorithm
from src.visualization.city_map import create_before_after_comparison
from src.visualization.metrics_plots import plot_fitness_evolution
import matplotlib.pyplot as plt

print("=" * 80)
print("ZERO-CARBON CITY OPTIMIZATION - QUICK TEST")
print("=" * 80)
print("\nRunning optimization on 30x30 grid...")
print("Population: 50, Generations: 500")
print("Target: <5% carbon ratio\n")

# Initialize GA with reasonable parameters for quick test
ga = GeneticAlgorithm(
    grid_size=30,
    population_size=50,
    generations=500,
    min_population=5000,      # Minimum viable city population
    max_budget=2000000        # Maximum construction budget
)

print("Initializing population...")
ga.initialize_population(seed=42)

print(f"Initial best fitness: {ga.best_fitness.fitness:.2f}")
print(f"Initial carbon ratio: {ga.best_fitness.metrics.carbon_ratio*100:.2f}%\n")

# Run optimization
print("Running optimization (this may take 2-5 minutes)...")
best_city, best_fitness, history = ga.run(verbose=True)

# Print results
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE!")
print("=" * 80)

print(f"\nFinal Metrics:")
print(f"  Fitness Score: {best_fitness.fitness:.2f}")
print(f"  Carbon Ratio: {best_fitness.metrics.carbon_ratio*100:.2f}%")
print(f"  Population: {best_fitness.metrics.population:,}")
print(f"  Total Cost: ${best_fitness.metrics.total_cost:,.0f}")
print(f"  Happiness Score: {best_fitness.metrics.happiness_score:.2f}")
print(f"  Energy Balance: {best_fitness.metrics.energy_balance:,.0f} kWh/year")

# Check if target achieved
if best_fitness.metrics.carbon_ratio < 0.05:
    print(f"\n✓ SUCCESS! Achieved <5% carbon target ({best_fitness.metrics.carbon_ratio*100:.2f}%)")
else:
    print(f"\n⚠ Target not reached. Current: {best_fitness.metrics.carbon_ratio*100:.2f}% (need <5%)")
    print("  Try increasing generations or population size for better results.")

print("\nGenerating visualizations...")

# Create visualizations
fig_comparison, _ = create_before_after_comparison(ga.population[0], best_city)
fig_comparison.savefig('results_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results_comparison.png")

fig_evolution, _ = plot_fitness_evolution(history)
fig_evolution.savefig('fitness_evolution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: fitness_evolution.png")

print("\n" + "=" * 80)
print("Test complete! Check the generated PNG files for visual results.")
print("=" * 80)
