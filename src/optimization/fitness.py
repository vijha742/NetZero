"""
Fitness Function for Multi-Objective Optimization

This module defines the fitness function that the Genetic Algorithm will optimize.

The fitness function combines multiple objectives using *normalized* components
so that no single raw-magnitude term dominates the weighted sum regardless of
scale (carbon ~10⁵, cost ~10⁶, happiness 0-100).

Normalization scheme (min-max over plausible domain extremes):
  - net_carbon_norm  = clip(net_carbon / CARBON_SCALE, -1, 1)   → [-1, 1]
  - total_cost_norm  = clip(total_cost  / COST_SCALE,   0, 1)   → [ 0, 1]
  - happiness_norm   = happiness_score / 100.0                   → [ 0, 1]

Fitness = W1*(1 - net_carbon_norm) + W2*happiness_norm + W3*(1 - total_cost_norm)
          + W4*soft_constraints
          [hard-constraint violators receive fitness = -inf]

Hard Constraints (strictly enforced — violators are inviable):
  1. Population ≥ MinPop
  2. Cost ≤ MaxBudget
  3. Energy Supply ≥ Energy Demand

The CARBON_SCALE and COST_SCALE constants represent worst-case reference
values derived from the building configuration (see optimization_config.py).
"""

import sys
sys.path.append('..')

from ..engine.simulation import calculate_metrics
from ..engine.spatial_constraints import evaluate_all_constraints
from ..config.optimization_config import FITNESS_WEIGHTS, HARD_CONSTRAINTS

# ---------------------------------------------------------------------------
# Normalization reference scales
# ---------------------------------------------------------------------------
# Worst-case estimates derived from building_config.py:
#   A 50×50 grid full of Coal Plants → 5000 × 2500 = 12 500 000 carbon units
#   A 50×50 grid full of Solar Farms → 3000 × 2500 = 7 500 000 cost units
# We use conservative but realistic upper bounds for the expected range.
# These constants make the normalization transparent and reproducible.

CARBON_SCALE = 500_000.0   # Reference carbon magnitude (50×50 realistic worst case)
COST_SCALE   = 7_500_000.0 # Reference cost magnitude   (50×50 realistic worst case)


class FitnessResult:
    """Container for fitness evaluation result."""

    def __init__(self):
        self.fitness = 0.0

        # Normalized component contributions (each in [-1, 1] or [0, 1])
        self.carbon_component = 0.0
        self.happiness_component = 0.0
        self.cost_component = 0.0
        self.constraint_component = 0.0
        self.hard_penalty = 0.0

        # Raw normalized values (for diagnostic / paper reporting)
        self.carbon_normalized = 0.0
        self.cost_normalized = 0.0
        self.happiness_normalized = 0.0

        self.metrics = None
        self.constraint_results = None

        self.is_viable = False
        self.success_achieved = False

    def __repr__(self):
        return (f"FitnessResult(fitness={self.fitness:.4f}, "
                f"viable={self.is_viable}, success={self.success_achieved})")

    # Comparison operators — higher fitness is better
    def __lt__(self, other):  return self.fitness < other.fitness
    def __le__(self, other):  return self.fitness <= other.fitness
    def __gt__(self, other):  return self.fitness > other.fitness
    def __ge__(self, other):  return self.fitness >= other.fitness

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'fitness': self.fitness,
            'carbon_component': self.carbon_component,
            'happiness_component': self.happiness_component,
            'cost_component': self.cost_component,
            'constraint_component': self.constraint_component,
            'hard_penalty': self.hard_penalty,
            'carbon_normalized': self.carbon_normalized,
            'cost_normalized': self.cost_normalized,
            'happiness_normalized': self.happiness_normalized,
            'is_viable': self.is_viable,
            'success_achieved': self.success_achieved,
        }


def calculate_fitness(city_grid,
                      weights=None,
                      min_population=None,
                      max_budget=None,
                      energy_balance_required=True):
    """
    Calculate *normalized* fitness score for a city grid.

    Normalization guarantees that carbon, cost, and happiness each contribute
    comparably to the weighted sum regardless of their raw magnitudes.  The
    weights then control genuine trade-off ratios rather than being dominated
    by whichever objective happens to be numerically largest.

    Hard constraints are absolute: any violating individual receives
    fitness = -inf and is_viable = False.  Soft spatial-constraint penalties
    are added as a (small) weighted bonus/malus on top of the base fitness.

    Args:
        city_grid (CityGrid): The city layout to evaluate.
        weights (dict): Custom weights {carbon, happiness, cost, ...}.
                        If None, uses defaults from optimization_config.
        min_population (int): Hard lower bound on population (constraint 1).
        max_budget (float): Hard upper bound on total cost (constraint 2).
        energy_balance_required (bool): Hard constraint 3 — energy supply ≥ demand.

    Returns:
        FitnessResult: Complete fitness evaluation with normalized components.
    """
    result = FitnessResult()

    # Use default weights / constraints when not explicitly provided
    if weights is None:
        weights = FITNESS_WEIGHTS
    if min_population is None:
        min_population = HARD_CONSTRAINTS.get('min_population')
    if max_budget is None:
        max_budget = HARD_CONSTRAINTS.get('max_budget')

    # ------------------------------------------------------------------
    # Step 1: Evaluate spatial constraints
    # ------------------------------------------------------------------
    constraint_results = evaluate_all_constraints(city_grid)
    result.constraint_results = constraint_results

    # ------------------------------------------------------------------
    # Step 2: Compute city metrics
    # ------------------------------------------------------------------
    metrics = calculate_metrics(city_grid, constraint_results)
    result.metrics = metrics

    # ------------------------------------------------------------------
    # Step 3: Hard constraint check
    #
    #  Population ≥ MinPop, Cost ≤ MaxBudget, Energy Supply ≥ Demand
    #  A violated hard constraint makes the individual INVIABLE.
    #  Inviable individuals receive fitness = -inf so they are never
    #  selected as parents regardless of how good their soft objectives are.
    # ------------------------------------------------------------------
    result.is_viable = True

    if min_population is not None and metrics.population < min_population:
        result.is_viable = False

    if max_budget is not None and metrics.total_cost > max_budget:
        result.is_viable = False

    if energy_balance_required and metrics.energy_balance < 0:
        result.is_viable = False

    if not result.is_viable:
        result.fitness = float('-inf')
        result.hard_penalty = float('-inf')
        return result

    # ------------------------------------------------------------------
    # Step 4: Success criterion
    # ------------------------------------------------------------------
    result.success_achieved = metrics.success_achieved(
        target_carbon_ratio=weights.get('target_carbon_ratio', 0.05)
    )

    # ------------------------------------------------------------------
    # Step 5: Normalized objective components
    #
    # All three objectives are mapped to [0, 1] before weighting:
    #
    #  carbon_norm  = clip(net_carbon / CARBON_SCALE, 0, 1)
    #    → 0 means no net carbon (ideal); 1 means worst-case carbon
    #    → We *minimize* carbon, so fitness contribution = 1 - carbon_norm
    #
    #  cost_norm    = clip(total_cost / COST_SCALE, 0, 1)
    #    → 0 means free; 1 means maximum realistic cost
    #    → We *minimize* cost, so fitness contribution = 1 - cost_norm
    #
    #  happiness_norm = happiness_score / 100
    #    → 0 = miserable; 1 = perfect
    #    → We *maximize* happiness, so fitness contribution = happiness_norm
    # ------------------------------------------------------------------
    carbon_norm = min(max(metrics.net_carbon / CARBON_SCALE, 0.0), 1.0)
    cost_norm   = min(max(metrics.total_cost  / COST_SCALE,   0.0), 1.0)
    happiness_norm = metrics.happiness_score / 100.0

    result.carbon_normalized   = carbon_norm
    result.cost_normalized     = cost_norm
    result.happiness_normalized = happiness_norm

    # Weighted contributions — all are now on the same [0, 1] scale
    result.carbon_component    = weights['carbon']    * (1.0 - carbon_norm)
    result.happiness_component = weights['happiness'] * happiness_norm
    result.cost_component      = weights['cost']      * (1.0 - cost_norm)

    # ------------------------------------------------------------------
    # Step 6: Soft spatial constraint bonus/malus
    #
    # Constraint scores are typically small integers (±1 … ±20).
    # We scale them by soft_constraints weight (default 0.1) to keep
    # their aggregate contribution below the main objectives.
    # ------------------------------------------------------------------
    constraint_score = sum(
        cr.get('score', 0) for cr in constraint_results.values()
    )
    # Normalize the raw constraint score so its contribution stays in [-1, 1]:
    #   penalty_scores range from ~-100 (transit) to +25 (industrial cluster)
    #   per cell.  Worst case: all cells penalized at max magnitude.
    #   Reference = grid_cells × |worst_per_cell_penalty| = n² × 100
    grid_cells = city_grid.size ** 2
    WORST_PENALTY_PER_CELL = 100.0   # transit_access penalty_score magnitude
    constraint_ref = max(grid_cells * WORST_PENALTY_PER_CELL, 1.0)
    soft_norm = max(-1.0, min(1.0, constraint_score / constraint_ref))
    result.constraint_component = weights.get('soft_constraints', 0.1) * soft_norm

    # ------------------------------------------------------------------
    # Step 7: Total fitness (maximized by GA)
    # Range of base fitness: 0 to (W_carbon + W_happiness + W_cost)
    # ------------------------------------------------------------------
    result.fitness = (
        result.carbon_component
        + result.happiness_component
        + result.cost_component
        + result.constraint_component
    )

    return result


def evaluate_population_fitness(population, **kwargs):
    """
    Evaluate fitness for an entire population.

    Args:
        population (list[CityGrid]): Candidate city grids.
        **kwargs: Passed through to calculate_fitness().

    Returns:
        list[FitnessResult]: One result per individual.
    """
    return [calculate_fitness(city_grid, **kwargs) for city_grid in population]


def get_best_solution(population, fitness_results):
    """
    Return the best viable solution from a population.

    Viable individuals always rank above inviable ones.
    Among viables the one with highest fitness wins.

    Args:
        population (list[CityGrid]): Candidate city grids.
        fitness_results (list[FitnessResult]): Corresponding evaluations.

    Returns:
        tuple: (best_city_grid, best_fitness_result) or (None, None).
    """
    if not population:
        return None, None

    # Prefer viable solutions; fall back to least-bad inviable one
    viable = [(i, fr) for i, fr in enumerate(fitness_results) if fr.is_viable]
    if viable:
        best_idx = max(viable, key=lambda x: x[1].fitness)[0]
    else:
        best_idx = max(range(len(fitness_results)),
                       key=lambda i: (
                           fitness_results[i].metrics.population
                           if fitness_results[i].metrics else 0
                       ))

    return population[best_idx], fitness_results[best_idx]


def fitness_summary(fitness_result):
    """
    Create a human-readable summary of a fitness evaluation.

    Args:
        fitness_result (FitnessResult): Result to summarize.

    Returns:
        str: Formatted summary string.
    """
    summary = "Fitness Evaluation Summary\n"
    summary += "=" * 70 + "\n\n"

    if fitness_result.is_viable:
        summary += f"TOTAL FITNESS: {fitness_result.fitness:.4f}  (range: 0 – {sum([FITNESS_WEIGHTS.get(k,0) for k in ('carbon','happiness','cost')])})\n"
    else:
        summary += f"TOTAL FITNESS: INVIABLE (hard constraint violated)\n"

    summary += f"Viable: {'✓ YES' if fitness_result.is_viable else '✗ NO (hard constraint violated)'}\n"
    summary += f"Success (<5% carbon): {'✓ YES' if fitness_result.success_achieved else '✗ NO'}\n\n"

    summary += "NORMALIZED FITNESS COMPONENTS (each objective on 0–1 scale):\n"
    if fitness_result.is_viable:
        summary += f"  Carbon Normalized:    {fitness_result.carbon_normalized:.4f}  → component {fitness_result.carbon_component:+.4f}\n"
        summary += f"  Happiness Normalized: {fitness_result.happiness_normalized:.4f}  → component {fitness_result.happiness_component:+.4f}\n"
        summary += f"  Cost Normalized:      {fitness_result.cost_normalized:.4f}  → component {fitness_result.cost_component:+.4f}\n"
        summary += f"  Soft Constraints:                    → component {fitness_result.constraint_component:+.4f}\n"

    if fitness_result.metrics:
        m = fitness_result.metrics
        summary += "\n"
        summary += "RAW CITY METRICS:\n"
        summary += f"  Population:     {m.population:,}\n"
        summary += f"  Net Carbon:     {m.net_carbon:,.0f}  (scale ref: {CARBON_SCALE:,.0f})\n"
        summary += f"  Carbon Ratio:   {m.carbon_ratio*100:.2f}%\n"
        summary += f"  Energy Balance: {m.energy_balance:+,.0f}\n"
        summary += f"  Cost:           ${m.total_cost:,.0f}  (scale ref: ${COST_SCALE:,.0f})\n"
        summary += f"  Happiness:      {m.happiness_score:.1f}/100\n"

    summary += "\n" + "=" * 70 + "\n"
    return summary


def compare_fitness(fitness_a, fitness_b, labels=None):
    """
    Compare two fitness results side-by-side.

    Args:
        fitness_a, fitness_b (FitnessResult): Results to compare.
        labels (tuple): Optional (label_a, label_b).

    Returns:
        str: Formatted comparison.
    """
    if labels is None:
        labels = ("A", "B")

    label_a, label_b = labels

    comparison = "Fitness Comparison\n"
    comparison += "=" * 80 + "\n\n"

    comparison += f"{'Metric':<30s} {label_a:>15s} {label_b:>15s} {'Difference':>15s}\n"
    comparison += "-" * 80 + "\n"

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and abs(v) < 1e9 else "INVIABLE"

    diff_fitness = (fitness_b.fitness - fitness_a.fitness
                    if (fitness_a.is_viable and fitness_b.is_viable) else float('nan'))

    comparison += f"{'Total Fitness':<30s} {_fmt(fitness_a.fitness):>15s} {_fmt(fitness_b.fitness):>15s}\n"
    comparison += f"{'  Carbon Component (norm.)':<30s} {fitness_a.carbon_component:>15.4f} {fitness_b.carbon_component:>15.4f}\n"
    comparison += f"{'  Happiness Component (norm.)':<30s} {fitness_a.happiness_component:>15.4f} {fitness_b.happiness_component:>15.4f}\n"
    comparison += f"{'  Cost Component (norm.)':<30s} {fitness_a.cost_component:>15.4f} {fitness_b.cost_component:>15.4f}\n"

    comparison += "\n"

    if fitness_a.metrics and fitness_b.metrics:
        ma, mb = fitness_a.metrics, fitness_b.metrics
        comparison += f"{'Population':<30s} {ma.population:>15,} {mb.population:>15,}\n"
        comparison += f"{'Net Carbon (raw)':<30s} {ma.net_carbon:>15,.0f} {mb.net_carbon:>15,.0f}\n"
        comparison += f"{'Carbon Ratio':<30s} {ma.carbon_ratio*100:>14.2f}% {mb.carbon_ratio*100:>14.2f}%\n"
        comparison += f"{'Energy Balance':<30s} {ma.energy_balance:>15,.0f} {mb.energy_balance:>15,.0f}\n"
        comparison += f"{'Total Cost (raw)':<30s} {ma.total_cost:>15,.0f} {mb.total_cost:>15,.0f}\n"
        comparison += f"{'Happiness':<30s} {ma.happiness_score:>15.1f} {mb.happiness_score:>15.1f}\n"

    comparison += "\n" + "=" * 80 + "\n"

    if fitness_a.is_viable and fitness_b.is_viable:
        if fitness_b.fitness > fitness_a.fitness:
            comparison += f"Winner: {label_b} (by {diff_fitness:.4f} fitness units)\n"
        elif fitness_a.fitness > fitness_b.fitness:
            comparison += f"Winner: {label_a} (by {-diff_fitness:.4f} fitness units)\n"
        else:
            comparison += "Tie\n"
    else:
        a_status = "viable" if fitness_a.is_viable else "INVIABLE"
        b_status = "viable" if fitness_b.is_viable else "INVIABLE"
        comparison += f"Note: {label_a} is {a_status}, {label_b} is {b_status}\n"

    return comparison


if __name__ == "__main__":
    from ..models.city_grid import CityGrid

    print("Testing Normalized Fitness Function...\n")
    print(f"Carbon scale reference: {CARBON_SCALE:,.0f}")
    print(f"Cost scale reference:   {COST_SCALE:,.0f}\n")

    city1 = CityGrid(size=20)
    city1.randomize(method='random_weighted', seed=42)

    city2 = CityGrid(size=20)
    city2.randomize(method='random_weighted', seed=123)

    print("Evaluating City 1...")
    fitness1 = calculate_fitness(city1, min_population=1000, max_budget=1_000_000)
    print(fitness_summary(fitness1))

    print("\nEvaluating City 2...")
    fitness2 = calculate_fitness(city2, min_population=1000, max_budget=1_000_000)
    print(fitness_summary(fitness2))

    print("\nComparing Cities...")
    print(compare_fitness(fitness1, fitness2, labels=("City 1", "City 2")))
