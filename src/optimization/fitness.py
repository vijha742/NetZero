"""
Fitness Function for Multi-Objective Optimization (NumPy/CPU backend)

This module defines the fitness function that the Genetic Algorithm will optimize.

The fitness function combines multiple objectives using *normalized* components
so that no single raw-magnitude term dominates the weighted sum regardless of
scale (carbon ~10⁵, cost ~10⁶, happiness 0-100).

Normalization scheme (min-max over plausible domain extremes):
  - net_carbon_norm  = clip(net_carbon / CARBON_SCALE, 0, 1)   → [0, 1]
  - total_cost_norm  = clip(total_cost  / COST_SCALE,   0, 1)  → [0, 1]
  - happiness_norm   = happiness_score / 100.0                  → [0, 1]

Fitness = W1*(1 - net_carbon_norm) + W2*happiness_norm + W3*(1 - total_cost_norm)
          + W4*soft_constraints_norm
          [hard-constraint violators receive fitness = -inf]

Hard Constraints (strictly enforced — violators are inviable):
  1. Population ≥ MinPop
  2. Total Cost  ≤ MaxBudget
  3. Energy Supply ≥ Energy Demand

Reference scales:
  CARBON_SCALE = 500 000  — based on 50×50 grid with ~10% coal density
  COST_SCALE   = 7 500 000 — based on 50×50 grid fully covered with solar farms
"""

import sys
import math

sys.path.append('..')

from ..engine.simulation import calculate_metrics
from ..engine.spatial_constraints import evaluate_all_constraints
from ..config.optimization_config import FITNESS_WEIGHTS, HARD_CONSTRAINTS

# ---------------------------------------------------------------------------
# Normalization reference constants
# ---------------------------------------------------------------------------
# CARBON_SCALE: upper-bound reference for net_carbon.
#   50×50 grid × ~10% coal density × 5000 carbon/cell = 625 000 → round to 500 000
CARBON_SCALE = 500_000.0

# COST_SCALE: upper-bound reference for total_cost.
#   50×50 = 2500 cells × solar farm cost ~3000 $/cell = 7 500 000
COST_SCALE = 7_500_000.0


class FitnessResult:
    """Container for fitness evaluation result."""

    def __init__(self):
        self.fitness = 0.0

        # Normalized components (each on [0, 1] before weighting)
        self.carbon_normalized    = 0.0
        self.happiness_normalized = 0.0
        self.cost_normalized      = 0.0

        # Weighted contributions to total fitness
        self.carbon_component    = 0.0
        self.happiness_component = 0.0
        self.cost_component      = 0.0
        self.constraint_component = 0.0
        self.hard_penalty         = 0.0

        self.metrics = None
        self.constraint_results = None

        self.is_viable       = False
        self.success_achieved = False

    def __repr__(self):
        return (f"FitnessResult(fitness={self.fitness}, viable={self.is_viable}, "
                f"success={self.success_achieved})")

    def __lt__(self, other):
        """Allow sorting by fitness (higher is better)."""
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'fitness':              self.fitness,
            'carbon_component':    self.carbon_component,
            'happiness_component': self.happiness_component,
            'cost_component':      self.cost_component,
            'constraint_component': self.constraint_component,
            'hard_penalty':        self.hard_penalty,
            'carbon_normalized':   self.carbon_normalized,
            'happiness_normalized': self.happiness_normalized,
            'cost_normalized':     self.cost_normalized,
            'is_viable':           self.is_viable,
            'success_achieved':    self.success_achieved,
        }


def calculate_fitness(city_grid,
                      weights=None,
                      min_population=None,
                      max_budget=None,
                      energy_balance_required=True):
    """
    Calculate normalized fitness score for a city grid.

    Hard-constraint violators receive fitness = -inf and are never
    selected as parents, ensuring the algorithm only propagates
    physically feasible city layouts.

    Args:
        city_grid (CityGrid): The city layout to evaluate.
        weights (dict): Custom weights {carbon, happiness, cost, ...}.
                        If None, uses FITNESS_WEIGHTS from config.
        min_population (int): Minimum required population (hard constraint).
        max_budget (float): Maximum allowed total cost (hard constraint).
        energy_balance_required (bool): Whether energy supply ≥ demand.

    Returns:
        FitnessResult: Complete fitness evaluation with normalized components.
    """
    result = FitnessResult()

    # ------------------------------------------------------------------
    # Step 1: Resolve weights and hard-constraint thresholds
    # ------------------------------------------------------------------
    if weights is None:
        weights = FITNESS_WEIGHTS

    if min_population is None:
        min_population = HARD_CONSTRAINTS.get('min_population')
    if max_budget is None:
        max_budget = HARD_CONSTRAINTS.get('max_budget')
    if energy_balance_required is None:
        energy_balance_required = HARD_CONSTRAINTS.get('energy_balance', True)

    # ------------------------------------------------------------------
    # Step 2: Evaluate spatial constraints
    # ------------------------------------------------------------------
    constraint_results = evaluate_all_constraints(city_grid)
    result.constraint_results = constraint_results

    # ------------------------------------------------------------------
    # Step 3: Calculate city metrics (including constraint effects)
    # ------------------------------------------------------------------
    metrics = calculate_metrics(city_grid, constraint_results)
    result.metrics = metrics

    # ------------------------------------------------------------------
    # Step 4: Hard constraint enforcement
    #
    # Any violation → fitness = -inf (inviable individual).
    # Selection logic must always prefer viable over inviable.
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
        return result

    # ------------------------------------------------------------------
    # Step 5: Check success criterion
    # ------------------------------------------------------------------
    result.success_achieved = metrics.success_achieved(
        target_carbon_ratio=weights.get('target_carbon_ratio', 0.05)
    )

    # ------------------------------------------------------------------
    # Step 6: Normalize objectives to [0, 1]
    #
    #  carbon_norm  = clip(net_carbon / CARBON_SCALE, 0, 1)
    #    → 0 means no net carbon (ideal); 1 means worst-case carbon
    #    → We minimize carbon: fitness contribution = 1 - carbon_norm
    #
    #  cost_norm    = clip(total_cost / COST_SCALE, 0, 1)
    #    → We minimize cost: fitness contribution = 1 - cost_norm
    #
    #  happiness_norm = happiness_score / 100
    #    → We maximize happiness: fitness contribution = happiness_norm
    # ------------------------------------------------------------------
    carbon_norm    = min(max(metrics.net_carbon / CARBON_SCALE, 0.0), 1.0)
    cost_norm      = min(max(metrics.total_cost  / COST_SCALE,  0.0), 1.0)
    happiness_norm = metrics.happiness_score / 100.0

    result.carbon_normalized    = carbon_norm
    result.cost_normalized      = cost_norm
    result.happiness_normalized = happiness_norm

    # Weighted contributions — all on the same [0, 1] scale
    result.carbon_component    = weights['carbon']    * (1.0 - carbon_norm)
    result.happiness_component = weights['happiness'] * happiness_norm
    result.cost_component      = weights['cost']      * (1.0 - cost_norm)

    # ------------------------------------------------------------------
    # Step 7: Soft spatial constraint bonus/malus (normalized)
    #
    # Penalty magnitudes: −100 (transit) to +25 (industrial cluster) per cell.
    # Worst case: all n² cells penalized at max magnitude.
    # We normalize by grid_area × 100 and clamp to [−1, 1] so the
    # constraint component never dominates the three main objectives.
    # ------------------------------------------------------------------
    constraint_score = sum(
        cr.get('score', 0) for cr in constraint_results.values()
    )
    grid_cells     = city_grid.size ** 2
    constraint_ref = max(grid_cells * 100.0, 1.0)
    soft_norm      = max(-1.0, min(1.0, constraint_score / constraint_ref))
    result.constraint_component = weights.get('soft_constraints', 0.1) * soft_norm

    # ------------------------------------------------------------------
    # Step 8: Total fitness
    # Range: 0 to (W_carbon + W_happiness + W_cost + W_soft)
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
    Evaluate fitness for a population of city grids.

    Args:
        population (list): List of CityGrid objects.
        **kwargs: Arguments passed to calculate_fitness().

    Returns:
        list[FitnessResult]: Same order as population.
    """
    return [calculate_fitness(city_grid, **kwargs) for city_grid in population]


def get_best_solution(population, fitness_results):
    """
    Get the best solution from a population.

    Viable individuals always rank above inviable ones.

    Args:
        population (list): List of CityGrid objects.
        fitness_results (list[FitnessResult]): Corresponding results.

    Returns:
        tuple: (best_city_grid, best_fitness_result)
    """
    if not population:
        return None, None

    viable   = [(i, f) for i, f in enumerate(fitness_results) if f.is_viable]
    inviable = [(i, f) for i, f in enumerate(fitness_results) if not f.is_viable]

    pool = viable if viable else inviable
    best_idx = max(pool, key=lambda t: t[1].fitness if math.isfinite(t[1].fitness)
                   else float('-inf'))[0]

    return population[best_idx], fitness_results[best_idx]


def fitness_summary(fitness_result):
    """
    Create a human-readable summary of fitness evaluation.

    Args:
        fitness_result (FitnessResult): Fitness result to summarize.

    Returns:
        str: Formatted summary.
    """
    summary  = "Fitness Evaluation Summary\n"
    summary += "=" * 70 + "\n\n"

    summary += f"TOTAL FITNESS: {fitness_result.fitness:.4f}  (range: 0 – 1.8)\n"
    summary += f"Viable: {'✓ YES' if fitness_result.is_viable else '✗ NO'}\n"
    summary += f"Success (<5% carbon): {'✓ YES' if fitness_result.success_achieved else '✗ NO'}\n\n"

    summary += "NORMALIZED FITNESS COMPONENTS (each objective on 0–1 scale):\n"
    summary += f"  Carbon Normalized:    {fitness_result.carbon_normalized:.4f}  → component {fitness_result.carbon_component:+.4f}\n"
    summary += f"  Happiness Normalized: {fitness_result.happiness_normalized:.4f}  → component {fitness_result.happiness_component:+.4f}\n"
    summary += f"  Cost Normalized:      {fitness_result.cost_normalized:.4f}  → component {fitness_result.cost_component:+.4f}\n"
    summary += f"  Soft Constraints:                    → component {fitness_result.constraint_component:+.4f}\n"

    if fitness_result.metrics:
        m = fitness_result.metrics
        summary += "\nRAW CITY METRICS:\n"
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
        fitness_a (FitnessResult): First fitness result.
        fitness_b (FitnessResult): Second fitness result.
        labels (tuple): Optional labels for A and B (default: "A", "B").

    Returns:
        str: Formatted comparison.
    """
    if labels is None:
        labels = ("A", "B")

    label_a, label_b = labels

    comparison  = "Fitness Comparison\n"
    comparison += "=" * 80 + "\n\n"

    comparison += f"{'Metric':<25s} {label_a:>15s} {label_b:>15s} {'Difference':>15s}\n"
    comparison += "-" * 80 + "\n"

    diff_fitness = fitness_b.fitness - fitness_a.fitness
    comparison += (f"{'Total Fitness':<25s} {fitness_a.fitness:>15.4f} "
                   f"{fitness_b.fitness:>15.4f} {diff_fitness:>+15.4f}\n")

    for label, fa, fb in [
        ("  Carbon Component",    fitness_a.carbon_component,    fitness_b.carbon_component),
        ("  Happiness Component", fitness_a.happiness_component, fitness_b.happiness_component),
        ("  Cost Component",      fitness_a.cost_component,      fitness_b.cost_component),
    ]:
        diff = fb - fa
        comparison += f"{label:<25s} {fa:>15.4f} {fb:>15.4f} {diff:>+15.4f}\n"

    comparison += "\n"

    if fitness_a.metrics and fitness_b.metrics:
        ma, mb = fitness_a.metrics, fitness_b.metrics
        for label, va, vb, fmt in [
            ("Population",   ma.population,       mb.population,      "{:,.0f}"),
            ("Net Carbon",   ma.net_carbon,        mb.net_carbon,      "{:,.0f}"),
            ("Energy Bal.",  ma.energy_balance,    mb.energy_balance,  "{:+,.0f}"),
            ("Happiness",    ma.happiness_score,   mb.happiness_score, "{:.1f}"),
            ("Cost ($)",     ma.total_cost,        mb.total_cost,      "{:,.0f}"),
        ]:
            comparison += (f"{label:<25s} {fmt.format(va):>15s} "
                           f"{fmt.format(vb):>15s}\n")

    comparison += "\n" + "=" * 80 + "\n"
    diff_fitness = fitness_b.fitness - fitness_a.fitness
    if fitness_b.fitness > fitness_a.fitness:
        comparison += f"Winner: {label_b} (by {diff_fitness:.4f} points)\n"
    elif fitness_a.fitness > fitness_b.fitness:
        comparison += f"Winner: {label_a} (by {-diff_fitness:.4f} points)\n"
    else:
        comparison += "Tie\n"

    return comparison


if __name__ == "__main__":
    from ..models.city_grid import CityGrid

    print("Testing Fitness Function (NumPy backend)...\n")

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
