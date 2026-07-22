"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II
Baseline Multi-Objective Optimizer for City Planning

This module implements NSGA-II as a baseline to compare against the weighted-sum
GA formulation.  Both optimizers share the same city grid representation,
building configuration, and simulation engine, so results are directly comparable.

NSGA-II treats the three objectives as truly independent:
    1. Minimise net_carbon
    2. Maximise happiness_score
    3. Minimise total_cost

Rather than collapsing them into a single scalar fitness, NSGA-II maintains a
Pareto front of non-dominated solutions, enabling principled trade-off analysis
without choosing weights a priori.

Hard constraints are enforced identically to the GA:
    • Population ≥ MinPop   → infeasible individuals ranked last
    • Cost ≤ MaxBudget      → infeasible individuals ranked last
    • Energy Supply ≥ Demand → infeasible individuals ranked last

Infeasible individuals are always dominated by all feasible individuals.

Algorithm:
    1. Initialise population P(0)
    2. For t = 0 … T:
        a. Create offspring Q(t) via tournament selection + 2D-block crossover
           + point mutation  (same operators as the improved GA)
        b. Combine R(t) = P(t) ∪ Q(t)
        c. Non-dominated sort R(t) into fronts F1 ⊇ F2 ⊇ …
        d. Assign crowding distance within each front
        e. Select best N individuals for P(t+1) by (rank, −crowding_distance)
    3. Return Pareto-front F1 of P(T)

Reference:
    Deb et al. (2002). "A fast and elitist multiobjective genetic algorithm:
    NSGA-II." IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
"""

import random
import math
import numpy as np
from copy import deepcopy

from ..models.city_grid import CityGrid
from ..engine.simulation import calculate_metrics
from ..engine.spatial_constraints import evaluate_all_constraints
from ..config.optimization_config import GA_CONFIG, HARD_CONSTRAINTS, INIT_CONFIG
from ..utils.array_backend import xp, to_numpy


# ---------------------------------------------------------------------------
# Individual wrapper
# ---------------------------------------------------------------------------

class Individual:
    """Wraps a CityGrid with multi-objective evaluation results."""

    def __init__(self, city_grid: CityGrid):
        self.city_grid = city_grid

        # Raw objectives (set by evaluate())
        self.carbon     = None   # net_carbon    (minimise)
        self.happiness  = None   # happiness_score (maximise, stored as −happiness to minimise)
        self.cost       = None   # total_cost    (minimise)
        self.population_count = None
        self.energy_balance   = None
        self.is_viable  = False

        # NSGA-II attributes
        self.rank              = None    # Pareto front rank (1 = best)
        self.crowding_distance = 0.0

    # ------------------------------------------------------------------
    def evaluate(self, min_population=None, max_budget=None,
                 energy_balance_required=True):
        """
        Compute objectives and check hard constraints.

        Infeasible individuals receive rank = ∞ (worst) during sorting.
        """
        constraint_results = evaluate_all_constraints(self.city_grid)
        metrics = calculate_metrics(self.city_grid, constraint_results)

        self.carbon    = float(metrics.net_carbon)
        self.happiness = float(metrics.happiness_score)   # higher is better
        self.cost      = float(metrics.total_cost)
        self.population_count = int(metrics.population)
        self.energy_balance   = float(metrics.energy_balance)

        # Hard constraints
        self.is_viable = True
        if min_population is not None and self.population_count < min_population:
            self.is_viable = False
        if max_budget is not None and self.cost > max_budget:
            self.is_viable = False
        if energy_balance_required and self.energy_balance < 0:
            self.is_viable = False

    @property
    def objectives(self):
        """Return objectives as (carbon, −happiness, cost) — all minimised."""
        return (self.carbon, -self.happiness, self.cost)

    def dominates(self, other: "Individual") -> bool:
        """
        Return True if self Pareto-dominates other.

        Feasibility rule: feasible always dominates infeasible.
        """
        if self.is_viable and not other.is_viable:
            return True
        if not self.is_viable and other.is_viable:
            return False
        if not self.is_viable and not other.is_viable:
            return False   # two infeasible: no dominance

        a, b = self.objectives, other.objectives
        # self dominates other iff no worse on all and strictly better on ≥1
        at_least_as_good = all(ai <= bi for ai, bi in zip(a, b))
        strictly_better  = any(ai < bi  for ai, bi in zip(a, b))
        return at_least_as_good and strictly_better

    def copy(self) -> "Individual":
        ind = Individual(self.city_grid.copy())
        ind.carbon    = self.carbon
        ind.happiness = self.happiness
        ind.cost      = self.cost
        ind.population_count = self.population_count
        ind.energy_balance   = self.energy_balance
        ind.is_viable = self.is_viable
        ind.rank      = self.rank
        ind.crowding_distance = self.crowding_distance
        return ind

    def __repr__(self):
        return (f"Individual(rank={self.rank}, viable={self.is_viable}, "
                f"C={self.carbon:.0f}, H={self.happiness:.1f}, $={self.cost:.0f})")


# ---------------------------------------------------------------------------
# Non-dominated sort (fast sort from Deb 2002)
# ---------------------------------------------------------------------------

def non_dominated_sort(population: list) -> list:
    """
    Partition population into Pareto fronts F1, F2, …

    Returns:
        list[list[Individual]]: Ordered list of fronts; F[0] is the best.
    """
    n = len(population)
    domination_count = [0] * n       # number of individuals that dominate i
    dominated_set    = [[] for _ in range(n)]  # individuals dominated by i
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if population[i].dominates(population[j]):
                dominated_set[i].append(j)
            elif population[j].dominates(population[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i].rank = 1
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = k + 2
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    # Remove empty trailing front
    return [[population[i] for i in front] for front in fronts if front]


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------

def crowding_distance_assignment(front: list):
    """
    Assign crowding distances to all individuals in a Pareto front.

    Boundary individuals receive ∞.  Interior individuals receive the
    sum of normalised objective-space distances to their neighbours.

    Args:
        front (list[Individual]): Individuals on the same Pareto front.
    """
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return

    n_obj = 3  # carbon, −happiness, cost
    for ind in front:
        ind.crowding_distance = 0.0

    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda x: x.objectives[m])
        obj_min = sorted_front[0].objectives[m]
        obj_max = sorted_front[-1].objectives[m]
        obj_range = obj_max - obj_min if obj_max != obj_min else 1.0

        sorted_front[0].crowding_distance  = float('inf')
        sorted_front[-1].crowding_distance = float('inf')

        for k in range(1, len(sorted_front) - 1):
            sorted_front[k].crowding_distance += (
                (sorted_front[k+1].objectives[m] - sorted_front[k-1].objectives[m])
                / obj_range
            )


# ---------------------------------------------------------------------------
# Crowded comparison operator
# ---------------------------------------------------------------------------

def crowded_compare(a: Individual, b: Individual) -> bool:
    """Return True if a is preferred over b (lower rank, or same rank and larger distance)."""
    if a.rank != b.rank:
        return a.rank < b.rank
    return a.crowding_distance > b.crowding_distance


# ---------------------------------------------------------------------------
# Genetic operators (mirrors the improved GA operators)
# ---------------------------------------------------------------------------

def _block_crossover(parent1: Individual, parent2: Individual,
                     grid_size: int) -> tuple:
    """2D block crossover — identical to GA._block_crossover."""
    child1 = Individual(parent1.city_grid.copy())
    child2 = Individual(parent2.city_grid.copy())

    n = grid_size
    min_block = max(2, n // 5)
    max_block = max(min_block + 1, n * 3 // 5)

    h = random.randint(min_block, max_block)
    w = random.randint(min_block, max_block)
    r0 = random.randint(0, n - h)
    c0 = random.randint(0, n - w)

    child1.city_grid.grid[r0:r0+h, c0:c0+w] = parent2.city_grid.grid[r0:r0+h, c0:c0+w]
    child2.city_grid.grid[r0:r0+h, c0:c0+w] = parent1.city_grid.grid[r0:r0+h, c0:c0+w]

    return child1, child2


def _point_mutation(ind: Individual, mutation_rate: float):
    """Point mutation — identical to GA._point_mutation."""
    probs_dict = INIT_CONFIG['building_probabilities']
    building_types = list(probs_dict.keys())
    weights_list   = list(probs_dict.values())
    total_w = sum(weights_list)
    weights_norm = [w / total_w for w in weights_list]

    grid_size = ind.city_grid.size
    num_mutations = int(grid_size * grid_size * mutation_rate)
    for _ in range(num_mutations):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        ind.city_grid.grid[x, y] = random.choices(building_types, weights=weights_norm, k=1)[0]


def _binary_tournament(population: list) -> Individual:
    """
    Binary tournament selection using the crowded-comparison operator.
    Selects the better of two random individuals.
    """
    a, b = random.sample(population, 2)
    return a if crowded_compare(a, b) else b


# ---------------------------------------------------------------------------
# NSGA-II class
# ---------------------------------------------------------------------------

class NSGA2:
    """
    NSGA-II optimizer for zero-carbon city planning.

    Produces a Pareto front instead of a single weighted-sum solution,
    enabling direct trade-off analysis between carbon, happiness, and cost.
    """

    def __init__(self,
                 grid_size=50,
                 population_size=None,
                 generations=None,
                 min_population=None,
                 max_budget=None,
                 mutation_rate=0.05,
                 crossover_probability=0.7):
        """
        Args:
            grid_size (int): City grid dimension.
            population_size (int): μ — parent population size.
            generations (int): Number of generations.
            min_population (int): Hard lower bound on city population.
            max_budget (float): Hard upper bound on total cost.
            mutation_rate (float): Per-cell mutation probability.
            crossover_probability (float): Probability of applying crossover.
        """
        self.grid_size           = grid_size
        self.population_size     = population_size or GA_CONFIG['population_size']
        self.generations         = generations     or GA_CONFIG['generations']
        self.min_population      = min_population
        self.max_budget          = max_budget
        self.mutation_rate       = mutation_rate
        self.crossover_probability = crossover_probability

        self.population: list[Individual] = []
        self.pareto_front: list[Individual] = []

        # History for convergence plots
        self.history = {
            'generation':       [],
            'pareto_size':      [],
            'best_carbon':      [],
            'best_happiness':   [],
            'best_cost':        [],
            'viable_count':     [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_population(self, method='random_weighted', seed=None):
        """Create and evaluate initial population."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.population = []
        for i in range(self.population_size):
            city = CityGrid(size=self.grid_size)
            city.randomize(method=method, seed=seed + i if seed else None)
            ind = Individual(city)
            ind.evaluate(self.min_population, self.max_budget)
            self.population.append(ind)

        self._assign_ranks_and_distances(self.population)

    def run(self, verbose=True) -> tuple:
        """
        Execute NSGA-II for the configured number of generations.

        Returns:
            tuple: (pareto_front, history)
                   pareto_front — list of Individual on the final Pareto front
                   history      — dict with per-generation statistics
        """
        if not self.population:
            raise RuntimeError("Call initialize_population() before run().")

        if verbose:
            print("Starting NSGA-II Optimization")
            print(f"Grid: {self.grid_size}×{self.grid_size}  |  "
                  f"Population: {self.population_size}  |  "
                  f"Generations: {self.generations}")
            print("=" * 70)

        for gen in range(self.generations):
            # --- offspring generation ---
            offspring = self._generate_offspring()
            for ind in offspring:
                ind.evaluate(self.min_population, self.max_budget)

            # --- combine parent + offspring (size 2N) ---
            combined = self.population + offspring

            # --- non-dominated sort and selection ---
            fronts = non_dominated_sort(combined)
            for front in fronts:
                crowding_distance_assignment(front)

            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.population_size:
                    new_pop.extend(front)
                else:
                    # Fill remaining slots by crowding distance (desc)
                    remaining = self.population_size - len(new_pop)
                    sorted_front = sorted(front, key=lambda x: -x.crowding_distance)
                    new_pop.extend(sorted_front[:remaining])
                    break

            self.population = new_pop

            # --- track Pareto front ---
            self.pareto_front = fronts[0] if fronts else []

            # --- history ---
            self._update_history(gen)

            if verbose and gen % GA_CONFIG.get('log_frequency', 10) == 0:
                self._print_progress(gen)

        if verbose:
            print("\n" + "=" * 70)
            print("NSGA-II Optimization Complete!")
            self._print_final_summary()

        return self.pareto_front, self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_offspring(self) -> list:
        """Create N offspring via tournament selection + crossover + mutation."""
        offspring = []
        while len(offspring) < self.population_size:
            p1 = _binary_tournament(self.population)
            p2 = _binary_tournament(self.population)

            if random.random() < self.crossover_probability:
                c1, c2 = _block_crossover(p1, p2, self.grid_size)
            else:
                c1, c2 = p1.copy(), p2.copy()

            _point_mutation(c1, self.mutation_rate)
            _point_mutation(c2, self.mutation_rate)

            offspring.append(c1)
            if len(offspring) < self.population_size:
                offspring.append(c2)

        return offspring

    def _assign_ranks_and_distances(self, population: list):
        """Run non-dominated sort + crowding distance in-place."""
        fronts = non_dominated_sort(population)
        for front in fronts:
            crowding_distance_assignment(front)

    def _update_history(self, generation: int):
        viable = [ind for ind in self.pareto_front if ind.is_viable]
        self.history['generation'].append(generation)
        self.history['pareto_size'].append(len(self.pareto_front))
        self.history['viable_count'].append(len(viable))

        if viable:
            self.history['best_carbon'].append(min(v.carbon for v in viable))
            self.history['best_happiness'].append(max(v.happiness for v in viable))
            self.history['best_cost'].append(min(v.cost for v in viable))
        else:
            self.history['best_carbon'].append(float('nan'))
            self.history['best_happiness'].append(float('nan'))
            self.history['best_cost'].append(float('nan'))

    def _print_progress(self, generation: int):
        viable = [ind for ind in self.pareto_front if ind.is_viable]
        n_viable = len(viable)
        best_c = min((v.carbon for v in viable), default=float('nan'))
        best_h = max((v.happiness for v in viable), default=float('nan'))
        print(f"Gen {generation:4d} | "
              f"Pareto size: {len(self.pareto_front):3d} | "
              f"Viable: {n_viable:3d} | "
              f"Best carbon: {best_c:,.0f} | "
              f"Best happiness: {best_h:.1f}")

    def _print_final_summary(self):
        viable = [ind for ind in self.pareto_front if ind.is_viable]
        print(f"\nFinal Pareto Front: {len(self.pareto_front)} solutions "
              f"({len(viable)} feasible)")
        if viable:
            print(f"  Best carbon:    {min(v.carbon for v in viable):,.0f}")
            print(f"  Best happiness: {max(v.happiness for v in viable):.1f}")
            print(f"  Best cost:      ${min(v.cost for v in viable):,.0f}")


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_ga_vs_nsga2(ga_best_fitness, nsga2_pareto_front: list) -> str:
    """
    Produce a formatted comparison between the GA's best solution and the
    NSGA-II Pareto front.

    This is the recommended entry point for generating Table comparisons in
    the paper: the GA weighted-sum scalar vs the NSGA-II front illustrates
    whether the weight vector used in the GA aligns with the true Pareto front.

    Args:
        ga_best_fitness: FitnessResult from the GA run.
        nsga2_pareto_front: list[Individual] from NSGA2.run().

    Returns:
        str: Formatted comparison text.
    """
    ga_m = ga_best_fitness.metrics

    viable_front = [ind for ind in nsga2_pareto_front if ind.is_viable]

    report = "GA vs NSGA-II Comparison\n"
    report += "=" * 80 + "\n\n"

    report += "GA Best Solution (weighted-sum, w_C=1.0, w_H=0.5, w_$=0.3):\n"
    if ga_m:
        report += f"  Net Carbon:     {ga_m.net_carbon:>12,.0f}\n"
        report += f"  Happiness:      {ga_m.happiness_score:>12.1f}\n"
        report += f"  Cost:           ${ga_m.total_cost:>11,.0f}\n"
        report += f"  Viable:         {ga_best_fitness.is_viable}\n"
    else:
        report += "  (no metrics available)\n"

    report += f"\nNSGA-II Pareto Front: {len(nsga2_pareto_front)} solutions "
    report += f"({len(viable_front)} feasible)\n"

    if viable_front:
        # Best on each objective
        best_c = min(viable_front, key=lambda x: x.carbon)
        best_h = max(viable_front, key=lambda x: x.happiness)
        best_cost_ind = min(viable_front, key=lambda x: x.cost)

        report += "\n  Best-carbon solution:\n"
        report += f"    Carbon={best_c.carbon:,.0f}  Happiness={best_c.happiness:.1f}  Cost=${best_c.cost:,.0f}\n"
        report += "\n  Best-happiness solution:\n"
        report += f"    Carbon={best_h.carbon:,.0f}  Happiness={best_h.happiness:.1f}  Cost=${best_h.cost:,.0f}\n"
        report += "\n  Best-cost solution:\n"
        report += f"    Carbon={best_cost_ind.carbon:,.0f}  Happiness={best_cost_ind.happiness:.1f}  Cost=${best_cost_ind.cost:,.0f}\n"

        # Check if GA solution is dominated by any NSGA-II front member
        if ga_m:
            ga_obj = (ga_m.net_carbon, -ga_m.happiness_score, ga_m.total_cost)
            dominated_by = [
                ind for ind in viable_front
                if all(io <= go for io, go in zip(ind.objectives, ga_obj))
                and any(io < go for io, go in zip(ind.objectives, ga_obj))
            ]
            report += f"\nGA solution dominated by {len(dominated_by)} NSGA-II front member(s).\n"
            if dominated_by:
                report += "  (GA weighted-sum solution is Pareto-suboptimal)\n"
            else:
                report += "  (GA solution is Pareto non-dominated — weights were well-calibrated)\n"

    report += "\n" + "=" * 80 + "\n"
    return report


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing NSGA-II...\n")

    nsga2 = NSGA2(
        grid_size=20,
        population_size=20,
        generations=30,
        min_population=1000,
        max_budget=1_000_000,
    )

    print("Initialising population...")
    nsga2.initialize_population(seed=42)

    print("Running NSGA-II...\n")
    pareto, history = nsga2.run(verbose=True)

    print(f"\nPareto front size: {len(pareto)}")
    viable = [ind for ind in pareto if ind.is_viable]
    print(f"Feasible Pareto members: {len(viable)}")
    if viable:
        print(f"Best carbon: {min(v.carbon for v in viable):,.0f}")
        print(f"Best happiness: {max(v.happiness for v in viable):.1f}")
        print(f"Best cost: ${min(v.cost for v in viable):,.0f}")
