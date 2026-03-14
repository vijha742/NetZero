"""
Fitness Function for Multi-Objective Optimization

This module defines the fitness function that the Genetic Algorithm will optimize.

The fitness function combines multiple objectives:
1. Minimize carbon emissions (primary goal)
2. Maximize happiness (livability)
3. Minimize cost (economic feasibility)

Fitness = W1 × (-NetCarbon) + W2 × Happiness + W3 × (-Cost) + ConstraintScores - HardPenalties

Where:
- W1, W2, W3 are weights (configurable)
- Hard penalties applied for violations of hard constraints (population, energy, budget)
- Constraint scores include both penalties and bonuses from spatial constraints
"""

import sys
sys.path.append('..')

from ..engine.simulation import calculate_metrics
from ..engine.spatial_constraints import evaluate_all_constraints
from ..config.optimization_config import FITNESS_WEIGHTS, HARD_CONSTRAINTS


class FitnessResult:
    """Container for fitness evaluation result."""
    
    def __init__(self):
        self.fitness = 0.0
        self.carbon_component = 0.0
        self.happiness_component = 0.0
        self.cost_component = 0.0
        self.constraint_component = 0.0
        self.hard_penalty = 0.0
        
        self.metrics = None
        self.constraint_results = None
        
        self.is_viable = False
        self.success_achieved = False
    
    def __repr__(self):
        return f"FitnessResult(fitness={self.fitness:.2f}, viable={self.is_viable}, success={self.success_achieved})"
    
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
            'fitness': self.fitness,
            'carbon_component': self.carbon_component,
            'happiness_component': self.happiness_component,
            'cost_component': self.cost_component,
            'constraint_component': self.constraint_component,
            'hard_penalty': self.hard_penalty,
            'is_viable': self.is_viable,
            'success_achieved': self.success_achieved,
        }


def calculate_fitness(city_grid, 
                     weights=None,
                     min_population=None, 
                     max_budget=None,
                     energy_balance_required=True):
    """
    Calculate fitness score for a city grid.
    
    This is the core function that the GA will optimize.
    
    Args:
        city_grid (CityGrid): The city layout to evaluate
        weights (dict): Custom weights {carbon, happiness, cost}. If None, uses defaults.
        min_population (int): Minimum required population (hard constraint)
        max_budget (float): Maximum allowed budget (hard constraint)
        energy_balance_required (bool): Whether energy must balance
        
    Returns:
        FitnessResult: Complete fitness evaluation
    """
    result = FitnessResult()
    
    # Use default weights if not provided
    if weights is None:
        weights = FITNESS_WEIGHTS
    
    # Set hard constraints if not provided
    if min_population is None:
        min_population = HARD_CONSTRAINTS['min_population']
    if max_budget is None:
        max_budget = HARD_CONSTRAINTS['max_budget']
    if energy_balance_required is None:
        energy_balance_required = HARD_CONSTRAINTS['energy_balance']
    
    # Step 1: Evaluate spatial constraints
    constraint_results = evaluate_all_constraints(city_grid)
    result.constraint_results = constraint_results
    
    # Step 2: Calculate city metrics (including constraint effects)
    metrics = calculate_metrics(city_grid, constraint_results)
    result.metrics = metrics
    
    # Step 3: Check viability (hard constraints)
    result.is_viable = metrics.is_viable(
        min_population=min_population,
        max_budget=max_budget,
        energy_balance_required=energy_balance_required
    )
    
    # Step 4: Check success criterion
    result.success_achieved = metrics.success_achieved(
        target_carbon_ratio=weights['target_carbon_ratio']
    )
    
    # Step 5: Calculate fitness components
    
    # Carbon component: Minimize net carbon (negative = better)
    # We want to maximize (-carbon), so lower carbon gives higher fitness
    result.carbon_component = weights['carbon'] * (-metrics.net_carbon)
    
    # Happiness component: Maximize happiness (0-100)
    result.happiness_component = weights['happiness'] * metrics.happiness_score
    
    # Cost component: Minimize cost (negative = better)
    result.cost_component = weights['cost'] * (-metrics.total_cost)
    
    # Constraint component: Sum of all constraint scores (penalties and bonuses)
    constraint_score = sum(
        cr.get('score', 0) for cr in constraint_results.values()
    )
    result.constraint_component = weights['soft_constraints'] * constraint_score
    
    # Step 6: Apply hard constraint penalties
    if not result.is_viable:
        result.hard_penalty = -weights['hard_constraints']
        
        # Detailed penalties for debugging
        if min_population is not None and metrics.population < min_population:
            population_deficit = min_population - metrics.population
            result.hard_penalty -= population_deficit * 10  # Extra penalty proportional to deficit
        
        if max_budget is not None and metrics.total_cost > max_budget:
            budget_excess = metrics.total_cost - max_budget
            result.hard_penalty -= budget_excess * 0.1
        
        if energy_balance_required and metrics.energy_balance < 0:
            energy_deficit = abs(metrics.energy_balance)
            result.hard_penalty -= energy_deficit * 50
    
    # Step 7: Calculate total fitness
    result.fitness = (
        result.carbon_component +
        result.happiness_component +
        result.cost_component +
        result.constraint_component +
        result.hard_penalty
    )
    
    return result


def evaluate_population_fitness(population, **kwargs):
    """
    Evaluate fitness for a population of city grids.
    
    Args:
        population (list): List of CityGrid objects
        **kwargs: Arguments passed to calculate_fitness()
        
    Returns:
        list: List of FitnessResult objects (same order as population)
    """
    fitness_results = []
    
    for city_grid in population:
        fitness = calculate_fitness(city_grid, **kwargs)
        fitness_results.append(fitness)
    
    return fitness_results


def get_best_solution(population, fitness_results):
    """
    Get the best solution from a population.
    
    Args:
        population (list): List of CityGrid objects
        fitness_results (list): Corresponding fitness results
        
    Returns:
        tuple: (best_city_grid, best_fitness_result)
    """
    if not population:
        return None, None
    
    best_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i].fitness)
    return population[best_idx], fitness_results[best_idx]


def fitness_summary(fitness_result):
    """
    Create a human-readable summary of fitness evaluation.
    
    Args:
        fitness_result (FitnessResult): Fitness result to summarize
        
    Returns:
        str: Formatted summary
    """
    summary = "Fitness Evaluation Summary\n"
    summary += "=" * 70 + "\n\n"
    
    summary += f"TOTAL FITNESS: {fitness_result.fitness:.2f}\n"
    summary += f"Viable: {'✓ YES' if fitness_result.is_viable else '✗ NO'}\n"
    summary += f"Success (<5% carbon): {'✓ YES' if fitness_result.success_achieved else '✗ NO'}\n\n"
    
    summary += "FITNESS COMPONENTS:\n"
    summary += f"  Carbon Component:     {fitness_result.carbon_component:+10.2f}\n"
    summary += f"  Happiness Component:  {fitness_result.happiness_component:+10.2f}\n"
    summary += f"  Cost Component:       {fitness_result.cost_component:+10.2f}\n"
    summary += f"  Constraint Component: {fitness_result.constraint_component:+10.2f}\n"
    
    if fitness_result.hard_penalty != 0:
        summary += f"  Hard Penalty:         {fitness_result.hard_penalty:+10.2f} ⚠️\n"
    
    if fitness_result.metrics:
        m = fitness_result.metrics
        summary += "\n"
        summary += "CITY METRICS:\n"
        summary += f"  Population:     {m.population:,}\n"
        summary += f"  Net Carbon:     {m.net_carbon:,.0f} ({m.carbon_ratio*100:.2f}%)\n"
        summary += f"  Energy Balance: {m.energy_balance:+,.0f}\n"
        summary += f"  Cost:           ${m.total_cost:,.0f}\n"
        summary += f"  Happiness:      {m.happiness_score:.1f}/100\n"
    
    summary += "\n" + "=" * 70 + "\n"
    
    return summary


def compare_fitness(fitness_a, fitness_b, labels=None):
    """
    Compare two fitness results side-by-side.
    
    Args:
        fitness_a (FitnessResult): First fitness result
        fitness_b (FitnessResult): Second fitness result
        labels (tuple): Optional labels for A and B (default: "A", "B")
        
    Returns:
        str: Formatted comparison
    """
    if labels is None:
        labels = ("A", "B")
    
    label_a, label_b = labels
    
    comparison = "Fitness Comparison\n"
    comparison += "=" * 80 + "\n\n"
    
    comparison += f"{'Metric':<25s} {label_a:>15s} {label_b:>15s} {'Difference':>15s}\n"
    comparison += "-" * 80 + "\n"
    
    # Total fitness
    diff_fitness = fitness_b.fitness - fitness_a.fitness
    comparison += f"{'Total Fitness':<25s} {fitness_a.fitness:>15.2f} {fitness_b.fitness:>15.2f} {diff_fitness:>+15.2f}\n"
    
    # Components
    diff_carbon = fitness_b.carbon_component - fitness_a.carbon_component
    comparison += f"{'  Carbon Component':<25s} {fitness_a.carbon_component:>15.2f} {fitness_b.carbon_component:>15.2f} {diff_carbon:>+15.2f}\n"
    
    diff_happiness = fitness_b.happiness_component - fitness_a.happiness_component
    comparison += f"{'  Happiness Component':<25s} {fitness_a.happiness_component:>15.2f} {fitness_b.happiness_component:>15.2f} {diff_happiness:>+15.2f}\n"
    
    diff_cost = fitness_b.cost_component - fitness_a.cost_component
    comparison += f"{'  Cost Component':<25s} {fitness_a.cost_component:>15.2f} {fitness_b.cost_component:>15.2f} {diff_cost:>+15.2f}\n"
    
    comparison += "\n"
    
    # Metrics
    if fitness_a.metrics and fitness_b.metrics:
        ma, mb = fitness_a.metrics, fitness_b.metrics
        
        diff_pop = mb.population - ma.population
        comparison += f"{'Population':<25s} {ma.population:>15,} {mb.population:>15,} {diff_pop:>+15,}\n"
        
        diff_carbon = mb.net_carbon - ma.net_carbon
        comparison += f"{'Net Carbon':<25s} {ma.net_carbon:>15,.0f} {mb.net_carbon:>15,.0f} {diff_carbon:>+15,.0f}\n"
        
        comparison += f"{'Carbon Ratio':<25s} {ma.carbon_ratio*100:>14.2f}% {mb.carbon_ratio*100:>14.2f}%\n"
        
        diff_energy = mb.energy_balance - ma.energy_balance
        comparison += f"{'Energy Balance':<25s} {ma.energy_balance:>15,.0f} {mb.energy_balance:>15,.0f} {diff_energy:>+15,.0f}\n"
        
        diff_happiness = mb.happiness_score - ma.happiness_score
        comparison += f"{'Happiness':<25s} {ma.happiness_score:>15.1f} {mb.happiness_score:>15.1f} {diff_happiness:>+15.1f}\n"
    
    comparison += "\n" + "=" * 80 + "\n"
    
    # Winner
    if fitness_b.fitness > fitness_a.fitness:
        comparison += f"Winner: {label_b} (by {diff_fitness:.2f} points)\n"
    elif fitness_a.fitness > fitness_b.fitness:
        comparison += f"Winner: {label_a} (by {-diff_fitness:.2f} points)\n"
    else:
        comparison += "Tie\n"
    
    return comparison


if __name__ == "__main__":
    # Test fitness function
    from ..models.city_grid import CityGrid
    
    print("Testing Fitness Function...\n")
    
    # Create two test cities
    city1 = CityGrid(size=20)
    city1.randomize(method='random_weighted', seed=42)
    
    city2 = CityGrid(size=20)
    city2.randomize(method='random_weighted', seed=123)
    
    print("Evaluating City 1...")
    fitness1 = calculate_fitness(city1, min_population=1000, max_budget=1000000)
    print(fitness_summary(fitness1))
    
    print("\nEvaluating City 2...")
    fitness2 = calculate_fitness(city2, min_population=1000, max_budget=1000000)
    print(fitness_summary(fitness2))
    
    print("\nComparing Cities...")
    print(compare_fitness(fitness1, fitness2, labels=("City 1", "City 2")))
