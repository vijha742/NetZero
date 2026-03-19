"""
Simulation Engine - City Metrics Calculation

This module contains the core "physics" of the simulation. The calculate_metrics()
function computes all relevant metrics for a given city grid:
- Population and jobs
- Energy production and consumption
- Carbon emissions and absorption
- Cost
- Happiness (affected by spatial constraints)

This is the heart of the fitness evaluation system.
"""

import sys
sys.path.append('..')

from ..utils.array_backend import xp, to_numpy
from ..config.building_config import BUILDING_TYPES
from ..utils.distance import get_cells_by_type


class CityMetrics:
    """Container for city simulation metrics."""
    
    def __init__(self):
        # Population and employment
        self.population = 0
        self.jobs = 0
        
        # Energy
        self.energy_demand = 0
        self.energy_supply = 0
        self.energy_balance = 0  # supply - demand
        
        # Carbon
        self.total_emissions = 0    # Sum of all positive carbon
        self.total_absorption = 0   # Sum of all negative carbon (absolute value)
        self.net_carbon = 0         # total_emissions + absorption (can be negative)
        self.carbon_ratio = 0       # |net_carbon| / total_emissions (target: <0.05)
        
        # Economics
        self.total_cost = 0
        
        # Quality of life
        self.base_happiness = 100.0  # Perfect happiness before constraints
        self.happiness_score = 100.0 # After applying constraint penalties
        self.happiness_penalties = {}  # Track which constraints reduced happiness
        
        # Spatial constraint violations
        self.constraint_violations = {}
        self.constraint_bonuses = {}
    
    def __repr__(self):
        return (f"CityMetrics(pop={self.population}, carbon={self.net_carbon:.1f}, "
                f"energy_bal={self.energy_balance:.1f}, cost={self.total_cost:.1f}, "
                f"happiness={self.happiness_score:.1f})")
    
    def to_dict(self):
        """Convert metrics to dictionary for easy serialization."""
        return {
            'population': self.population,
            'jobs': self.jobs,
            'energy_demand': self.energy_demand,
            'energy_supply': self.energy_supply,
            'energy_balance': self.energy_balance,
            'total_emissions': self.total_emissions,
            'total_absorption': self.total_absorption,
            'net_carbon': self.net_carbon,
            'carbon_ratio': self.carbon_ratio,
            'total_cost': self.total_cost,
            'base_happiness': self.base_happiness,
            'happiness_score': self.happiness_score,
            'happiness_penalties': self.happiness_penalties,
            'constraint_violations': self.constraint_violations,
            'constraint_bonuses': self.constraint_bonuses,
        }
    
    def is_viable(self, min_population=None, max_budget=None, energy_balance_required=True):
        """
        Check if city meets hard constraints (viability).
        
        Args:
            min_population (int): Minimum required population
            max_budget (float): Maximum allowed budget
            energy_balance_required (bool): Whether energy supply must >= demand
            
        Returns:
            bool: True if city is viable
        """
        if min_population is not None and self.population < min_population:
            return False
        
        if max_budget is not None and self.total_cost > max_budget:
            return False
        
        if energy_balance_required and self.energy_balance < 0:
            return False
        
        return True
    
    def success_achieved(self, target_carbon_ratio=0.05):
        """
        Check if city achieves the success criterion (<5% net carbon).
        
        Args:
            target_carbon_ratio (float): Maximum allowed carbon ratio
            
        Returns:
            bool: True if success criterion met
        """
        return self.carbon_ratio < target_carbon_ratio


def calculate_metrics(city_grid, spatial_constraints_results=None):
    """
    Calculate all metrics for a city grid.
    
    This is the core simulation function that computes:
    1. Basic metrics (population, energy, carbon, cost) by summing building attributes
    2. Spatial constraint effects (happiness penalties, carbon modifiers)
    
    Args:
        city_grid (CityGrid): The city grid to evaluate
        spatial_constraints_results (dict): Pre-computed spatial constraint results
                                           If None, will compute them here
        
    Returns:
        CityMetrics: Complete metrics for the city
    """
    metrics = CityMetrics()
    
    # Get the grid array
    grid = city_grid.grid
    grid_size = city_grid.size
    
    # ========================================================================
    # STEP 1: Calculate base metrics by iterating through all cells
    # ========================================================================
    
    for building_type, props in BUILDING_TYPES.items():
        # Count how many cells have this building type
        count = int(xp.sum(grid == building_type))
        
        if count == 0:
            continue
        
        # Accumulate metrics
        metrics.population += count * props['pop']
        metrics.jobs += count * props['jobs']
        metrics.energy_demand += count * props['energy_demand']
        metrics.energy_supply += count * props['energy_supply']
        metrics.total_cost += count * props['cost']
        
        # Carbon accounting
        carbon = props['carbon']
        if carbon > 0:
            # Emitter
            metrics.total_emissions += count * carbon
        else:
            # Absorber (forest, etc.)
            metrics.total_absorption += count * abs(carbon)
    
    # Calculate derived metrics
    metrics.energy_balance = metrics.energy_supply - metrics.energy_demand
    metrics.net_carbon = metrics.total_emissions - metrics.total_absorption
    
    # Calculate carbon ratio (avoid division by zero)
    if metrics.total_emissions > 0:
        metrics.carbon_ratio = abs(metrics.net_carbon) / metrics.total_emissions
    else:
        metrics.carbon_ratio = 0.0
    
    # ========================================================================
    # STEP 2: Apply spatial constraint effects
    # ========================================================================
    
    if spatial_constraints_results is not None:
        # Use pre-computed results (more efficient)
        _apply_constraint_effects(metrics, spatial_constraints_results)
    else:
        # Will be computed and applied by the constraint evaluator
        # For now, happiness remains at base 100.0
        pass
    
    return metrics


def _apply_constraint_effects(metrics, constraint_results):
    """
    Apply spatial constraint effects to metrics.
    
    Constraints can affect:
    - Happiness (NIMBY, pollution, transit, etc.)
    - Carbon (heat island effect, industrial clustering)
    
    Args:
        metrics (CityMetrics): Metrics object to modify
        constraint_results (dict): Results from spatial constraint evaluation
    """
    # Start with base happiness
    happiness_multiplier = 1.0
    
    # Track individual constraint contributions
    for constraint_name, result in constraint_results.items():
        if 'happiness_penalty' in result:
            penalty = result['happiness_penalty']
            happiness_multiplier *= (1.0 - penalty)
            metrics.happiness_penalties[constraint_name] = penalty
        
        if 'carbon_modifier' in result:
            # Some constraints modify carbon (e.g., heat island increases carbon)
            carbon_change = result['carbon_modifier']
            if carbon_change > 0:
                metrics.total_emissions += carbon_change
            else:
                metrics.total_absorption += abs(carbon_change)
            
            # Recalculate net carbon
            metrics.net_carbon = metrics.total_emissions - metrics.total_absorption
            if metrics.total_emissions > 0:
                metrics.carbon_ratio = abs(metrics.net_carbon) / metrics.total_emissions
        
        if 'bonus' in result:
            metrics.constraint_bonuses[constraint_name] = result['bonus']
        
        if 'violations' in result:
            metrics.constraint_violations[constraint_name] = result['violations']
    
    # Apply happiness multiplier
    metrics.happiness_score = metrics.base_happiness * happiness_multiplier
    
    # Cap happiness between 0 and 100
    metrics.happiness_score = max(0.0, min(100.0, metrics.happiness_score))


def calculate_total_emissions_without_offsets(city_grid):
    """
    Calculate total carbon emissions without counting carbon sinks.
    
    This is used for the carbon ratio calculation denominator.
    
    Args:
        city_grid (CityGrid): The city grid
        
    Returns:
        float: Total emissions from all emitters
    """
    grid = city_grid.grid
    total = 0.0
    
    for building_type, props in BUILDING_TYPES.items():
        if props['carbon'] > 0:
            count_tensor = xp.sum(grid == building_type)
            count = int(count_tensor.item()) if hasattr(count_tensor, 'item') else int(count_tensor)
            total += count * props['carbon']
    
    return total


def calculate_metrics_summary(city_grid, spatial_constraints_results=None):
    """
    Calculate metrics and return a formatted summary string.
    
    Args:
        city_grid (CityGrid): The city grid to evaluate
        spatial_constraints_results (dict): Pre-computed constraint results
        
    Returns:
        str: Formatted summary
    """
    metrics = calculate_metrics(city_grid, spatial_constraints_results)
    
    summary = "City Simulation Metrics\n"
    summary += "=" * 60 + "\n\n"
    
    summary += "POPULATION & EMPLOYMENT\n"
    summary += f"  Population:       {metrics.population:,}\n"
    summary += f"  Jobs:             {metrics.jobs:,}\n"
    summary += f"  Jobs/Population:  {metrics.jobs/max(metrics.population,1):.2f}\n\n"
    
    summary += "ENERGY\n"
    summary += f"  Demand:           {metrics.energy_demand:,.0f} units\n"
    summary += f"  Supply:           {metrics.energy_supply:,.0f} units\n"
    summary += f"  Balance:          {metrics.energy_balance:,.0f} units"
    if metrics.energy_balance < 0:
        summary += " ⚠️ DEFICIT"
    else:
        summary += " ✓"
    summary += "\n\n"
    
    summary += "CARBON EMISSIONS\n"
    summary += f"  Total Emissions:  {metrics.total_emissions:,.0f} units\n"
    summary += f"  Total Absorption: {metrics.total_absorption:,.0f} units\n"
    summary += f"  Net Carbon:       {metrics.net_carbon:,.0f} units\n"
    summary += f"  Carbon Ratio:     {metrics.carbon_ratio*100:.2f}%"
    if metrics.carbon_ratio < 0.05:
        summary += " ✓ SUCCESS (<5%)"
    else:
        summary += f" (Target: <5%)"
    summary += "\n\n"
    
    summary += "ECONOMICS\n"
    summary += f"  Total Cost:       ${metrics.total_cost:,.0f}\n"
    summary += f"  Cost per Capita:  ${metrics.total_cost/max(metrics.population,1):,.0f}\n\n"
    
    summary += "QUALITY OF LIFE\n"
    summary += f"  Happiness Score:  {metrics.happiness_score:.1f}/100.0\n"
    
    if metrics.happiness_penalties:
        summary += "  Happiness Penalties:\n"
        for constraint, penalty in metrics.happiness_penalties.items():
            summary += f"    - {constraint}: -{penalty*100:.1f}%\n"
    
    if metrics.constraint_bonuses:
        summary += "  Bonuses:\n"
        for constraint, bonus in metrics.constraint_bonuses.items():
            summary += f"    + {constraint}: +{bonus}\n"
    
    summary += "\n" + "=" * 60 + "\n"
    
    return summary


if __name__ == "__main__":
    # Test simulation engine
    from ..models.city_grid import CityGrid
    
    print("Testing Simulation Engine...\n")
    
    # Create a test city
    city = CityGrid(size=20)
    city.randomize(method='random_weighted', seed=42)
    
    print("Test City:")
    print(city.get_summary())
    print("\n")
    
    # Calculate metrics
    metrics = calculate_metrics(city)
    
    print("Calculated Metrics:")
    print(metrics)
    print("\n")
    
    # Get detailed summary
    print(calculate_metrics_summary(city))
    
    # Test viability checks
    print("\nViability Checks:")
    print(f"  Viable (no constraints): {metrics.is_viable()}")
    print(f"  Viable (min 1000 pop):   {metrics.is_viable(min_population=1000)}")
    print(f"  Success achieved (<5%):  {metrics.success_achieved()}")
