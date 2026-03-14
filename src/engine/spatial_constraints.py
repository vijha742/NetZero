"""
Spatial Constraints Evaluator

This module evaluates all spatial constraints defined in constraints_config.py
and calculates their effects on happiness, carbon emissions, and fitness scores.

Each constraint function:
1. Examines the spatial arrangement of buildings
2. Calculates penalties/bonuses
3. Returns a result dictionary with effects

The evaluate_all_constraints() function orchestrates all constraint evaluations.
"""

import sys
sys.path.append('..')

from ..utils.array_backend import xp, to_numpy
from ..utils.distance import (count_neighbors_in_radius, get_cells_by_type,
                              euclidean_distance, manhattan_distance)
from ..config.constraints_config import CONSTRAINTS
from ..config.building_config import BUILDING_TYPES


def evaluate_all_constraints(city_grid):
    """
    Evaluate all enabled spatial constraints for a city grid.
    
    Args:
        city_grid (CityGrid): The city grid to evaluate
        
    Returns:
        dict: Results for each constraint
            {constraint_name: {
                'happiness_penalty': float (0-1),
                'carbon_modifier': float,
                'bonus': float,
                'violations': int,
                'score': float  # Fitness contribution
            }}
    """
    results = {}
    
    grid = city_grid.grid
    grid_size = city_grid.size
    
    # Evaluate each enabled constraint
    if CONSTRAINTS['nimby']['enabled']:
        results['nimby'] = evaluate_nimby(grid, grid_size)
    
    if CONSTRAINTS['heat_island']['enabled']:
        results['heat_island'] = evaluate_heat_island(grid, grid_size)
    
    if CONSTRAINTS['transit_access']['enabled']:
        results['transit_access'] = evaluate_transit_access(grid, grid_size)
    
    if CONSTRAINTS['industrial_cluster']['enabled']:
        results['industrial_cluster'] = evaluate_industrial_cluster(grid, grid_size)
    
    if CONSTRAINTS['green_space']['enabled']:
        results['green_space'] = evaluate_green_space(grid, grid_size)
    
    if CONSTRAINTS['energy_transmission']['enabled']:
        results['energy_transmission'] = evaluate_energy_transmission(grid, grid_size)
    
    if CONSTRAINTS['pollution_dispersion']['enabled']:
        results['pollution_dispersion'] = evaluate_pollution_dispersion(grid, grid_size)
    
    if CONSTRAINTS['zoning_compatibility']['enabled']:
        results['zoning_compatibility'] = evaluate_zoning_compatibility(grid, grid_size)
    
    return results


# ============================================================================
# Individual Constraint Evaluation Functions
# ============================================================================

def evaluate_nimby(grid, grid_size):
    """
    NIMBY (Not In My Backyard): Residential unhappiness near industrial.
    
    Checks if residential buildings are too close to factories.
    """
    config = CONSTRAINTS['nimby']
    
    res_types = config['residential_types']
    ind_types = config['industrial_types']
    threshold = config['distance_threshold']
    happiness_penalty_rate = config['happiness_penalty']
    penalty_score = config['penalty_score']
    
    # Find all residential and industrial cells
    residential_cells = get_cells_by_type(grid, res_types)
    industrial_cells = get_cells_by_type(grid, ind_types)
    
    violations = 0
    affected_residential = []
    
    # Check each residential cell
    for rx, ry in residential_cells:
        for ix, iy in industrial_cells:
            dist = euclidean_distance(rx, ry, ix, iy)
            if dist <= threshold:
                violations += 1
                affected_residential.append((rx, ry))
                break  # Count each residential only once
    
    # Calculate overall happiness penalty
    if len(residential_cells) > 0:
        affected_ratio = len(affected_residential) / len(residential_cells)
        happiness_penalty = happiness_penalty_rate * affected_ratio
    else:
        happiness_penalty = 0.0
    
    score = violations * penalty_score
    
    return {
        'happiness_penalty': happiness_penalty,
        'violations': violations,
        'score': score,
        'affected_cells': affected_residential,
    }


def evaluate_heat_island(grid, grid_size):
    """
    Heat Island Effect: Houses without forests have increased cooling carbon.
    """
    config = CONSTRAINTS['heat_island']
    
    res_types = config['residential_types']
    forest_types = config['forest_types']
    threshold = config['distance_threshold']
    carbon_increase_rate = config['carbon_increase']
    penalty_score = config['penalty_score']
    
    residential_cells = get_cells_by_type(grid, res_types)
    
    affected_count = 0
    affected_cells = []
    
    for rx, ry in residential_cells:
        # Check if there's a forest within threshold
        has_nearby_forest = count_neighbors_in_radius(
            grid, rx, ry, forest_types, threshold
        ) > 0
        
        if not has_nearby_forest:
            affected_count += 1
            affected_cells.append((rx, ry))
    
    # Calculate carbon increase
    # Each affected residential building gets carbon increase
    total_carbon_increase = 0
    for rx, ry in affected_cells:
        building_type = int(grid[rx, ry])
        base_carbon = BUILDING_TYPES[building_type]['carbon']
        total_carbon_increase += base_carbon * carbon_increase_rate
    
    score = affected_count * penalty_score
    
    return {
        'carbon_modifier': total_carbon_increase,
        'violations': affected_count,
        'score': score,
        'affected_cells': affected_cells,
    }


def evaluate_transit_access(grid, grid_size):
    """
    Transit Accessibility: Residential must have road access.
    """
    config = CONSTRAINTS['transit_access']
    
    res_types = config['residential_types']
    road_types = config['road_types']
    threshold = config['distance_threshold']
    happiness_penalty_rate = config['happiness_penalty']
    penalty_score = config['penalty_score']
    
    residential_cells = get_cells_by_type(grid, res_types)
    
    isolated_count = 0
    isolated_cells = []
    
    for rx, ry in residential_cells:
        has_road_access = count_neighbors_in_radius(
            grid, rx, ry, road_types, threshold
        ) > 0
        
        if not has_road_access:
            isolated_count += 1
            isolated_cells.append((rx, ry))
    
    # Calculate happiness penalty
    if len(residential_cells) > 0:
        isolated_ratio = isolated_count / len(residential_cells)
        happiness_penalty = happiness_penalty_rate * isolated_ratio
    else:
        happiness_penalty = 0.0
    
    score = isolated_count * penalty_score
    
    return {
        'happiness_penalty': happiness_penalty,
        'violations': isolated_count,
        'score': score,
        'affected_cells': isolated_cells,
    }


def evaluate_industrial_cluster(grid, grid_size):
    """
    Industrial Clustering: Factories near each other get efficiency bonus.
    """
    config = CONSTRAINTS['industrial_cluster']
    
    ind_types = config['industrial_types']
    threshold = config['distance_threshold']
    carbon_reduction_rate = config['carbon_reduction']
    bonus_score = config['bonus_score']
    max_bonus_per = config['max_bonus_per_factory']
    
    industrial_cells = get_cells_by_type(grid, ind_types)
    
    total_carbon_reduction = 0
    cluster_pairs = 0
    
    for i, (ix, iy) in enumerate(industrial_cells):
        # Count neighbors for this factory
        neighbor_count = count_neighbors_in_radius(
            grid, ix, iy, ind_types, threshold
        )
        
        # Cap neighbors counted
        neighbor_count = min(neighbor_count, max_bonus_per)
        
        if neighbor_count > 0:
            # Calculate carbon reduction for this factory
            building_type = int(grid[ix, iy])
            base_carbon = BUILDING_TYPES[building_type]['carbon']
            reduction = base_carbon * carbon_reduction_rate * neighbor_count
            total_carbon_reduction += reduction
            cluster_pairs += neighbor_count
    
    # Bonus score (positive contribution to fitness)
    bonus = (cluster_pairs // 2) * bonus_score  # Divide by 2 to avoid double-counting pairs
    
    return {
        'carbon_modifier': -total_carbon_reduction,  # Negative because it's a reduction
        'bonus': bonus,
        'score': bonus,  # Positive score
        'cluster_pairs': cluster_pairs,
    }


def evaluate_green_space(grid, grid_size):
    """
    Green Space Requirements: Adequate forest per residential area.
    """
    config = CONSTRAINTS['green_space']
    
    res_types = config['residential_types']
    forest_types = config['forest_types']
    required_ratio = config['ratio']  # 1 forest per 10 residential
    search_radius = config['search_radius']
    penalty_score = config['penalty_score']
    
    residential_cells = get_cells_by_type(grid, res_types)
    forest_cells = get_cells_by_type(grid, forest_types)
    
    # Calculate required forests
    num_residential = len(residential_cells)
    required_forests = num_residential * required_ratio
    
    # Count forests that are actually serving residential areas
    serving_forests = 0
    for fx, fy in forest_cells:
        nearby_residential = count_neighbors_in_radius(
            grid, fx, fy, res_types, search_radius
        )
        if nearby_residential > 0:
            serving_forests += 1
    
    # Calculate shortage
    shortage = max(0, required_forests - serving_forests)
    
    score = int(shortage) * penalty_score
    
    return {
        'violations': int(shortage),
        'score': score,
        'required_forests': required_forests,
        'serving_forests': serving_forests,
    }


def evaluate_energy_transmission(grid, grid_size):
    """
    Energy Transmission Loss: Power plants have limited range.
    
    Simplified model: Check if energy consumers are within range of producers.
    """
    config = CONSTRAINTS['energy_transmission']
    
    power_types = config['power_plant_types']
    ranges = config['distance_ranges']
    penalty_score = config['penalty_score']
    
    # For this simplified version, we'll just check if there are power plants
    # A more complex version would calculate actual transmission losses
    
    power_plants = get_cells_by_type(grid, power_types)
    
    if len(power_plants) == 0:
        # No power plants = severe penalty
        return {
            'violations': 1,
            'score': penalty_score,
        }
    
    # Check if power plants are reasonably distributed
    # Simple heuristic: divide grid into quadrants, each should have at least one plant
    # or be within range of one
    
    quadrants_covered = 0
    half_size = grid_size // 2
    
    quadrant_coords = [
        (half_size // 2, half_size // 2),  # Top-left
        (half_size // 2, half_size + half_size // 2),  # Top-right
        (half_size + half_size // 2, half_size // 2),  # Bottom-left
        (half_size + half_size // 2, half_size + half_size // 2),  # Bottom-right
    ]
    
    for qx, qy in quadrant_coords:
        has_power = count_neighbors_in_radius(
            grid, qx, qy, power_types, ranges['full']
        ) > 0
        if has_power:
            quadrants_covered += 1
    
    # Penalty if not all quadrants covered
    uncovered = 4 - quadrants_covered
    score = uncovered * (penalty_score // 4)
    
    return {
        'violations': uncovered,
        'score': score,
        'quadrants_covered': quadrants_covered,
    }


def evaluate_pollution_dispersion(grid, grid_size):
    """
    Pollution Dispersion: Factories create pollution zones affecting residential.
    """
    config = CONSTRAINTS['pollution_dispersion']
    
    pollution_sources = config['pollution_sources']
    res_types = config['residential_types']
    radius = config['pollution_radius']
    happiness_penalty_rate = config['happiness_penalty']
    penalty_score = config['penalty_score']
    
    source_cells = get_cells_by_type(grid, pollution_sources)
    residential_cells = get_cells_by_type(grid, res_types)
    
    affected_count = 0
    affected_cells = []
    
    for rx, ry in residential_cells:
        # Check if within pollution radius of any source
        for sx, sy in source_cells:
            dist = euclidean_distance(rx, ry, sx, sy)
            if dist <= radius:
                affected_count += 1
                affected_cells.append((rx, ry))
                break
    
    # Calculate happiness penalty
    if len(residential_cells) > 0:
        affected_ratio = affected_count / len(residential_cells)
        happiness_penalty = happiness_penalty_rate * affected_ratio
    else:
        happiness_penalty = 0.0
    
    score = affected_count * penalty_score
    
    return {
        'happiness_penalty': happiness_penalty,
        'violations': affected_count,
        'score': score,
        'affected_cells': affected_cells,
    }


def evaluate_zoning_compatibility(grid, grid_size):
    """
    Zoning Compatibility: Certain building combinations prohibited.
    """
    config = CONSTRAINTS['zoning_compatibility']
    
    incompatible_pairs = config['incompatible_pairs']
    penalty_score = config['penalty_score']
    
    violations = 0
    violation_details = []
    
    for type1, type2, min_dist in incompatible_pairs:
        # Find all cells of each type
        cells1 = get_cells_by_type(grid, type1)
        cells2 = get_cells_by_type(grid, type2)
        
        # Check all pairs
        for x1, y1 in cells1:
            for x2, y2 in cells2:
                dist = euclidean_distance(x1, y1, x2, y2)
                if dist < min_dist:
                    violations += 1
                    violation_details.append({
                        'type1': type1,
                        'type2': type2,
                        'pos1': (x1, y1),
                        'pos2': (x2, y2),
                        'distance': dist,
                        'required': min_dist,
                    })
    
    score = violations * penalty_score
    
    return {
        'violations': violations,
        'score': score,
        'details': violation_details,
    }


# ============================================================================
# Summary Functions
# ============================================================================

def get_constraint_summary(results):
    """
    Create a human-readable summary of constraint evaluation results.
    
    Args:
        results (dict): Results from evaluate_all_constraints()
        
    Returns:
        str: Formatted summary
    """
    summary = "Spatial Constraints Evaluation\n"
    summary += "=" * 70 + "\n\n"
    
    total_score = 0
    total_violations = 0
    total_bonuses = 0
    
    for constraint_name, result in results.items():
        config = CONSTRAINTS[constraint_name]
        summary += f"{config['name']}\n"
        summary += f"  Status: {'✓ PASS' if result['violations'] == 0 else '⚠ VIOLATIONS'}\n"
        
        if 'violations' in result:
            violations = result['violations']
            summary += f"  Violations: {violations}\n"
            total_violations += violations
        
        if 'score' in result:
            score = result['score']
            summary += f"  Score: {score:+.0f}\n"
            total_score += score
        
        if 'bonus' in result:
            bonus = result['bonus']
            summary += f"  Bonus: +{bonus:.0f}\n"
            total_bonuses += bonus
        
        if 'happiness_penalty' in result:
            penalty = result['happiness_penalty']
            summary += f"  Happiness Penalty: -{penalty*100:.1f}%\n"
        
        if 'carbon_modifier' in result:
            modifier = result['carbon_modifier']
            summary += f"  Carbon Modifier: {modifier:+.0f} units\n"
        
        summary += "\n"
    
    summary += "=" * 70 + "\n"
    summary += f"TOTALS\n"
    summary += f"  Total Violations: {total_violations}\n"
    summary += f"  Total Penalty Score: {total_score:.0f}\n"
    summary += f"  Total Bonuses: +{total_bonuses:.0f}\n"
    summary += "=" * 70 + "\n"
    
    return summary


if __name__ == "__main__":
    # Test constraint evaluation
    from ..models.city_grid import CityGrid
    
    print("Testing Spatial Constraints Evaluator...\n")
    
    # Create a test city
    city = CityGrid(size=30)
    city.randomize(method='random_weighted', seed=42)
    
    print("Test City Created")
    print(f"Size: {city.size}x{city.size}\n")
    
    # Evaluate all constraints
    print("Evaluating spatial constraints...")
    results = evaluate_all_constraints(city)
    
    # Print summary
    print(get_constraint_summary(results))
