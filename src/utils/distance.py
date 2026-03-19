"""
Distance and Spatial Utility Functions

This module provides efficient distance calculations and spatial analysis
functions used by the constraint evaluation system.

All functions are hardware-agnostic (work with both NumPy and CuPy).
"""

from .array_backend import xp


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
        
    Returns:
        Distance between points (scalar or tensor)
    """
    x1, y1, x2, y2 = [float(v) if not hasattr(v, 'shape') else v for v in [x1, y1, x2, y2]]
    return xp.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def manhattan_distance(x1, y1, x2, y2):
    """
    Calculate Manhattan (city block) distance between two points.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
        
    Returns:
        Manhattan distance (scalar or tensor)
    """
    x1, y1, x2, y2 = [float(v) if not hasattr(v, 'shape') else v for v in [x1, y1, x2, y2]]
    return xp.abs(x2 - x1) + xp.abs(y2 - y1)


def create_distance_matrix(grid_size, metric='euclidean'):
    """
    Create a distance matrix for a grid.
    
    For each cell (i,j), computes distance to all other cells.
    Result is a 4D array: distance_matrix[i, j, k, l] = distance from (i,j) to (k,l)
    
    Args:
        grid_size (int): Size of the square grid
        metric (str): 'euclidean' or 'manhattan'
        
    Returns:
        4D array of distances
    """
    # Create coordinate grids
    coords = xp.arange(grid_size)
    i, j = xp.meshgrid(coords, coords, indexing='ij')
    
    # Flatten for broadcasting
    i_flat = i.flatten()[:, None, None]
    j_flat = j.flatten()[:, None, None]
    
    k = xp.arange(grid_size)[None, :, None]
    l = xp.arange(grid_size)[None, None, :]
    
    if metric == 'euclidean':
        distances = xp.sqrt((k - i_flat)**2 + (l - j_flat)**2)
    elif metric == 'manhattan':
        distances = xp.abs(k - i_flat) + xp.abs(l - j_flat)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Reshape to 4D
    return distances.reshape(grid_size, grid_size, grid_size, grid_size)


def find_neighbors(grid, building_type, distance_threshold, metric='euclidean'):
    """
    Find all cells of a specific building type within distance threshold.
    
    Args:
        grid: 2D array of building type IDs
        building_type (int): Building type to search for
        distance_threshold (float): Maximum distance
        metric (str): Distance metric
        
    Returns:
        Dictionary mapping (x,y) -> list of (neighbor_x, neighbor_y, distance)
    """
    grid_size = grid.shape[0]
    
    # Find all cells with the target building type
    target_coords = xp.argwhere(grid == building_type)
    
    neighbors = {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            nearby = []
            for target in target_coords:
                tx, ty = int(target[0]), int(target[1])
                
                if metric == 'euclidean':
                    dist = euclidean_distance(i, j, tx, ty)
                else:
                    dist = manhattan_distance(i, j, tx, ty)
                
                # Convert tensor to scalar if needed
                dist_val = float(dist.item()) if hasattr(dist, 'item') else float(dist)
                
                if dist_val <= distance_threshold and dist_val > 0:  # Exclude self
                    nearby.append((tx, ty, dist_val))
            
            if nearby:
                neighbors[(i, j)] = nearby
    
    return neighbors


def count_neighbors_in_radius(grid, center_x, center_y, building_types, radius, metric='euclidean'):
    """
    Count number of specific building types within radius of a center point.
    
    Args:
        grid: 2D array of building type IDs
        center_x, center_y: Center coordinates
        building_types: List of building type IDs to count
        radius (float): Search radius
        metric (str): Distance metric
        
    Returns:
        int: Count of matching buildings within radius
    """
    grid_size = grid.shape[0]
    count = 0
    
    # Optimize by only checking cells within bounding box
    min_x = max(0, int(center_x - radius - 1))
    max_x = min(grid_size, int(center_x + radius + 2))
    min_y = max(0, int(center_y - radius - 1))
    max_y = min(grid_size, int(center_y + radius + 2))
    
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            # Convert tensor element to int for comparison
            cell_value = int(grid[i, j].item()) if hasattr(grid[i, j], 'item') else int(grid[i, j])
            if cell_value in building_types:
                if metric == 'euclidean':
                    dist = euclidean_distance(center_x, center_y, i, j)
                else:
                    dist = manhattan_distance(center_x, center_y, i, j)
                
                # Convert dist to scalar if needed
                dist_val = float(dist.item()) if hasattr(dist, 'item') else float(dist)
                
                if dist_val <= radius and dist_val > 0:  # Exclude self
                    count += 1
    
    return count


def get_cells_by_type(grid, building_types):
    """
    Get coordinates of all cells containing specific building types.
    
    Args:
        grid: 2D array of building type IDs
        building_types: Single int or list of building type IDs
        
    Returns:
        List of (x, y) tuples
    """
    if isinstance(building_types, int):
        building_types = [building_types]
    
    mask = xp.isin(grid, building_types)
    coords = xp.argwhere(mask)
    
    return [(int(c[0]), int(c[1])) for c in coords]


def create_circular_mask(grid_size, center_x, center_y, radius):
    """
    Create a circular mask around a center point.
    
    Args:
        grid_size (int): Size of the grid
        center_x, center_y: Center coordinates
        radius (float): Radius of circle
        
    Returns:
        2D boolean array (True inside circle)
    """
    y, x = xp.ogrid[:grid_size, :grid_size]
    dist_from_center = xp.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = dist_from_center <= radius
    return mask


def calculate_pollution_overlay(grid, pollution_sources, pollution_radius, decay_factor=0.5):
    """
    Calculate pollution intensity overlay for visualization.
    
    Pollution decays with distance from source.
    
    Args:
        grid: 2D array of building type IDs
        pollution_sources: List of building types that emit pollution
        pollution_radius (float): Maximum pollution spread distance
        decay_factor (float): How quickly pollution decays (0-1)
        
    Returns:
        2D array of pollution intensity (0-1)
    """
    grid_size = grid.shape[0]
    pollution_map = xp.zeros((grid_size, grid_size), dtype=float)
    
    # Find all pollution sources
    source_coords = get_cells_by_type(grid, pollution_sources)
    
    for sx, sy in source_coords:
        # Create circular gradient
        y, x = xp.ogrid[:grid_size, :grid_size]
        dist = xp.sqrt((x - sy)**2 + (y - sx)**2)
        
        # Exponential decay
        intensity = xp.exp(-decay_factor * dist / pollution_radius)
        intensity = xp.where(dist <= pollution_radius, intensity, 0)
        
        # Add to pollution map (overlapping sources accumulate)
        pollution_map += intensity
    
    # Normalize to 0-1 range
    max_val = xp.max(pollution_map)
    # Handle scalar tensor conversion
    max_scalar = float(max_val.item()) if hasattr(max_val, 'item') else float(max_val)
    if max_scalar > 0:
        pollution_map = pollution_map / max_val
    
    return pollution_map


def calculate_grid_statistics(grid):
    """
    Calculate basic statistics about grid composition.
    
    Args:
        grid: 2D array of building type IDs
        
    Returns:
        Dictionary with building type counts and percentages
    """
    unique, counts = xp.unique(grid, return_counts=True)
    total_cells = grid.size
    
    stats = {}
    for building_type, count in zip(unique, counts):
        # Convert tensors to Python scalars
        bt_val = int(building_type.item()) if hasattr(building_type, 'item') else int(building_type)
        count_val = int(count.item()) if hasattr(count, 'item') else int(count)
        stats[bt_val] = {
            'count': count_val,
            'percentage': (count_val / total_cells) * 100
        }
    
    return stats


if __name__ == "__main__":
    # Test distance functions
    print("Testing distance functions...")
    
    # Test with small grid
    test_grid = xp.array([
        [0, 1, 0],
        [2, 0, 3],
        [0, 1, 0]
    ])
    
    print(f"\nTest grid:\n{test_grid}")
    
    # Test neighbor finding
    neighbors = find_neighbors(test_grid, building_type=1, distance_threshold=2.0)
    print(f"\nNeighbors of building type 1 within distance 2.0:")
    for pos, neighbor_list in neighbors.items():
        print(f"  {pos}: {len(neighbor_list)} neighbors")
    
    # Test pollution overlay
    pollution = calculate_pollution_overlay(test_grid, pollution_sources=[3], 
                                           pollution_radius=2.0)
    print(f"\nPollution overlay:\n{pollution}")
    
    # Test statistics
    stats = calculate_grid_statistics(test_grid)
    print(f"\nGrid statistics:")
    for btype, info in stats.items():
        print(f"  Type {btype}: {info['count']} cells ({info['percentage']:.1f}%)")
