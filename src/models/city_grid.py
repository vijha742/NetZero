"""
City Grid Model

This module defines the CityGrid class, which represents the spatial layout
of the city as a 2D matrix. Each cell contains a building type ID.

The grid is the core data structure that the genetic algorithm will optimize.
"""

import sys
sys.path.append('..')

from ..utils.array_backend import xp, to_numpy
from ..config.building_config import BUILDING_TYPES, NUM_BUILDING_TYPES
from ..config.optimization_config import INIT_CONFIG, GRID_CONFIG
import random


class CityGrid:
    """
    Represents a city as a 2D grid of building types.
    
    Attributes:
        size (int): Grid dimension (size x size)
        grid (array): 2D array of building type IDs
        cell_size_meters (int): Physical size of each cell in meters
    """
    
    def __init__(self, size=None, grid=None):
        """
        Initialize a city grid.
        
        Args:
            size (int): Grid dimension. If None, uses default from config.
            grid (array): Pre-existing grid to use. If None, creates empty grid.
        """
        if size is None:
            size = GRID_CONFIG['default_size']
        
        # Validate size
        if size < GRID_CONFIG['min_size'] or size > GRID_CONFIG['max_size']:
            raise ValueError(f"Grid size must be between {GRID_CONFIG['min_size']} "
                           f"and {GRID_CONFIG['max_size']}")
        
        self.size = size
        self.cell_size_meters = GRID_CONFIG['cell_size_meters']
        
        if grid is not None:
            if grid.shape != (size, size):
                raise ValueError(f"Grid shape {grid.shape} doesn't match size {size}x{size}")
            self.grid = xp.array(grid)
        else:
            # Initialize with empty land (building type 0)
            self.grid = xp.zeros((size, size), dtype=int)
    
    def __repr__(self):
        return f"CityGrid(size={self.size}x{self.size}, cells={self.size**2})"
    
    def copy(self):
        """Create a deep copy of this grid."""
        return CityGrid(size=self.size, grid=self.grid.copy())
    
    def get_building_at(self, x, y):
        """
        Get building type at specific coordinates.
        
        Args:
            x, y: Grid coordinates
            
        Returns:
            int: Building type ID
        """
        return int(self.grid[x, y])
    
    def set_building_at(self, x, y, building_type):
        """
        Set building type at specific coordinates.
        
        Args:
            x, y: Grid coordinates
            building_type (int): Building type ID to place
        """
        if building_type not in BUILDING_TYPES:
            raise ValueError(f"Invalid building type: {building_type}")
        self.grid[x, y] = building_type
    
    def randomize(self, method='random_weighted', seed=None):
        """
        Randomly populate the grid with buildings.
        
        Args:
            method (str): Initialization method
                - 'random': Uniform random selection
                - 'random_weighted': Weighted random based on probabilities
                - 'template': Load a predefined template
            seed (int): Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            xp.random.seed(seed)
        
        if method == 'random':
            # Uniform random selection from all building types
            self.grid = xp.random.randint(0, NUM_BUILDING_TYPES, 
                                         size=(self.size, self.size))
        
        elif method == 'random_weighted':
            # Weighted random selection based on probabilities
            probs = INIT_CONFIG['building_probabilities']
            building_types = list(probs.keys())
            probabilities = list(probs.values())
            
            # Normalize probabilities
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            
            # Generate random grid
            flat_grid = random.choices(building_types, weights=probabilities, 
                                      k=self.size * self.size)
            self.grid = xp.array(flat_grid).reshape(self.size, self.size)
        
        elif method == 'template':
            self._load_template(INIT_CONFIG['template_name'])
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        # Ensure basic viability if configured
        if INIT_CONFIG['ensure_basic_viability']:
            self._ensure_viability()
    
    def _load_template(self, template_name):
        """Load a predefined template (e.g., 'coal_heavy_city')."""
        if template_name == 'coal_heavy_city':
            # Create a deliberately inefficient city with lots of coal plants
            self.grid = xp.zeros((self.size, self.size), dtype=int)
            
            # Fill with high-density residential (50%)
            mask = xp.random.rand(self.size, self.size) < 0.5
            self.grid[mask] = 1  # Res_HD
            
            # Add many coal plants (20%)
            mask = xp.random.rand(self.size, self.size) < 0.2
            self.grid[mask] = 4  # Coal
            
            # Add factories (15%)
            mask = xp.random.rand(self.size, self.size) < 0.15
            self.grid[mask] = 3  # Factory
            
            # Add minimal forests (10%)
            mask = xp.random.rand(self.size, self.size) < 0.1
            self.grid[mask] = 6  # Forest
            
            # Add few roads (5%)
            mask = xp.random.rand(self.size, self.size) < 0.05
            self.grid[mask] = 7  # Road
        
        else:
            raise ValueError(f"Unknown template: {template_name}")
    
    def _ensure_viability(self):
        """
        Ensure the grid has minimum viable infrastructure.
        
        Checks:
        - At least some residential buildings
        - At least some energy production
        - At least some roads for connectivity
        """
        grid_numpy = to_numpy(self.grid)
        
        # Count key building types
        has_residential = xp.any(xp.isin(self.grid, [1, 2]))
        has_energy = xp.any(xp.isin(self.grid, [4, 5]))
        has_roads = xp.any(self.grid == 7)
        
        # Add missing critical infrastructure
        if not has_residential:
            # Place some residential randomly
            num_residential = max(5, self.size // 10)
            for _ in range(num_residential):
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                self.grid[x, y] = 1  # Res_HD
        
        if not has_energy:
            # Place at least one power plant
            num_plants = max(2, self.size // 20)
            for _ in range(num_plants):
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                self.grid[x, y] = random.choice([4, 5])  # Coal or Solar
        
        if not has_roads:
            # Place some roads
            num_roads = max(10, self.size // 5)
            for _ in range(num_roads):
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                self.grid[x, y] = 7  # Road
    
    def to_numpy(self):
        """Convert grid to NumPy array (for export/visualization)."""
        return to_numpy(self.grid)
    
    def get_building_counts(self):
        """
        Get count of each building type in the grid.
        
        Returns:
            dict: {building_type: count}
        """
        unique, counts = xp.unique(self.grid, return_counts=True)
        return {int(bt): int(c) for bt, c in zip(unique, counts)}
    
    def get_area_km2(self):
        """
        Calculate total area covered by the grid in square kilometers.
        
        Returns:
            float: Area in km²
        """
        total_meters_sq = (self.size * self.cell_size_meters) ** 2
        return total_meters_sq / 1e6  # Convert to km²
    
    def swap_buildings(self, x1, y1, x2, y2):
        """
        Swap two buildings in the grid.
        
        Args:
            x1, y1: First building coordinates
            x2, y2: Second building coordinates
        """
        temp = self.grid[x1, y1]
        self.grid[x1, y1] = self.grid[x2, y2]
        self.grid[x2, y2] = temp
    
    def mutate_cell(self, x, y, new_type=None):
        """
        Change a single cell to a different building type.
        
        Args:
            x, y: Cell coordinates
            new_type (int): New building type. If None, chooses random.
        """
        if new_type is None:
            # Choose random different type
            current_type = self.grid[x, y]
            available_types = [t for t in BUILDING_TYPES.keys() if t != current_type]
            new_type = random.choice(available_types)
        
        self.set_building_at(x, y, new_type)
    
    def get_subgrid(self, x_start, y_start, x_end, y_end):
        """
        Extract a rectangular subgrid.
        
        Args:
            x_start, y_start: Top-left corner
            x_end, y_end: Bottom-right corner
            
        Returns:
            2D array: Subgrid
        """
        return self.grid[x_start:x_end, y_start:y_end]
    
    def set_subgrid(self, x_start, y_start, subgrid):
        """
        Replace a rectangular region with a subgrid.
        
        Args:
            x_start, y_start: Top-left corner
            subgrid: 2D array to insert
        """
        x_end = x_start + subgrid.shape[0]
        y_end = y_start + subgrid.shape[1]
        self.grid[x_start:x_end, y_start:y_end] = subgrid
    
    def export_to_csv(self, filepath):
        """
        Export grid to CSV file.
        
        Args:
            filepath (str): Path to save CSV
        """
        import numpy as np
        grid_numpy = self.to_numpy()
        np.savetxt(filepath, grid_numpy, fmt='%d', delimiter=',')
    
    @classmethod
    def load_from_csv(cls, filepath):
        """
        Load grid from CSV file.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            CityGrid: Loaded city grid
        """
        import numpy as np
        grid_numpy = np.loadtxt(filepath, delimiter=',', dtype=int)
        size = grid_numpy.shape[0]
        return cls(size=size, grid=grid_numpy)
    
    def get_summary(self):
        """
        Get a human-readable summary of the grid.
        
        Returns:
            str: Summary text
        """
        counts = self.get_building_counts()
        total_cells = self.size ** 2
        
        summary = f"City Grid Summary\n"
        summary += f"{'='*50}\n"
        summary += f"Size: {self.size}x{self.size} ({total_cells} cells)\n"
        summary += f"Area: {self.get_area_km2():.2f} km²\n"
        summary += f"Cell Size: {self.cell_size_meters}m x {self.cell_size_meters}m\n\n"
        summary += f"Building Distribution:\n"
        
        for building_type in sorted(counts.keys()):
            count = counts[building_type]
            percentage = (count / total_cells) * 100
            name = BUILDING_TYPES[building_type]['name']
            summary += f"  [{building_type}] {name:25s}: {count:5d} ({percentage:5.1f}%)\n"
        
        return summary


if __name__ == "__main__":
    # Test CityGrid class
    print("Testing CityGrid class...\n")
    
    # Create a small grid
    city = CityGrid(size=10)
    print(city)
    print()
    
    # Randomize with weighted probabilities
    city.randomize(method='random_weighted', seed=42)
    print("After weighted randomization:")
    print(city.get_summary())
    
    # Test mutation
    print("\nTesting mutation...")
    old_type = city.get_building_at(5, 5)
    city.mutate_cell(5, 5)
    new_type = city.get_building_at(5, 5)
    print(f"Cell (5,5) changed from type {old_type} to {new_type}")
    
    # Test copy
    city2 = city.copy()
    print(f"\nCopied grid: {city2}")
    
    # Test CSV export
    print("\nExporting to CSV...")
    city.export_to_csv("/tmp/test_city.csv")
    print("Exported to /tmp/test_city.csv")
