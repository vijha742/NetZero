"""
Genetic Algorithm for City Optimization

This module implements the Genetic Algorithm that optimizes city layouts
to achieve net-zero carbon emissions while maintaining livability and cost-effectiveness.

GA Components:
1. Initialization: Create random population of cities
2. Selection: Choose parents based on fitness (tournament/roulette/rank)
3. Crossover: Combine two parent cities to create offspring
4. Mutation: Random modifications to maintain diversity
5. Elitism: Preserve best solutions across generations
6. Convergence: Track progress and stop when optimized

The algorithm iteratively improves the population over many generations.
"""

import sys
sys.path.append('..')

import random
import numpy as np
from copy import deepcopy

from ..models.city_grid import CityGrid
from ..optimization.fitness import calculate_fitness, evaluate_population_fitness, get_best_solution
from ..config.optimization_config import GA_CONFIG, FITNESS_WEIGHTS, HARD_CONSTRAINTS
from ..utils.array_backend import xp, to_numpy


class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer for zero-carbon city planning.
    """
    
    def __init__(self, 
                 grid_size=50,
                 population_size=None,
                 generations=None,
                 min_population=None,
                 max_budget=None,
                 fitness_weights=None,
                 config=None):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            grid_size (int): Size of city grid
            population_size (int): Number of cities in population
            generations (int): Maximum generations to run
            min_population (int): Minimum city population (hard constraint)
            max_budget (float): Maximum city budget (hard constraint)
            fitness_weights (dict): Custom fitness weights
            config (dict): Custom GA configuration
        """
        self.grid_size = grid_size
        
        # Use defaults from config if not provided
        self.config = GA_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.population_size = population_size or self.config['population_size']
        self.generations = generations or self.config['generations']
        
        # Hard constraints
        self.min_population = min_population
        self.max_budget = max_budget
        
        # Fitness weights
        self.fitness_weights = fitness_weights or FITNESS_WEIGHTS
        
        # Population and fitness tracking
        self.population = []
        self.fitness_results = []
        self.best_solution = None
        self.best_fitness = None
        
        # Evolution history
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_carbon': [],
            'best_carbon_ratio': [],
            'best_happiness': [],
            'best_cost': [],
            'diversity': [],
        }
        
        # Convergence tracking
        self.generations_without_improvement = 0
        self.best_fitness_ever = float('-inf')
        
        # Callback for real-time updates
        self.callback = None
    
    def initialize_population(self, method='random_weighted', seed=None):
        """
        Create initial population of random cities.
        
        Args:
            method (str): Initialization method ('random', 'random_weighted', 'template')
            seed (int): Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.population = []
        
        for i in range(self.population_size):
            city = CityGrid(size=self.grid_size)
            city.randomize(method=method, seed=seed+i if seed else None)
            self.population.append(city)
        
        # Evaluate initial population
        self.fitness_results = evaluate_population_fitness(
            self.population,
            weights=self.fitness_weights,
            min_population=self.min_population,
            max_budget=self.max_budget
        )
        
        # Track best solution
        self.best_solution, self.best_fitness = get_best_solution(
            self.population, self.fitness_results
        )
        self.best_fitness_ever = self.best_fitness.fitness
    
    def run(self, callback=None, verbose=True):
        """
        Run the genetic algorithm optimization.
        
        Args:
            callback (callable): Optional callback(generation, best_fitness, best_grid)
            verbose (bool): Print progress
            
        Returns:
            tuple: (best_city_grid, best_fitness_result, history)
        """
        self.callback = callback
        
        if verbose:
            print(f"Starting Genetic Algorithm Optimization")
            print(f"Grid Size: {self.grid_size}x{self.grid_size}")
            print(f"Population Size: {self.population_size}")
            print(f"Max Generations: {self.generations}")
            print(f"=" * 70)
        
        for generation in range(self.generations):
            # Perform one generation
            self._evolve_generation()
            
            # Update history
            self._update_history(generation)
            
            # Check for improvement
            if self.best_fitness.fitness > self.best_fitness_ever:
                self.best_fitness_ever = self.best_fitness.fitness
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
            
            # Print progress
            if verbose and generation % self.config.get('log_frequency', 10) == 0:
                self._print_progress(generation)
            
            # Callback
            if self.callback:
                self.callback(generation, self.best_fitness, self.best_solution)
            
            # Check early stopping
            if self.config['early_stopping']:
                # Stop if target achieved
                if self.best_fitness.success_achieved:
                    if verbose:
                        print(f"\n✓ Success criterion achieved at generation {generation}!")
                    break
                
                # Stop if converged (no improvement)
                if self.generations_without_improvement >= self.config['convergence_threshold']:
                    if verbose:
                        print(f"\n✓ Converged (no improvement for {self.config['convergence_threshold']} generations)")
                    break
            
            # Adaptive mutation
            if self.config['adaptive_mutation']:
                self._adjust_mutation_rate()
        
        if verbose:
            print("\n" + "=" * 70)
            print("Optimization Complete!")
            self._print_final_summary()
        
        return self.best_solution, self.best_fitness, self.history
    
    def _evolve_generation(self):
        """Perform one generation of evolution."""
        
        # 1. Selection
        parents = self._selection()
        
        # 2. Crossover and Mutation to create offspring
        offspring = []
        while len(offspring) < self.population_size - self.config['elitism_count']:
            # Select two parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Crossover
            if random.random() < self.config['crossover_probability']:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            self._mutate(child1)
            self._mutate(child2)
            
            offspring.append(child1)
            if len(offspring) < self.population_size - self.config['elitism_count']:
                offspring.append(child2)
        
        # 3. Elitism: Keep best solutions from previous generation
        elite_indices = sorted(range(len(self.fitness_results)), 
                              key=lambda i: self.fitness_results[i].fitness, 
                              reverse=True)[:self.config['elitism_count']]
        elite = [self.population[i].copy() for i in elite_indices]
        
        # 4. Form new population
        self.population = elite + offspring
        
        # 5. Evaluate new population
        self.fitness_results = evaluate_population_fitness(
            self.population,
            weights=self.fitness_weights,
            min_population=self.min_population,
            max_budget=self.max_budget
        )
        
        # 6. Update best solution
        current_best, current_best_fitness = get_best_solution(
            self.population, self.fitness_results
        )
        if current_best_fitness.fitness > self.best_fitness.fitness:
            self.best_solution = current_best.copy()
            self.best_fitness = current_best_fitness
    
    def _selection(self):
        """
        Select parents for reproduction.
        
        Returns:
            list: Selected parent city grids
        """
        method = self.config['selection_method']
        
        if method == 'tournament':
            return self._tournament_selection()
        elif method == 'roulette':
            return self._roulette_selection()
        elif method == 'rank':
            return self._rank_selection()
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _tournament_selection(self):
        """Tournament selection: Best from random subsets."""
        parents = []
        tournament_size = self.config['tournament_size']
        
        for _ in range(self.population_size):
            # Select random candidates
            candidates_idx = random.sample(range(self.population_size), tournament_size)
            # Choose best
            winner_idx = max(candidates_idx, key=lambda i: self.fitness_results[i].fitness)
            parents.append(self.population[winner_idx])
        
        return parents
    
    def _roulette_selection(self):
        """Roulette wheel selection: Probability proportional to fitness."""
        # Shift fitness to be positive (add offset)
        fitnesses = [f.fitness for f in self.fitness_results]
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        
        # Calculate probabilities
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            probabilities = [1/self.population_size] * self.population_size
        else:
            probabilities = [f / total_fitness for f in fitnesses]
        
        # Select with replacement
        parents = random.choices(self.population, weights=probabilities, k=self.population_size)
        
        return parents
    
    def _rank_selection(self):
        """Rank selection: Probability based on rank, not absolute fitness."""
        # Sort by fitness
        sorted_indices = sorted(range(self.population_size), 
                               key=lambda i: self.fitness_results[i].fitness)
        
        # Assign ranks (1 to population_size)
        ranks = list(range(1, self.population_size + 1))
        
        # Select based on ranks
        parents = random.choices([self.population[i] for i in sorted_indices],
                                weights=ranks, k=self.population_size)
        
        return parents
    
    def _crossover(self, parent1, parent2):
        """
        Create offspring by combining two parents.
        
        Args:
            parent1, parent2 (CityGrid): Parent grids
            
        Returns:
            tuple: (child1, child2) offspring grids
        """
        method = self.config['crossover_method']
        
        if method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif method == 'multi_point':
            return self._multi_point_crossover(parent1, parent2)
        elif method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {method}")
    
    def _single_point_crossover(self, parent1, parent2):
        """Single-point crossover: Split at one point."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Choose random split point
        split_row = random.randint(1, self.grid_size - 1)
        
        # Swap bottom halves
        child1.grid[split_row:, :] = parent2.grid[split_row:, :]
        child2.grid[split_row:, :] = parent1.grid[split_row:, :]
        
        return child1, child2
    
    def _multi_point_crossover(self, parent1, parent2):
        """Multi-point crossover: Split at multiple points."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        num_points = self.config['crossover_points']
        
        # Choose random split rows
        split_rows = sorted(random.sample(range(1, self.grid_size), min(num_points, self.grid_size-1)))
        
        # Alternate between parents
        swap = False
        prev_row = 0
        
        for split_row in split_rows + [self.grid_size]:
            if swap:
                child1.grid[prev_row:split_row, :] = parent2.grid[prev_row:split_row, :]
                child2.grid[prev_row:split_row, :] = parent1.grid[prev_row:split_row, :]
            swap = not swap
            prev_row = split_row
        
        return child1, child2
    
    def _uniform_crossover(self, parent1, parent2):
        """Uniform crossover: Each cell has 50% chance from each parent."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Random mask
        mask = xp.random.rand(self.grid_size, self.grid_size) < 0.5
        
        child1.grid = xp.where(mask, parent1.grid, parent2.grid)
        child2.grid = xp.where(mask, parent2.grid, parent1.grid)
        
        return child1, child2
    
    def _mutate(self, city_grid):
        """
        Apply mutation to a city grid.
        
        Args:
            city_grid (CityGrid): Grid to mutate (modified in-place)
        """
        mutation_rate = self.config['mutation_rate']
        method = self.config['mutation_method']
        
        if method == 'swap':
            self._swap_mutation(city_grid, mutation_rate)
        elif method == 'random':
            self._random_mutation(city_grid, mutation_rate)
        elif method == 'adaptive':
            # Use current mutation rate (may have been adjusted)
            self._swap_mutation(city_grid, mutation_rate)
        else:
            raise ValueError(f"Unknown mutation method: {method}")
    
    def _swap_mutation(self, city_grid, mutation_rate):
        """Swap mutation: Swap random pairs of buildings."""
        grid_size = city_grid.size
        num_mutations = int(grid_size * grid_size * mutation_rate)
        
        for _ in range(num_mutations):
            # Choose two random cells
            x1, y1 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            x2, y2 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            
            # Swap
            city_grid.swap_buildings(x1, y1, x2, y2)
    
    def _random_mutation(self, city_grid, mutation_rate):
        """Random mutation: Change cells to random building types."""
        grid_size = city_grid.size
        num_mutations = int(grid_size * grid_size * mutation_rate)
        
        for _ in range(num_mutations):
            x = random.randint(0, grid_size-1)
            y = random.randint(0, grid_size-1)
            city_grid.mutate_cell(x, y)
    
    def _adjust_mutation_rate(self):
        """Adjust mutation rate if convergence stalls."""
        threshold = self.config['convergence_threshold'] // 2
        
        if self.generations_without_improvement > threshold:
            # Increase mutation rate
            current = self.config['mutation_rate']
            max_rate = self.config['mutation_rate_max']
            self.config['mutation_rate'] = min(current * 1.2, max_rate)
        else:
            # Decrease mutation rate back to normal
            min_rate = self.config['mutation_rate_min']
            target = GA_CONFIG['mutation_rate']
            current = self.config['mutation_rate']
            self.config['mutation_rate'] = max(current * 0.9, target, min_rate)
    
    def _update_history(self, generation):
        """Update evolution history tracking."""
        fitnesses = [f.fitness for f in self.fitness_results]
        
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(self.best_fitness.fitness)
        self.history['avg_fitness'].append(np.mean(fitnesses))
        
        if self.best_fitness.metrics:
            m = self.best_fitness.metrics
            self.history['best_carbon'].append(m.net_carbon)
            self.history['best_carbon_ratio'].append(m.carbon_ratio)
            self.history['best_happiness'].append(m.happiness_score)
            self.history['best_cost'].append(m.total_cost)
        
        # Calculate diversity (average pairwise difference)
        diversity = self._calculate_diversity()
        self.history['diversity'].append(diversity)
    
    def _calculate_diversity(self):
        """Calculate population diversity (0-1)."""
        if len(self.population) < 2:
            return 0.0
        
        # Sample pairs to avoid O(n²) complexity
        num_samples = min(100, len(self.population) * (len(self.population) - 1) // 2)
        
        total_diff = 0
        for _ in range(num_samples):
            i, j = random.sample(range(len(self.population)), 2)
            grid1 = to_numpy(self.population[i].grid)
            grid2 = to_numpy(self.population[j].grid)
            diff = np.sum(grid1 != grid2) / grid1.size
            total_diff += diff
        
        return total_diff / num_samples if num_samples > 0 else 0.0
    
    def _print_progress(self, generation):
        """Print current progress."""
        m = self.best_fitness.metrics
        print(f"Gen {generation:4d} | "
              f"Fitness: {self.best_fitness.fitness:10.2f} | "
              f"Carbon: {m.net_carbon:10,.0f} ({m.carbon_ratio*100:5.2f}%) | "
              f"Pop: {m.population:6,} | "
              f"Happy: {m.happiness_score:5.1f} | "
              f"NoImprove: {self.generations_without_improvement:3d}")
    
    def _print_final_summary(self):
        """Print final optimization summary."""
        print(f"\nBest Solution Found:")
        print(f"  Fitness: {self.best_fitness.fitness:.2f}")
        print(f"  Viable: {self.best_fitness.is_viable}")
        print(f"  Success (<5% carbon): {self.best_fitness.success_achieved}")
        
        if self.best_fitness.metrics:
            m = self.best_fitness.metrics
            print(f"\n  Population: {m.population:,}")
            print(f"  Net Carbon: {m.net_carbon:,.0f} units ({m.carbon_ratio*100:.2f}%)")
            print(f"  Energy Balance: {m.energy_balance:+,.0f} units")
            print(f"  Cost: ${m.total_cost:,.0f}")
            print(f"  Happiness: {m.happiness_score:.1f}/100")


if __name__ == "__main__":
    print("Testing Genetic Algorithm...\n")
    
    # Quick test run
    ga = GeneticAlgorithm(
        grid_size=20,
        population_size=10,
        generations=50,
        min_population=1000,
        max_budget=500000
    )
    
    print("Initializing population...")
    ga.initialize_population(seed=42)
    
    print(f"Initial best fitness: {ga.best_fitness.fitness:.2f}\n")
    
    print("Running optimization...\n")
    best_city, best_fitness, history = ga.run(verbose=True)
    
    print("\nOptimization complete!")
    print(f"Generations run: {len(history['generation'])}")
    print(f"Final best fitness: {best_fitness.fitness:.2f}")
