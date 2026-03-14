"""
City Map Visualization

This module creates visual representations of city layouts including:
- Color-coded heatmaps showing building types
- Pollution overlay layers
- Constraint violation highlighting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.building_config import BUILDING_TYPES
from utils.array_backend import to_numpy
from utils.distance import calculate_pollution_overlay


def create_city_heatmap(city_grid, title="City Layout", figsize=(12, 10), show_grid=False):
    """
    Create a color-coded heatmap of the city.
    
    Args:
        city_grid (CityGrid): The city grid to visualize
        title (str): Plot title
        figsize (tuple): Figure size
        show_grid (bool): Show grid lines
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axis
    """
    grid_numpy = to_numpy(city_grid.grid)
    
    # Create custom colormap based on building colors
    colors = [BUILDING_TYPES[i]['color'] for i in range(len(BUILDING_TYPES))]
    cmap = ListedColormap(colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(grid_numpy, cmap=cmap, vmin=0, vmax=len(BUILDING_TYPES)-1, 
                   interpolation='nearest')
    
    # Add grid lines if requested
    if show_grid:
        ax.set_xticks(np.arange(-0.5, city_grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, city_grid.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Grid X (each cell = {city_grid.cell_size_meters}m)', fontsize=12)
    ax.set_ylabel(f'Grid Y (each cell = {city_grid.cell_size_meters}m)', fontsize=12)
    
    # Create legend
    legend_patches = []
    for building_type, props in BUILDING_TYPES.items():
        patch = mpatches.Patch(color=props['color'], label=props['short_name'])
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left',
             fontsize=10, frameon=True)
    
    plt.tight_layout()
    
    return fig, ax


def create_pollution_overlay(city_grid, pollution_sources=None, figsize=(12, 10)):
    """
    Create a pollution intensity overlay visualization.
    
    Args:
        city_grid (CityGrid): The city grid
        pollution_sources (list): Building types that emit pollution. 
                                  If None, uses factories and coal plants.
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axis
    """
    if pollution_sources is None:
        pollution_sources = [3, 4]  # Factory, Coal
    
    # Calculate pollution overlay
    pollution_map = calculate_pollution_overlay(
        city_grid.grid, 
        pollution_sources, 
        pollution_radius=5.0,
        decay_factor=0.5
    )
    
    pollution_numpy = to_numpy(pollution_map)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: City layout
    grid_numpy = to_numpy(city_grid.grid)
    colors = [BUILDING_TYPES[i]['color'] for i in range(len(BUILDING_TYPES))]
    cmap = ListedColormap(colors)
    ax1.imshow(grid_numpy, cmap=cmap, vmin=0, vmax=len(BUILDING_TYPES)-1)
    ax1.set_title('City Layout', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Pollution overlay
    pollution_cmap = LinearSegmentedColormap.from_list('pollution', 
                                                       ['white', 'yellow', 'orange', 'red'])
    im = ax2.imshow(pollution_numpy, cmap=pollution_cmap, vmin=0, vmax=1, alpha=0.8)
    ax2.set_title('Pollution Distribution', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Pollution Intensity', rotation=270, labelpad=20)
    
    plt.suptitle('City Pollution Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def create_before_after_comparison(city_before, city_after, 
                                   metrics_before=None, metrics_after=None,
                                   figsize=(16, 8)):
    """
    Create side-by-side comparison of two city layouts.
    
    Args:
        city_before (CityGrid): Initial city
        city_after (CityGrid): Optimized city
        metrics_before (CityMetrics): Metrics for initial city
        metrics_after (CityMetrics): Metrics for optimized city
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes) matplotlib figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare colormap
    colors = [BUILDING_TYPES[i]['color'] for i in range(len(BUILDING_TYPES))]
    cmap = ListedColormap(colors)
    
    # Before
    grid_before = to_numpy(city_before.grid)
    ax1.imshow(grid_before, cmap=cmap, vmin=0, vmax=len(BUILDING_TYPES)-1)
    title_before = 'Before Optimization'
    if metrics_before:
        title_before += f'\nCarbon: {metrics_before.net_carbon:,.0f} ({metrics_before.carbon_ratio*100:.1f}%)'
    ax1.set_title(title_before, fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # After
    grid_after = to_numpy(city_after.grid)
    ax2.imshow(grid_after, cmap=cmap, vmin=0, vmax=len(BUILDING_TYPES)-1)
    title_after = 'After Optimization'
    if metrics_after:
        title_after += f'\nCarbon: {metrics_after.net_carbon:,.0f} ({metrics_after.carbon_ratio*100:.1f}%)'
        if metrics_after.success_achieved():
            title_after += ' ✓'
    ax2.set_title(title_after, fontsize=14, fontweight='bold', color='green' if metrics_after and metrics_after.success_achieved() else 'black')
    ax2.axis('off')
    
    # Add legend
    legend_patches = []
    for building_type, props in BUILDING_TYPES.items():
        patch = mpatches.Patch(color=props['color'], label=props['short_name'])
        legend_patches.append(patch)
    
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(BUILDING_TYPES),
              bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.suptitle('Optimization Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def save_city_visualization(city_grid, filepath, dpi=150):
    """
    Save city visualization to file.
    
    Args:
        city_grid (CityGrid): The city to visualize
        filepath (str): Output file path
        dpi (int): Image resolution
    """
    fig, ax = create_city_heatmap(city_grid)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_building_distribution_chart(city_grid, figsize=(10, 6)):
    """
    Create a bar chart showing building type distribution.
    
    Args:
        city_grid (CityGrid): The city grid
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, ax)
    """
    counts = city_grid.get_building_counts()
    
    # Prepare data
    building_names = []
    building_counts = []
    building_colors = []
    
    for building_type in sorted(counts.keys()):
        building_names.append(BUILDING_TYPES[building_type]['short_name'])
        building_counts.append(counts[building_type])
        building_colors.append(BUILDING_TYPES[building_type]['color'])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(building_names, building_counts, color=building_colors, edgecolor='black', linewidth=1)
    
    ax.set_title('Building Type Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Building Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig, ax


def create_metrics_dashboard(city_grid, metrics, constraint_results=None, figsize=(16, 10)):
    """
    Create a comprehensive dashboard showing city state.
    
    Args:
        city_grid (CityGrid): The city grid
        metrics (CityMetrics): City metrics
        constraint_results (dict): Spatial constraint results
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. City map (large, top-left)
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    grid_numpy = to_numpy(city_grid.grid)
    colors = [BUILDING_TYPES[i]['color'] for i in range(len(BUILDING_TYPES))]
    cmap = ListedColormap(colors)
    ax_map.imshow(grid_numpy, cmap=cmap, vmin=0, vmax=len(BUILDING_TYPES)-1)
    ax_map.set_title('City Layout', fontsize=12, fontweight='bold')
    ax_map.axis('off')
    
    # 2. Building distribution (top-right)
    ax_dist = fig.add_subplot(gs[0, 2])
    counts = city_grid.get_building_counts()
    names = [BUILDING_TYPES[bt]['short_name'] for bt in sorted(counts.keys())]
    values = [counts[bt] for bt in sorted(counts.keys())]
    colors_dist = [BUILDING_TYPES[bt]['color'] for bt in sorted(counts.keys())]
    ax_dist.pie(values, labels=names, colors=colors_dist, autopct='%1.0f%%', textprops={'fontsize': 8})
    ax_dist.set_title('Building Mix', fontsize=10, fontweight='bold')
    
    # 3. Key metrics (middle-right)
    ax_metrics = fig.add_subplot(gs[1, 2])
    ax_metrics.axis('off')
    metrics_text = f"""
KEY METRICS

Population: {metrics.population:,}
Jobs: {metrics.jobs:,}

Net Carbon: {metrics.net_carbon:,.0f}
Carbon Ratio: {metrics.carbon_ratio*100:.2f}%
{"✓ SUCCESS" if metrics.carbon_ratio < 0.05 else "✗ Above 5%"}

Energy: {metrics.energy_balance:+,.0f}
{"✓ Balanced" if metrics.energy_balance >= 0 else "✗ Deficit"}

Cost: ${metrics.total_cost:,.0f}
Happiness: {metrics.happiness_score:.1f}/100
"""
    ax_metrics.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 4. Carbon breakdown (bottom-left)
    ax_carbon = fig.add_subplot(gs[2, 0])
    carbon_data = {
        'Emissions': metrics.total_emissions,
        'Absorption': metrics.total_absorption,
        'Net': abs(metrics.net_carbon)
    }
    bars = ax_carbon.bar(carbon_data.keys(), carbon_data.values(), 
                        color=['red', 'green', 'orange'], alpha=0.7)
    ax_carbon.set_title('Carbon Budget', fontsize=10, fontweight='bold')
    ax_carbon.set_ylabel('Units')
    ax_carbon.grid(axis='y', alpha=0.3)
    
    # 5. Energy breakdown (bottom-middle)
    ax_energy = fig.add_subplot(gs[2, 1])
    energy_data = {
        'Demand': metrics.energy_demand,
        'Supply': metrics.energy_supply
    }
    ax_energy.bar(energy_data.keys(), energy_data.values(), 
                 color=['orange', 'blue'], alpha=0.7)
    ax_energy.set_title('Energy Balance', fontsize=10, fontweight='bold')
    ax_energy.set_ylabel('Units')
    ax_energy.axhline(y=metrics.energy_demand, color='r', linestyle='--', alpha=0.5, label='Demand')
    ax_energy.grid(axis='y', alpha=0.3)
    
    # 6. Constraint violations (bottom-right)
    if constraint_results:
        ax_constraints = fig.add_subplot(gs[2, 2])
        constraint_names = []
        violation_counts = []
        
        for name, result in constraint_results.items():
            if result.get('violations', 0) > 0:
                constraint_names.append(name[:8])  # Abbreviate
                violation_counts.append(result['violations'])
        
        if constraint_names:
            ax_constraints.barh(constraint_names, violation_counts, color='red', alpha=0.7)
            ax_constraints.set_title('Violations', fontsize=10, fontweight='bold')
            ax_constraints.set_xlabel('Count')
        else:
            ax_constraints.text(0.5, 0.5, '✓ No Violations', 
                              ha='center', va='center', fontsize=12, color='green', fontweight='bold')
            ax_constraints.axis('off')
    
    plt.suptitle(f'City Dashboard - {city_grid.size}x{city_grid.size} Grid', 
                fontsize=14, fontweight='bold')
    
    return fig, fig.axes


if __name__ == "__main__":
    # Test visualizations
    import sys
    sys.path.insert(0, '..')
    
    from models.city_grid import CityGrid
    from engine.simulation import calculate_metrics
    from engine.spatial_constraints import evaluate_all_constraints
    
    print("Testing visualization module...")
    
    # Create test city
    city = CityGrid(size=30)
    city.randomize(method='random_weighted', seed=42)
    
    # Calculate metrics
    constraint_results = evaluate_all_constraints(city)
    metrics = calculate_metrics(city, constraint_results)
    
    # Test heatmap
    print("Creating city heatmap...")
    fig, ax = create_city_heatmap(city, title="Test City")
    plt.savefig('/tmp/test_city_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/test_city_heatmap.png")
    
    # Test pollution overlay
    print("Creating pollution overlay...")
    fig, axes = create_pollution_overlay(city)
    plt.savefig('/tmp/test_pollution.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/test_pollution.png")
    
    # Test dashboard
    print("Creating metrics dashboard...")
    fig, axes = create_metrics_dashboard(city, metrics, constraint_results)
    plt.savefig('/tmp/test_dashboard.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/test_dashboard.png")
    
    print("\n✓ All visualizations created successfully!")
