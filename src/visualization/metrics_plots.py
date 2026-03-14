"""
Metrics Plotting for Optimization Visualization

This module creates time-series plots showing optimization progress:
- Fitness evolution over generations
- Carbon reduction trends
- Constraint violation trends
- Population diversity
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_evolution(history, figsize=(12, 6)):
    """
    Plot fitness evolution over generations.
    
    Args:
        history (dict): GA history dictionary
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    generations = history['generation']
    best_fitness = history['best_fitness']
    avg_fitness = history['avg_fitness']
    
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    ax.plot(generations, avg_fitness, 'r--', linewidth=1.5, label='Average Fitness', alpha=0.7)
    
    ax.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Highlight improvements
    improvements = []
    prev_best = best_fitness[0]
    for i, fit in enumerate(best_fitness):
        if fit > prev_best:
            improvements.append((generations[i], fit))
            prev_best = fit
    
    if improvements:
        imp_gens, imp_fits = zip(*improvements)
        ax.scatter(imp_gens, imp_fits, color='green', s=50, zorder=5, label='Improvements')
    
    plt.tight_layout()
    
    return fig, ax


def plot_carbon_reduction(history, target_ratio=0.05, figsize=(12, 6)):
    """
    Plot carbon emission reduction over generations.
    
    Args:
        history (dict): GA history dictionary
        target_ratio (float): Target carbon ratio (default 5%)
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, ax)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    generations = history['generation']
    carbon = history['best_carbon']
    carbon_ratio = history['best_carbon_ratio']
    
    # Net carbon plot
    ax1.plot(generations, carbon, 'r-', linewidth=2)
    ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Net Zero')
    ax1.fill_between(generations, carbon, 0, where=(np.array(carbon) > 0), 
                     color='red', alpha=0.2, label='Emissions')
    ax1.fill_between(generations, carbon, 0, where=(np.array(carbon) <= 0),
                     color='green', alpha=0.2, label='Carbon Negative')
    
    ax1.set_title('Net Carbon Emissions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Generation', fontsize=10)
    ax1.set_ylabel('Net Carbon (units)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Carbon ratio plot
    ax2.plot(generations, np.array(carbon_ratio) * 100, 'b-', linewidth=2)
    ax2.axhline(y=target_ratio * 100, color='green', linestyle='--', linewidth=2, 
               label=f'Target ({target_ratio*100:.0f}%)')
    ax2.fill_between(generations, np.array(carbon_ratio) * 100, target_ratio * 100,
                    where=(np.array(carbon_ratio) > target_ratio), 
                    color='red', alpha=0.2, label='Above Target')
    ax2.fill_between(generations, 0, np.array(carbon_ratio) * 100,
                    where=(np.array(carbon_ratio) <= target_ratio),
                    color='green', alpha=0.2, label='Success Zone')
    
    ax2.set_title('Carbon Ratio Progress', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Generation', fontsize=10)
    ax2.set_ylabel('Carbon Ratio (%)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Carbon Reduction Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_multi_metric_evolution(history, figsize=(14, 10)):
    """
    Plot multiple metrics evolution in a dashboard layout.
    
    Args:
        history (dict): GA history dictionary
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    generations = history['generation']
    
    # 1. Fitness
    ax = axes[0, 0]
    ax.plot(generations, history['best_fitness'], 'b-', linewidth=2)
    ax.set_title('Fitness Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.3)
    
    # 2. Carbon
    ax = axes[0, 1]
    ax.plot(generations, np.array(history['best_carbon_ratio']) * 100, 'r-', linewidth=2)
    ax.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Target (5%)')
    ax.set_title('Carbon Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Carbon Ratio (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Happiness
    ax = axes[1, 0]
    ax.plot(generations, history['best_happiness'], 'g-', linewidth=2)
    ax.set_title('Happiness Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Happiness')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    # 4. Diversity
    ax = axes[1, 1]
    ax.plot(generations, np.array(history['diversity']) * 100, 'purple', linewidth=2)
    ax.set_title('Population Diversity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity (%)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Optimization Metrics Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


def plot_convergence_analysis(history, figsize=(12, 8)):
    """
    Analyze and plot convergence characteristics.
    
    Args:
        history (dict): GA history dictionary
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    generations = np.array(history['generation'])
    best_fitness = np.array(history['best_fitness'])
    avg_fitness = np.array(history['avg_fitness'])
    
    # 1. Fitness with improvement rate
    ax = axes[0, 0]
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    
    # Calculate improvement rate
    if len(best_fitness) > 10:
        window = 10
        improvement_rate = np.convolve(np.diff(best_fitness), np.ones(window)/window, mode='valid')
        ax2 = ax.twinx()
        ax2.plot(generations[window:], improvement_rate, 'r--', alpha=0.5, label='Improvement Rate')
        ax2.set_ylabel('Improvement Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_title('Fitness & Improvement Rate', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    
    # 2. Fitness variance (best vs average)
    ax = axes[0, 1]
    fitness_gap = best_fitness - avg_fitness
    ax.plot(generations, fitness_gap, 'purple', linewidth=2)
    ax.fill_between(generations, 0, fitness_gap, alpha=0.3, color='purple')
    ax.set_title('Selection Pressure (Best - Avg)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Gap')
    ax.grid(True, alpha=0.3)
    
    # 3. Convergence speed (generations to milestones)
    ax = axes[1, 0]
    
    # Find milestones (25%, 50%, 75%, 90% of final improvement)
    initial_fitness = best_fitness[0]
    final_fitness = best_fitness[-1]
    improvement_range = final_fitness - initial_fitness
    
    milestones = [0.25, 0.5, 0.75, 0.9]
    milestone_gens = []
    
    for milestone in milestones:
        target = initial_fitness + improvement_range * milestone
        idx = np.where(best_fitness >= target)[0]
        if len(idx) > 0:
            milestone_gens.append(generations[idx[0]])
        else:
            milestone_gens.append(generations[-1])
    
    ax.bar(['25%', '50%', '75%', '90%'], milestone_gens, color=['lightblue', 'blue', 'darkblue', 'navy'])
    ax.set_title('Generations to Reach Milestones', fontsize=12, fontweight='bold')
    ax.set_ylabel('Generation')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Cumulative improvement
    ax = axes[1, 1]
    cumulative_improvement = (best_fitness - initial_fitness) / improvement_range * 100
    ax.plot(generations, cumulative_improvement, 'g-', linewidth=2)
    ax.fill_between(generations, 0, cumulative_improvement, alpha=0.3, color='green')
    ax.set_title('Cumulative Improvement', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Progress (%)')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


def create_optimization_summary(history, initial_metrics, final_metrics, figsize=(14, 8)):
    """
    Create a comprehensive summary visualization of optimization results.
    
    Args:
        history (dict): GA history dictionary
        initial_metrics (CityMetrics): Initial city metrics
        final_metrics (CityMetrics): Final city metrics
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    generations = history['generation']
    
    # 1. Fitness evolution (top-left, large)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(generations, history['best_fitness'], 'b-', linewidth=2.5, label='Best')
    ax1.plot(generations, history['avg_fitness'], 'r--', linewidth=1.5, label='Average')
    ax1.set_title('Fitness Evolution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Before/After comparison (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    metrics_comparison = {
        'Population': [initial_metrics.population, final_metrics.population],
        'Carbon': [initial_metrics.net_carbon, final_metrics.net_carbon],
        'Happiness': [initial_metrics.happiness_score, final_metrics.happiness_score]
    }
    
    x = np.arange(len(metrics_comparison))
    width = 0.35
    
    for i, (metric_name, values) in enumerate(metrics_comparison.items()):
        ax2.text(i, max(values) * 1.1, metric_name, ha='center', fontweight='bold', fontsize=9)
    
    ax2.axis('off')
    ax2.set_title('Improvement Summary', fontsize=12, fontweight='bold')
    
    # Add text summary
    summary_text = f"""
Initial → Final

Population:
{initial_metrics.population:,} → {final_metrics.population:,}

Net Carbon:
{initial_metrics.net_carbon:,.0f} → {final_metrics.net_carbon:,.0f}
({initial_metrics.carbon_ratio*100:.1f}% → {final_metrics.carbon_ratio*100:.1f}%)

Happiness:
{initial_metrics.happiness_score:.1f} → {final_metrics.happiness_score:.1f}

Cost:
${initial_metrics.total_cost:,.0f} → ${final_metrics.total_cost:,.0f}
"""
    ax2.text(0.5, 0.5, summary_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='center', horizontalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Carbon reduction (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(generations, np.array(history['best_carbon_ratio']) * 100, 'r-', linewidth=2)
    ax3.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Target')
    ax3.fill_between(generations, 0, np.array(history['best_carbon_ratio']) * 100,
                    where=(np.array(history['best_carbon_ratio']) <= 0.05),
                    color='green', alpha=0.2)
    ax3.set_title('Carbon Ratio Progress', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Carbon Ratio (%)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Happiness (bottom-middle)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(generations, history['best_happiness'], 'g-', linewidth=2)
    ax4.set_title('Happiness Evolution', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Happiness Score')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3)
    
    # 5. Diversity (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(generations, np.array(history['diversity']) * 100, 'purple', linewidth=2)
    ax5.set_title('Population Diversity', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Generation')
    ax5.set_ylabel('Diversity (%)')
    ax5.grid(True, alpha=0.3)
    
    # Overall title
    success_text = "✓ SUCCESS" if final_metrics.carbon_ratio < 0.05 else "OPTIMIZED"
    plt.suptitle(f'Optimization Results - {success_text}', 
                fontsize=14, fontweight='bold',
                color='green' if final_metrics.carbon_ratio < 0.05 else 'blue')
    
    return fig, fig.axes


if __name__ == "__main__":
    # Test with mock data
    print("Testing metrics plotting module...")
    
    # Create mock history
    gens = list(range(100))
    history = {
        'generation': gens,
        'best_fitness': [100 + i * 10 + np.random.randn() * 5 for i in gens],
        'avg_fitness': [80 + i * 8 + np.random.randn() * 10 for i in gens],
        'best_carbon': [10000 - i * 95 + np.random.randn() * 100 for i in gens],
        'best_carbon_ratio': [0.5 - i * 0.004 + np.random.randn() * 0.01 for i in gens],
        'best_happiness': [50 + i * 0.3 + np.random.randn() * 2 for i in gens],
        'best_cost': [500000 - i * 1000 for i in gens],
        'diversity': [0.8 - i * 0.005 for i in gens],
    }
    
    # Clamp values
    history['best_carbon_ratio'] = [max(0, min(1, x)) for x in history['best_carbon_ratio']]
    history['best_happiness'] = [max(0, min(100, x)) for x in history['best_happiness']]
    history['diversity'] = [max(0, min(1, x)) for x in history['diversity']]
    
    print("Creating test plots...")
    
    # Test fitness evolution
    fig, ax = plot_fitness_evolution(history)
    plt.savefig('/tmp/test_fitness_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved fitness evolution plot")
    
    # Test carbon reduction
    fig, axes = plot_carbon_reduction(history)
    plt.savefig('/tmp/test_carbon_reduction.png', dpi=150, bbox_inches='tight')
    print("Saved carbon reduction plot")
    
    # Test multi-metric
    fig, axes = plot_multi_metric_evolution(history)
    plt.savefig('/tmp/test_multi_metric.png', dpi=150, bbox_inches='tight')
    print("Saved multi-metric plot")
    
    # Test convergence analysis
    fig, axes = plot_convergence_analysis(history)
    plt.savefig('/tmp/test_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved convergence analysis plot")
    
    print("\n✓ All test plots created successfully!")
    plt.close('all')
