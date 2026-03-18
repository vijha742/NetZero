"""
Zero-Carbon City Optimization - Streamlit Dashboard

Interactive web interface for the genetic algorithm city optimizer.
Allows users to configure parameters, run optimization, and visualize results in real-time.
"""

import streamlit as st
import sys
import os
import time
import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib to prevent memory issues
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['figure.max_open_warning'] = 0  # Disable warning for many figures
plt.rcParams['figure.dpi'] = 100  # Set reasonable default DPI
plt.rcParams['savefig.dpi'] = 100  # Set reasonable save DPI

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.city_grid import CityGrid
from src.optimization.genetic_algorithm import GeneticAlgorithm
from src.engine.simulation import calculate_metrics
from src.engine.spatial_constraints import evaluate_all_constraints
from src.visualization.city_map import (create_city_heatmap, create_pollution_overlay,
                                         create_before_after_comparison, create_metrics_dashboard)
from src.visualization.metrics_plots import (plot_fitness_evolution, plot_carbon_reduction,
                                              plot_multi_metric_evolution, create_optimization_summary)
from src.config.optimization_config import HARDWARE_PROFILES, load_hardware_profile

# Page configuration
st.set_page_config(
    page_title="Zero-Carbon City Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #4CAF50;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'best_city' not in st.session_state:
    st.session_state.best_city = None
if 'initial_city' not in st.session_state:
    st.session_state.initial_city = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'best_fitness' not in st.session_state:
    st.session_state.best_fitness = None

# Header
st.markdown('<div class="main-header">🌱 Zero-Carbon City Optimizer</div>', unsafe_allow_html=True)
st.markdown("### AI-Powered Urban Planning for Sustainable Cities")
st.markdown("---")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Hardware profile selection
hardware_profile = st.sidebar.selectbox(
    "Hardware Profile",
    options=list(HARDWARE_PROFILES.keys()),
    format_func=lambda x: f"{x.title()} - {HARDWARE_PROFILES[x]['description']}",
    help="Select hardware profile for optimal performance"
)

profile = HARDWARE_PROFILES[hardware_profile]

st.sidebar.info(f"""
**Selected Profile:** {hardware_profile.title()}

- Grid: {profile['grid_size']}×{profile['grid_size']}
- Population: {profile['population_size']}
- Generations: {profile['generations']}
- Expected time: {profile['expected_runtime_minutes']}
""")

# Custom parameters (optional override)
with st.sidebar.expander("🔧 Advanced Settings"):
    custom_grid_size = st.number_input("Grid Size", min_value=20, max_value=100, 
                                       value=profile['grid_size'], step=10)
    custom_pop_size = st.number_input("Population Size", min_value=10, max_value=200,
                                      value=profile['population_size'], step=10)
    custom_generations = st.number_input("Generations", min_value=50, max_value=5000,
                                         value=profile['generations'], step=50)

st.sidebar.markdown("---")

# Hard constraints
st.sidebar.header("🎯 Constraints")
min_population = st.sidebar.number_input("Minimum Population", min_value=0, value=5000, step=1000,
                                         help="Minimum required city population")
max_budget = st.sidebar.number_input("Maximum Budget ($)", min_value=0, value=1000000, step=50000,
                                      help="Maximum allowed construction cost")

st.sidebar.markdown("---")

# Fitness weights
st.sidebar.header("⚖️ Optimization Weights")
st.sidebar.markdown("Adjust importance of each objective:")

weight_carbon = st.sidebar.slider("Carbon Weight", 0.0, 2.0, 1.0, 0.1,
                                   help="Higher = prioritize carbon reduction")
weight_happiness = st.sidebar.slider("Happiness Weight", 0.0, 2.0, 0.5, 0.1,
                                      help="Higher = prioritize livability")
weight_cost = st.sidebar.slider("Cost Weight", 0.0, 2.0, 0.3, 0.1,
                                 help="Higher = prioritize cost minimization")

fitness_weights = {
    'carbon': weight_carbon,
    'happiness': weight_happiness,
    'cost': weight_cost,
    'hard_constraints': 1e6,
    'soft_constraints': 1.0,
    'target_carbon_ratio': 0.05
}

st.sidebar.markdown("---")

# Run button
run_optimization = st.sidebar.button("🚀 Start Optimization", type="primary")

# Main content
if not st.session_state.optimization_complete:
    # Before optimization view
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Grid Size", f"{custom_grid_size}×{custom_grid_size}")
    with col2:
        st.metric("Population Size", custom_pop_size)
    with col3:
        st.metric("Max Generations", custom_generations)
    
    st.markdown("---")
    
    # Information tabs
    tab1, tab2, tab3 = st.tabs(["📖 About", "🧬 Algorithm", "🏗️ Building Types"])
    
    with tab1:
        st.markdown("""
        ### About This Tool
        
        This application uses **Genetic Algorithms** to optimize urban layouts for **zero net carbon emissions** 
        while maintaining livability and cost-effectiveness.
        
        **Key Features:**
        - Multi-objective optimization (carbon, happiness, cost)
        - 8 comprehensive spatial constraints (NIMBY, heat island, transit, etc.)
        - Real-time visualization of optimization progress
        - Proof-of-concept for research and urban planning
        
        **Success Criterion:** Net carbon emissions < 5% of total emissions
        
        **Research Focus:** Demonstrates the significance of AI-driven urban planning systems 
        for sustainability, while acknowledging data gaps in real-world emission databases.
        """)
    
    with tab2:
        st.markdown("""
        ### Genetic Algorithm Process
        
        1. **Initialization**: Create random population of city layouts
        2. **Evaluation**: Calculate fitness (carbon, happiness, cost, constraints)
        3. **Selection**: Choose best cities as parents
        4. **Crossover**: Combine parent cities to create offspring
        5. **Mutation**: Random modifications for diversity
        6. **Elitism**: Preserve top solutions
        7. **Repeat**: Iterate until convergence or max generations
        
        **Fitness Function:**
        ```
        Fitness = W1×(-Carbon) + W2×Happiness + W3×(-Cost) + Constraint Scores
        ```
        
        **Convergence:** Algorithm stops when target achieved or no improvement for 100 generations.
        """)
    
    with tab3:
        from src.config.building_config import BUILDING_TYPES
        
        st.markdown("### Building Types in Simulation")
        
        for building_id, props in BUILDING_TYPES.items():
            with st.expander(f"{props['name']} ({props['short_name']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    - **Category:** {props['category']}
                    - **Carbon:** {props['carbon']:+} units
                    - **Cost:** ${props['cost']:,}
                    """)
                with col2:
                    st.markdown(f"""
                    - **Population:** {props['pop']}
                    - **Energy Demand:** {props['energy_demand']}
                    - **Energy Supply:** {props['energy_supply']}
                    """)

# Run optimization
if run_optimization:
    st.session_state.optimization_complete = False
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    
    # Initialize GA
    ga = GeneticAlgorithm(
        grid_size=custom_grid_size,
        population_size=custom_pop_size,
        generations=custom_generations,
        min_population=min_population,
        max_budget=max_budget,
        fitness_weights=fitness_weights
    )
    
    status_text.info("🔄 Initializing population...")
    ga.initialize_population(seed=42)
    
    # Store initial city
    st.session_state.initial_city = ga.population[0].copy()
    
    # Callback for real-time updates
    def update_callback(generation, best_fitness, best_grid):
        progress = min((generation + 1) / custom_generations, 1.0)
        progress_bar.progress(progress)
        
        if generation % 10 == 0:
            status_text.info(f"⚙️ Generation {generation}/{custom_generations} - "
                           f"Fitness: {best_fitness.fitness:.2f} - "
                           f"Carbon: {best_fitness.metrics.carbon_ratio*100:.2f}%")
            
            # Show metrics
            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Generation", generation)
                with col2:
                    st.metric("Best Fitness", f"{best_fitness.fitness:.2f}")
                with col3:
                    st.metric("Carbon Ratio", f"{best_fitness.metrics.carbon_ratio*100:.2f}%")
                with col4:
                    st.metric("Population", f"{best_fitness.metrics.population:,}")
    
    # Run optimization
    status_text.success("🚀 Starting optimization...")
    
    best_city, best_fitness, history = ga.run(callback=update_callback, verbose=False)
    
    # Store results
    st.session_state.best_city = best_city
    st.session_state.best_fitness = best_fitness
    st.session_state.history = history
    st.session_state.optimization_complete = True
    
    progress_bar.progress(1.0)
    status_text.success("✅ Optimization complete!")
    
    time.sleep(1)
    st.rerun()

# Show results
if st.session_state.optimization_complete:
    best_city = st.session_state.best_city
    best_fitness = st.session_state.best_fitness
    history = st.session_state.history
    initial_city = st.session_state.initial_city
    
    # Success banner
    if best_fitness.success_achieved:
        st.markdown("""
        <div class="success-box">
            <h2 style="color: #2E7D32; text-align: center;">
                ✅ SUCCESS! Net-Zero Carbon Target Achieved! 
            </h2>
            <p style="text-align: center;">Carbon ratio: {:.2f}% (Target: <5%)</p>
        </div>
        """.format(best_fitness.metrics.carbon_ratio*100), unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Target not reached. Carbon ratio: {best_fitness.metrics.carbon_ratio*100:.2f}% "
                  f"(Target: <5%). Try increasing generations or adjusting weights.")
    
    st.markdown("---")
    
    # Key metrics
    st.header("📊 Optimization Results")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Fitness Score", f"{best_fitness.fitness:.2f}")
    with col2:
        st.metric("Net Carbon", f"{best_fitness.metrics.net_carbon:,.0f}",
                 delta=f"{best_fitness.metrics.carbon_ratio*100:.2f}%")
    with col3:
        st.metric("Population", f"{best_fitness.metrics.population:,}")
    with col4:
        st.metric("Happiness", f"{best_fitness.metrics.happiness_score:.1f}/100")
    with col5:
        st.metric("Total Cost", f"${best_fitness.metrics.total_cost:,.0f}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ City Map", "📈 Evolution", "📋 Detailed Metrics", "💾 Export"])
    
    with tab1:
        st.subheader("Optimized City Layout")
        
        # Calculate metrics for both cities
        initial_constraint_results = evaluate_all_constraints(initial_city)
        initial_metrics = calculate_metrics(initial_city, initial_constraint_results)
        
        final_constraint_results = evaluate_all_constraints(best_city)
        final_metrics = calculate_metrics(best_city, final_constraint_results)
        
        # Before/After comparison
        fig, axes = create_before_after_comparison(initial_city, best_city,
                                                   initial_metrics, final_metrics)
        st.pyplot(fig)
        plt.close(fig)
        
        # Detailed dashboard
        st.subheader("Detailed City Dashboard")
        fig, axes = create_metrics_dashboard(best_city, final_metrics, final_constraint_results)
        st.pyplot(fig)
        plt.close(fig)
        
        # Pollution overlay
        with st.expander("🌫️ View Pollution Analysis"):
            fig, axes = create_pollution_overlay(best_city)
            st.pyplot(fig)
            plt.close(fig)
    
    with tab2:
        st.subheader("Optimization Progress")
        
        # Validate history data before plotting
        def validate_history(history):
            """Validate and clean history data to prevent plotting errors."""
            validated = {}
            
            # Check if history is valid
            if not isinstance(history, dict):
                st.error("Invalid history data: not a dictionary")
                return {}
            
            for key, values in history.items():
                if isinstance(values, list):
                    # Convert to list if needed
                    validated[key] = list(values)
                    # Check for extreme values that could cause issues
                    if key != 'generation' and len(values) > 0:
                        try:
                            numeric_values = [float(v) for v in values if v is not None]
                            if numeric_values:
                                max_val = max(abs(v) for v in numeric_values)
                                min_val = min(v for v in numeric_values)
                                avg_val = sum(numeric_values) / len(numeric_values)
                                
                                # Debug output
                                with st.expander(f"Debug: {key} statistics"):
                                    st.write(f"Min: {min_val:.2e}, Max: {max_val:.2e}, Avg: {avg_val:.2e}, Count: {len(numeric_values)}")
                                
                                if max_val > 1e10:
                                    st.warning(f"Warning: Extremely large values detected in {key} (max: {max_val:.2e}). Data may be corrupted.")
                        except (ValueError, TypeError) as e:
                            st.warning(f"Warning: Could not validate {key}: {e}")
                else:
                    validated[key] = values
            return validated
        
        validated_history = validate_history(history)
        
        # Show history data length for debugging
        if len(validated_history.get('generation', [])) > 0:
            st.info(f"Showing optimization data for {len(validated_history['generation'])} generations")
        
        try:
            # Fitness evolution
            fig, ax = plot_fitness_evolution(validated_history)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting fitness evolution: {str(e)}")
        
        try:
            # Carbon reduction
            fig, axes = plot_carbon_reduction(validated_history)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting carbon reduction: {str(e)}")
        
        try:
            # Multi-metric evolution
            fig, axes = plot_multi_metric_evolution(validated_history)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting multi-metric evolution: {str(e)}")
        
        try:
            # Summary
            # Add additional safety checks
            if len(validated_history.get('generation', [])) > 0:
                fig, axes = create_optimization_summary(validated_history, initial_metrics, final_metrics)
                st.pyplot(fig, width='stretch')
                plt.close(fig)
            else:
                st.warning("No history data available for summary plot")
        except Exception as e:
            st.error(f"Error plotting optimization summary: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    with tab3:
        st.subheader("Comprehensive Metrics Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Initial City")
            st.json(initial_metrics.to_dict())
        
        with col2:
            st.markdown("### Optimized City")
            st.json(final_metrics.to_dict())
        
        st.markdown("### Optimization History")
        import pandas as pd
        df = pd.DataFrame(history)
        st.dataframe(df)
    
    with tab4:
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Export Grid Layout")
            if st.button("Download City Grid (CSV)"):
                csv_path = "/tmp/optimized_city_grid.csv"
                best_city.export_to_csv(csv_path)
                with open(csv_path, 'r') as f:
                    st.download_button("Download CSV", f.read(), "optimized_city.csv", "text/csv")
        
        with col2:
            st.markdown("### Export Metrics")
            if st.button("Download Metrics Report (JSON)"):
                import json
                report = {
                    'initial_metrics': initial_metrics.to_dict(),
                    'final_metrics': final_metrics.to_dict(),
                    'history': history,
                    'configuration': {
                        'grid_size': custom_grid_size,
                        'population_size': custom_pop_size,
                        'generations': custom_generations,
                        'min_population': min_population,
                        'max_budget': max_budget,
                        'fitness_weights': fitness_weights
                    }
                }
                st.download_button("Download JSON", json.dumps(report, indent=2),
                                  "optimization_report.json", "application/json")
        
        st.markdown("### Summary Text Report")
        summary_text = f"""
ZERO-CARBON CITY OPTIMIZATION REPORT
=====================================

Configuration:
- Grid Size: {custom_grid_size}×{custom_grid_size}
- Population Size: {custom_pop_size}
- Generations: {len(history['generation'])}
- Hardware Profile: {hardware_profile}

Results:
- Success: {'YES ✓' if best_fitness.success_achieved else 'NO'}
- Final Fitness: {best_fitness.fitness:.2f}
- Net Carbon: {final_metrics.net_carbon:,.0f} units ({final_metrics.carbon_ratio*100:.2f}%)
- Population: {final_metrics.population:,}
- Happiness: {final_metrics.happiness_score:.1f}/100
- Total Cost: ${final_metrics.total_cost:,.0f}

Improvements:
- Carbon Reduction: {initial_metrics.net_carbon - final_metrics.net_carbon:,.0f} units
- Carbon Ratio: {initial_metrics.carbon_ratio*100:.2f}% → {final_metrics.carbon_ratio*100:.2f}%
- Happiness Gain: {final_metrics.happiness_score - initial_metrics.happiness_score:.1f} points
"""
        st.text_area("Report", summary_text, height=400)
        st.download_button("Download Report", summary_text, "optimization_report.txt", "text/plain")
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start New Optimization"):
        st.session_state.optimization_complete = False
        st.session_state.best_city = None
        st.session_state.initial_city = None
        st.session_state.history = None
        st.session_state.best_fitness = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Zero-Carbon City Simulation</strong> - Research Prototype</p>
    <p>Demonstrates AI-driven urban planning for sustainability</p>
    <p>Built with Python, NumPy, Matplotlib, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
