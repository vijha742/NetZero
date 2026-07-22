"""
Zero-Carbon City Optimization - Streamlit Dashboard

Interactive web interface for the genetic algorithm city optimizer.
Allows users to configure parameters, run optimization, and visualize results in real-time.
Includes NSGA-II multi-objective baseline for comparison.
"""

import streamlit as st
import sys
import os
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Configure matplotlib to prevent memory issues
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['figure.max_open_warning'] = 0  # Disable warning for many figures
plt.rcParams['figure.dpi'] = 100  # Set reasonable default DPI
plt.rcParams['savefig.dpi'] = 100  # Set reasonable save DPI

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.city_grid import CityGrid
from src.optimization.genetic_algorithm import GeneticAlgorithm
from src.optimization.nsga2 import NSGA2, compare_ga_vs_nsga2
from src.engine.simulation import calculate_metrics
from src.engine.spatial_constraints import evaluate_all_constraints
from src.visualization.city_map import (create_city_heatmap, create_pollution_overlay,
                                         create_before_after_comparison, create_metrics_dashboard)
from src.visualization.metrics_plots import (plot_fitness_evolution, plot_carbon_reduction,
                                              plot_multi_metric_evolution, create_optimization_summary)
from src.config.optimization_config import HARDWARE_PROFILES, load_hardware_profile

# ─── Page configuration ───────────────────────────────────────────────────────
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
    .nsga2-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1976D2;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session state initialisation ─────────────────────────────────────────────
for key in ['optimization_complete', 'best_city', 'initial_city', 'history',
            'best_fitness', 'nsga2_complete', 'nsga2_pareto', 'nsga2_history',
            'nsga2_initial_city', 'optimizer_mode']:
    if key not in st.session_state:
        st.session_state[key] = None if key not in (
            'optimization_complete', 'nsga2_complete') else False

if st.session_state.optimizer_mode is None:
    st.session_state.optimizer_mode = 'GA'

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌱 Zero-Carbon City Optimizer</div>', unsafe_allow_html=True)
st.markdown("### AI-Powered Urban Planning for Sustainable Cities")
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# Optimizer selection
optimizer_choice = st.sidebar.radio(
    "🧬 Optimizer",
    options=["GA (Weighted-Sum)", "NSGA-II (Multi-Objective)"],
    index=0 if st.session_state.optimizer_mode == 'GA' else 1,
    help=("GA uses a single weighted fitness score.\n"
          "NSGA-II evolves a Pareto front across all three objectives simultaneously.")
)
st.session_state.optimizer_mode = 'GA' if optimizer_choice.startswith('GA') else 'NSGA2'
mode = st.session_state.optimizer_mode

# Hardware profile
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

# Custom parameters
with st.sidebar.expander("🔧 Advanced Settings"):
    custom_grid_size = st.number_input("Grid Size", min_value=20, max_value=100,
                                       value=profile['grid_size'], step=10)
    custom_pop_size = st.number_input("Population Size", min_value=10, max_value=200,
                                      value=profile['population_size'], step=10)
    custom_generations = st.number_input("Generations", min_value=50, max_value=5000,
                                         value=profile['generations'], step=50)

st.sidebar.markdown("---")

# Hard constraints (shared by both optimizers)
st.sidebar.header("🎯 Hard Constraints")
min_population = st.sidebar.number_input("Minimum Population", min_value=0, value=5000, step=1000,
                                         help="Minimum required city population (strictly enforced)")
max_budget = st.sidebar.number_input("Maximum Budget ($)", min_value=0, value=1_000_000, step=50_000,
                                      help="Maximum allowed construction cost (strictly enforced)")

st.sidebar.markdown("---")

# ─── GA-specific sidebar section ──────────────────────────────────────────────
if mode == 'GA':
    st.sidebar.header("⚖️ Optimization Weights")
    st.sidebar.markdown("Adjust importance of each objective (all normalized to [0,1]):  \n"
                        "_Weights now control genuine trade-off ratios._")

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
        'soft_constraints': 0.1,
        'target_carbon_ratio': 0.05,
    }

    st.sidebar.markdown("---")
    run_optimization = st.sidebar.button("🚀 Start GA Optimization", type="primary")

# ─── NSGA-II-specific sidebar section ─────────────────────────────────────────
else:
    st.sidebar.header("🔀 NSGA-II Settings")
    st.sidebar.markdown(
        "NSGA-II needs no weights — it evolves a **Pareto front** across "
        "all three objectives simultaneously, letting you pick the best trade-off "
        "after optimization."
    )

    nsga2_mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.20, 0.05, 0.01,
                                             help="Per-cell point-mutation probability")
    nsga2_crossover_prob = st.sidebar.slider("Crossover Probability", 0.4, 1.0, 0.7, 0.05,
                                              help="Probability of applying 2D-block crossover")

    st.sidebar.markdown("---")
    run_optimization = st.sidebar.button("🚀 Start NSGA-II Optimization", type="primary")

# ─── Pre-run info panels ───────────────────────────────────────────────────────
completed = (st.session_state.optimization_complete if mode == 'GA'
             else st.session_state.nsga2_complete)

if not completed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Grid Size", f"{custom_grid_size}×{custom_grid_size}")
    with col2:
        st.metric("Population Size", custom_pop_size)
    with col3:
        st.metric("Max Generations", custom_generations)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📖 About", "🧬 GA Algorithm", "🔀 NSGA-II", "🏗️ Building Types"])

    with tab1:
        st.markdown("""
        ### About This Tool

        This application uses **Genetic Algorithms** and **NSGA-II** to optimize urban layouts
        for **zero net carbon emissions** while maintaining livability and cost-effectiveness.

        **Key Features:**
        - Multi-objective optimization (carbon, happiness, cost) — *properly normalized*
        - Hard constraints strictly enforced (Population, Budget, Energy)
        - 8 comprehensive spatial constraints (NIMBY, heat island, transit, etc.)
        - NSGA-II Pareto-front baseline alongside weighted-sum GA
        - Real-time visualization of optimization progress

        **Success Criterion:** Net carbon emissions < 5% of total emissions
        """)

    with tab2:
        st.markdown("""
        ### Genetic Algorithm (Weighted-Sum)

        1. **Initialization**: Create random population of city layouts
        2. **Evaluation**: Normalized fitness — carbon, happiness & cost each mapped to [0,1]
        3. **Selection**: Tournament selection
        4. **Crossover**: *2D block crossover* (preserves spatial neighbourhood clusters)
        5. **Mutation**: *Point mutation* (changes building type — explores composition space)
        6. **Elitism**: Preserve top solutions
        7. **Repeat**: Iterate until convergence or max generations

        **Normalized Fitness Function (Eq. 1):**
        ```
        f = W_C×(1−carbon_norm) + W_H×happiness_norm + W_$×(1−cost_norm)
        ```
        where each norm ∈ [0,1], eliminating scale dominance.

        **Hard constraints** → fitness = −∞ (never selected as parents).
        """)

    with tab3:
        st.markdown("""
        ### NSGA-II (Multi-Objective Pareto Baseline)

        NSGA-II treats the three objectives independently — **no weights needed**:
        - Minimise **net carbon**
        - Maximise **happiness**
        - Minimise **total cost**

        **Algorithm:**
        1. Non-dominated sort → assign Pareto rank to every individual
        2. Crowding-distance assignment → preserve spread on the Pareto front
        3. Binary tournament selection (rank first, crowding-distance for ties)
        4. 2D-block crossover + point mutation (same operators as GA)
        5. Combine parent + offspring (2N), select best N for next generation

        **Output:** A set of Pareto-optimal city layouts representing the full trade-off curve.
        You can pick the solution that best matches your priorities after the run.

        **Hard constraints** → infeasible individuals dominated by all feasible ones.
        """)

    with tab4:
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

# ─── GA Run ───────────────────────────────────────────────────────────────────
if run_optimization and mode == 'GA':
    st.session_state.optimization_complete = False

    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()

    ga = GeneticAlgorithm(
        grid_size=custom_grid_size,
        population_size=custom_pop_size,
        generations=custom_generations,
        min_population=min_population,
        max_budget=max_budget,
        fitness_weights=fitness_weights
    )

    status_text.info("🔄 Initializing population…")
    ga.initialize_population(seed=42)

    st.session_state.initial_city = ga.population[0].copy()

    def update_callback(generation, best_fitness, best_grid):
        progress = min((generation + 1) / custom_generations, 1.0)
        progress_bar.progress(progress)
        if generation % 10 == 0:
            carbon_pct = best_fitness.metrics.carbon_ratio * 100 if best_fitness.metrics else 0
            status_text.info(
                f"⚙️ Generation {generation}/{custom_generations} — "
                f"Fitness: {best_fitness.fitness:.4f} — "
                f"Carbon: {carbon_pct:.2f}%"
            )
            if best_fitness.metrics:
                with metrics_container.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Generation", generation)
                    c2.metric("Best Fitness", f"{best_fitness.fitness:.4f}")
                    c3.metric("Carbon Ratio", f"{carbon_pct:.2f}%")
                    c4.metric("Population", f"{best_fitness.metrics.population:,}")

    status_text.success("🚀 Starting GA optimization…")
    best_city, best_fitness, history = ga.run(callback=update_callback, verbose=False)

    st.session_state.best_city = best_city
    st.session_state.best_fitness = best_fitness
    st.session_state.history = history
    st.session_state.optimization_complete = True

    progress_bar.progress(1.0)
    status_text.success("✅ GA Optimization complete!")
    time.sleep(1)
    st.rerun()

# ─── NSGA-II Run ──────────────────────────────────────────────────────────────
if run_optimization and mode == 'NSGA2':
    st.session_state.nsga2_complete = False

    progress_bar = st.progress(0)
    status_text = st.empty()
    front_container = st.empty()

    status_text.info("🔄 Initializing NSGA-II population…")

    nsga2 = NSGA2(
        grid_size=custom_grid_size,
        population_size=custom_pop_size,
        generations=custom_generations,
        min_population=min_population,
        max_budget=max_budget,
        mutation_rate=nsga2_mutation_rate,
        crossover_probability=nsga2_crossover_prob,
    )
    nsga2.initialize_population(seed=42)
    st.session_state.nsga2_initial_city = nsga2.population[0].city_grid.copy()

    # Monkey-patch run to get generation-by-generation progress in Streamlit
    status_text.success("🚀 Starting NSGA-II optimization…")

    for gen in range(custom_generations):
        offspring = nsga2._generate_offspring()
        for ind in offspring:
            ind.evaluate(min_population, max_budget)
        from src.optimization.nsga2 import non_dominated_sort, crowding_distance_assignment
        combined = nsga2.population + offspring
        fronts = non_dominated_sort(combined)
        for front in fronts:
            crowding_distance_assignment(front)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= nsga2.population_size:
                new_pop.extend(front)
            else:
                remaining = nsga2.population_size - len(new_pop)
                sorted_f = sorted(front, key=lambda x: -x.crowding_distance)
                new_pop.extend(sorted_f[:remaining])
                break
        nsga2.population = new_pop
        nsga2.pareto_front = fronts[0] if fronts else []
        nsga2._update_history(gen)

        # Progress
        progress_bar.progress(min((gen + 1) / custom_generations, 1.0))
        if gen % 10 == 0:
            viable = [i for i in nsga2.pareto_front if i.is_viable]
            n_v = len(viable)
            best_c = min((v.carbon for v in viable), default=float('nan'))
            best_h = max((v.happiness for v in viable), default=float('nan'))
            status_text.info(
                f"⚙️ Generation {gen}/{custom_generations} — "
                f"Pareto size: {len(nsga2.pareto_front)} ({n_v} feasible) — "
                f"Best carbon: {best_c:,.0f} — Best happiness: {best_h:.1f}"
            )

    st.session_state.nsga2_pareto = nsga2.pareto_front
    st.session_state.nsga2_history = nsga2.history
    st.session_state.nsga2_complete = True

    progress_bar.progress(1.0)
    status_text.success("✅ NSGA-II Optimization complete!")
    time.sleep(1)
    st.rerun()

# ─── GA Results ───────────────────────────────────────────────────────────────
if st.session_state.optimization_complete and mode == 'GA':
    best_city   = st.session_state.best_city
    best_fitness = st.session_state.best_fitness
    history     = st.session_state.history
    initial_city = st.session_state.initial_city

    if best_fitness and best_fitness.success_achieved:
        st.markdown("""
        <div class="success-box">
            <h2 style="color: #2E7D32; text-align: center;">
                ✅ SUCCESS! Net-Zero Carbon Target Achieved!
            </h2>
            <p style="text-align: center;">Carbon ratio: {:.2f}% (Target: &lt;5%)</p>
        </div>
        """.format(best_fitness.metrics.carbon_ratio * 100 if best_fitness.metrics else 0),
            unsafe_allow_html=True)
    elif best_fitness and not best_fitness.is_viable:
        st.error("⚠️ No viable solution found — all hard constraints were violated. "
                 "Try relaxing min population or max budget.")
    else:
        carbon_pct = best_fitness.metrics.carbon_ratio * 100 if best_fitness and best_fitness.metrics else 0
        st.warning(f"⚠️ Target not reached. Carbon ratio: {carbon_pct:.2f}% "
                   f"(Target: <5%). Try increasing generations or adjusting weights.")

    st.markdown("---")
    st.header("📊 GA Optimization Results")

    if best_fitness and best_fitness.metrics:
        m = best_fitness.metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Fitness Score", f"{best_fitness.fitness:.4f}")
        col2.metric("Net Carbon", f"{m.net_carbon:,.0f}", delta=f"{m.carbon_ratio*100:.2f}%")
        col3.metric("Population", f"{m.population:,}")
        col4.metric("Happiness", f"{m.happiness_score:.1f}/100")
        col5.metric("Total Cost", f"${m.total_cost:,.0f}")

        # Normalized breakdown
        st.caption(
            f"Normalized components — Carbon: {best_fitness.carbon_normalized:.3f}  "
            f"| Happiness: {best_fitness.happiness_normalized:.3f}  "
            f"| Cost: {best_fitness.cost_normalized:.3f}  "
            f"(each on 0–1 scale, weights applied on top)"
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ City Map", "📈 Evolution", "📋 Detailed Metrics", "💾 Export"])

    with tab1:
        st.subheader("Optimized City Layout")
        initial_constraint_results = evaluate_all_constraints(initial_city)
        initial_metrics = calculate_metrics(initial_city, initial_constraint_results)
        final_constraint_results = evaluate_all_constraints(best_city)
        final_metrics = calculate_metrics(best_city, final_constraint_results)

        fig, axes = create_before_after_comparison(initial_city, best_city,
                                                   initial_metrics, final_metrics)
        st.pyplot(fig); plt.close(fig)

        st.subheader("Detailed City Dashboard")
        fig, axes = create_metrics_dashboard(best_city, final_metrics, final_constraint_results)
        st.pyplot(fig); plt.close(fig)

        with st.expander("🌫️ View Pollution Analysis"):
            fig, axes = create_pollution_overlay(best_city)
            st.pyplot(fig); plt.close(fig)

    with tab2:
        st.subheader("Optimization Progress")

        def validate_history(h):
            validated = {}
            for key, values in h.items():
                if isinstance(values, list):
                    validated[key] = list(values)
                else:
                    validated[key] = values
            return validated

        vh = validate_history(history)
        if len(vh.get('generation', [])) > 0:
            st.info(f"Showing data for {len(vh['generation'])} generations")

        for plot_fn, label in [
            (plot_fitness_evolution, "fitness evolution"),
            (plot_carbon_reduction, "carbon reduction"),
            (plot_multi_metric_evolution, "multi-metric evolution"),
        ]:
            try:
                fig, _ = plot_fn(vh)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            except Exception as e:
                st.error(f"Error plotting {label}: {e}")

        try:
            if len(vh.get('generation', [])) > 0:
                fig, _ = create_optimization_summary(vh, initial_metrics, final_metrics)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting optimization summary: {e}")

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
        st.dataframe(pd.DataFrame(history))

    with tab4:
        st.subheader("Export Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Export Grid Layout")
            import json
            grid_csv = "\n".join(
                ",".join(str(int(v)) for v in row)
                for row in best_city.to_numpy()
            )
            st.download_button("Download City Grid (CSV)", grid_csv, "optimized_city.csv", "text/csv")
        with col2:
            st.markdown("### Export Metrics Report (JSON)")
            report = {
                'optimizer': 'GA',
                'initial_metrics': initial_metrics.to_dict(),
                'final_metrics': final_metrics.to_dict(),
                'configuration': {
                    'grid_size': custom_grid_size,
                    'population_size': custom_pop_size,
                    'generations': custom_generations,
                    'min_population': min_population,
                    'max_budget': max_budget,
                },
            }
            st.download_button("Download JSON", json.dumps(report, indent=2),
                               "ga_report.json", "application/json")

# ─── NSGA-II Results ──────────────────────────────────────────────────────────
if st.session_state.nsga2_complete and mode == 'NSGA2':
    pareto_front = st.session_state.nsga2_pareto or []
    nsga2_history = st.session_state.nsga2_history or {}
    initial_city = st.session_state.nsga2_initial_city

    viable_front = [ind for ind in pareto_front if ind.is_viable]

    st.markdown(f"""
    <div class="nsga2-box">
        <h2 style="color: #1976D2; text-align: center;">
            🔀 NSGA-II Complete — Pareto Front Ready
        </h2>
        <p style="text-align: center;">
            {len(pareto_front)} solutions on Pareto front
            ({len(viable_front)} feasible, satisfying all hard constraints)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("📊 NSGA-II Multi-Objective Results")

    if viable_front:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pareto Front Size", len(pareto_front))
        col2.metric("Feasible Solutions", len(viable_front))
        col3.metric("Best Carbon", f"{min(v.carbon for v in viable_front):,.0f}")
        col4.metric("Best Happiness", f"{max(v.happiness for v in viable_front):.1f}")
    else:
        st.error("No feasible Pareto solutions found — all individuals violated at least one "
                 "hard constraint. Try relaxing min population or max budget.")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Pareto Front", "📈 Convergence", "📋 Solution Table", "💾 Export"])

    with tab1:
        st.subheader("Pareto Front — Trade-off Visualisation")

        if viable_front:
            # ── Scatter matrix of the three objectives ──────────────────────
            carbons     = [v.carbon    for v in viable_front]
            happinesses = [v.happiness for v in viable_front]
            costs       = [v.cost      for v in viable_front]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle("NSGA-II Pareto Front — Feasible Solutions", fontsize=14, fontweight='bold')

            # Carbon vs Happiness
            sc0 = axes[0].scatter(carbons, happinesses, c=costs,
                                  cmap='RdYlGn_r', s=60, alpha=0.8, edgecolors='k', linewidths=0.5)
            axes[0].set_xlabel("Net Carbon (↓ better)")
            axes[0].set_ylabel("Happiness (↑ better)")
            axes[0].set_title("Carbon vs Happiness")
            plt.colorbar(sc0, ax=axes[0], label="Cost ($)")

            # Carbon vs Cost
            sc1 = axes[1].scatter(carbons, costs, c=happinesses,
                                  cmap='RdYlGn', s=60, alpha=0.8, edgecolors='k', linewidths=0.5)
            axes[1].set_xlabel("Net Carbon (↓ better)")
            axes[1].set_ylabel("Total Cost $ (↓ better)")
            axes[1].set_title("Carbon vs Cost")
            plt.colorbar(sc1, ax=axes[1], label="Happiness")

            # Happiness vs Cost
            sc2 = axes[2].scatter(happinesses, costs, c=carbons,
                                  cmap='RdYlGn_r', s=60, alpha=0.8, edgecolors='k', linewidths=0.5)
            axes[2].set_xlabel("Happiness (↑ better)")
            axes[2].set_ylabel("Total Cost $ (↓ better)")
            axes[2].set_title("Happiness vs Cost")
            plt.colorbar(sc2, ax=axes[2], label="Carbon")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Best-on-each-objective cards ────────────────────────────────
            st.subheader("Extreme Pareto Solutions")
            best_c    = min(viable_front, key=lambda x: x.carbon)
            best_h    = max(viable_front, key=lambda x: x.happiness)
            best_cost = min(viable_front, key=lambda x: x.cost)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### 🌿 Lowest Carbon")
                st.metric("Net Carbon", f"{best_c.carbon:,.0f}")
                st.metric("Happiness",  f"{best_c.happiness:.1f}/100")
                st.metric("Cost",       f"${best_c.cost:,.0f}")

            with c2:
                st.markdown("#### 😊 Highest Happiness")
                st.metric("Net Carbon", f"{best_h.carbon:,.0f}")
                st.metric("Happiness",  f"{best_h.happiness:.1f}/100")
                st.metric("Cost",       f"${best_h.cost:,.0f}")

            with c3:
                st.markdown("#### 💰 Lowest Cost")
                st.metric("Net Carbon", f"{best_cost.carbon:,.0f}")
                st.metric("Happiness",  f"{best_cost.happiness:.1f}/100")
                st.metric("Cost",       f"${best_cost.cost:,.0f}")

            # ── Interactive city map for selected Pareto member ──────────────
            st.subheader("🗺️ Inspect a Pareto Solution")
            st.caption("Pick an index from the Pareto front table (tab 3) to visualise it.")
            sel_idx = st.number_input("Pareto solution index", min_value=0,
                                      max_value=len(viable_front) - 1, value=0, step=1)
            sel_ind = viable_front[sel_idx]

            sel_constraint_results = evaluate_all_constraints(sel_ind.city_grid)
            sel_metrics = calculate_metrics(sel_ind.city_grid, sel_constraint_results)

            if initial_city is not None:
                init_cr = evaluate_all_constraints(initial_city)
                init_m  = calculate_metrics(initial_city, init_cr)
                fig, axes = create_before_after_comparison(
                    initial_city, sel_ind.city_grid, init_m, sel_metrics)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig, axes = create_metrics_dashboard(sel_ind.city_grid, sel_metrics,
                                                     sel_constraint_results)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        else:
            st.warning("No feasible Pareto solutions to display.")

    with tab2:
        st.subheader("NSGA-II Convergence")

        if nsga2_history.get('generation'):
            gen_vals = nsga2_history['generation']

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("NSGA-II Convergence History", fontsize=14, fontweight='bold')

            axes[0, 0].plot(gen_vals, nsga2_history.get('pareto_size', []), color='steelblue')
            axes[0, 0].set_title("Pareto Front Size")
            axes[0, 0].set_xlabel("Generation")
            axes[0, 0].set_ylabel("# Solutions")

            axes[0, 1].plot(gen_vals, nsga2_history.get('viable_count', []), color='green')
            axes[0, 1].set_title("Feasible Solutions on Front")
            axes[0, 1].set_xlabel("Generation")
            axes[0, 1].set_ylabel("# Feasible")

            best_carbons = nsga2_history.get('best_carbon', [])
            if any(not math.isnan(v) for v in best_carbons):
                axes[1, 0].plot(gen_vals, best_carbons, color='red')
                axes[1, 0].set_title("Best Carbon (Pareto front min)")
                axes[1, 0].set_xlabel("Generation")
                axes[1, 0].set_ylabel("Net Carbon")

            best_hap = nsga2_history.get('best_happiness', [])
            if any(not math.isnan(v) for v in best_hap):
                axes[1, 1].plot(gen_vals, best_hap, color='orange')
                axes[1, 1].set_title("Best Happiness (Pareto front max)")
                axes[1, 1].set_xlabel("Generation")
                axes[1, 1].set_ylabel("Happiness /100")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No convergence history available.")

    with tab3:
        st.subheader("Pareto Front — Full Solution Table")

        if viable_front:
            rows = [
                {
                    "Index": i,
                    "Carbon":    f"{ind.carbon:,.0f}",
                    "Happiness": f"{ind.happiness:.1f}",
                    "Cost ($)":  f"{ind.cost:,.0f}",
                    "Population": f"{ind.population_count:,}",
                    "Energy Balance": f"{ind.energy_balance:+,.0f}",
                    "Viable": "✓" if ind.is_viable else "✗",
                }
                for i, ind in enumerate(viable_front)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No feasible Pareto solutions.")

        if pareto_front and not viable_front:
            st.markdown("#### Infeasible Pareto Members")
            rows_inf = [
                {
                    "Index": i,
                    "Carbon":    f"{ind.carbon:,.0f}",
                    "Happiness": f"{ind.happiness:.1f}",
                    "Cost ($)":  f"{ind.cost:,.0f}",
                    "Population": f"{ind.population_count:,}",
                    "Energy Balance": f"{ind.energy_balance:+,.0f}",
                }
                for i, ind in enumerate(pareto_front)
            ]
            st.dataframe(pd.DataFrame(rows_inf), use_container_width=True)

    with tab4:
        st.subheader("Export NSGA-II Results")

        if viable_front:
            import json

            pareto_data = [
                {
                    "index":          i,
                    "carbon":         ind.carbon,
                    "happiness":      ind.happiness,
                    "cost":           ind.cost,
                    "population":     ind.population_count,
                    "energy_balance": ind.energy_balance,
                    "is_viable":      ind.is_viable,
                }
                for i, ind in enumerate(viable_front)
            ]
            report = {
                'optimizer': 'NSGA-II',
                'pareto_front': pareto_data,
                'configuration': {
                    'grid_size': custom_grid_size,
                    'population_size': custom_pop_size,
                    'generations': custom_generations,
                    'min_population': min_population,
                    'max_budget': max_budget,
                    'mutation_rate': nsga2_mutation_rate,
                    'crossover_probability': nsga2_crossover_prob,
                },
            }
            st.download_button("Download Pareto Front (JSON)",
                               json.dumps(report, indent=2),
                               "nsga2_pareto_report.json", "application/json")

            # CSV of the Pareto table
            pareto_df = pd.DataFrame(pareto_data)
            st.download_button("Download Pareto Front (CSV)",
                               pareto_df.to_csv(index=False),
                               "nsga2_pareto_front.csv", "text/csv")
        else:
            st.info("No feasible Pareto solutions to export.")

# ─── Sidebar reset / compare buttons ──────────────────────────────────────────
if st.session_state.optimization_complete or st.session_state.nsga2_complete:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start New Optimization"):
        for key in ['optimization_complete', 'best_city', 'initial_city', 'history',
                    'best_fitness', 'nsga2_complete', 'nsga2_pareto',
                    'nsga2_history', 'nsga2_initial_city']:
            st.session_state[key] = None if key not in (
                'optimization_complete', 'nsga2_complete') else False
        st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Zero-Carbon City Simulation</strong> - Research Prototype</p>
    <p>GA (weighted-sum, normalized) · NSGA-II (Pareto multi-objective)</p>
    <p>Built with Python, PyTorch, Matplotlib, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
