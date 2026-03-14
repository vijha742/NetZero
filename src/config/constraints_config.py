"""
Spatial Constraints Configuration for Zero-Carbon City Simulation

This module defines spatial constraints that represent real-world urban planning
principles and environmental effects. These constraints affect the happiness score
and carbon emissions based on the spatial arrangement of buildings.

The 8 comprehensive spatial constraints implemented:
1. NIMBY (Not In My Backyard) - Residential unhappiness near factories
2. Heat Island Effect - Increased cooling carbon for houses without nearby forests
3. Transit Accessibility - Residential areas need road access
4. Industrial Clustering - Efficiency bonuses for factory clusters
5. Green Space Requirements - Minimum forest coverage per residential area
6. Energy Transmission Loss - Distance-based power plant effectiveness
7. Pollution Dispersion - Factories create pollution zones affecting residential
8. Zoning Compatibility - Certain building combinations prohibited

References:
- Urban Heat Island Effect: EPA studies show 1-7°F temperature increase in dense urban areas
- NIMBY Effect: Documented property value decreases (10-30%) near industrial facilities
- Green Space Standards: WHO recommends 9m² of green space per capita
- Energy Transmission: Typical distribution losses are 5-10% over medium distances
"""

# ============================================================================
# CONSTRAINT 1: NIMBY (Not In My Backyard)
# ============================================================================
NIMBY = {
    "name": "NIMBY (Not In My Backyard)",
    "description": "Residential happiness drops when near industrial facilities",
    "enabled": True,
    "residential_types": [1, 2],  # Res_HD, Res_Eco
    "industrial_types": [3],      # Factory
    "distance_threshold": 3,      # cells
    "happiness_penalty": 0.30,    # 30% happiness reduction
    "penalty_score": -50,         # Fitness penalty per violation
}

# ============================================================================
# CONSTRAINT 2: Heat Island Effect
# ============================================================================
HEAT_ISLAND = {
    "name": "Heat Island Effect",
    "description": "Houses without nearby forests have increased cooling carbon",
    "enabled": True,
    "residential_types": [1, 2],
    "forest_types": [6],
    "distance_threshold": 5,      # cells
    "carbon_increase": 0.10,      # 10% increase in carbon emissions
    "penalty_score": -20,         # Fitness penalty per affected house
}

# ============================================================================
# CONSTRAINT 3: Transit Accessibility
# ============================================================================
TRANSIT_ACCESS = {
    "name": "Transit Accessibility",
    "description": "Residential areas must be accessible via road network",
    "enabled": True,
    "residential_types": [1, 2],
    "road_types": [7],
    "distance_threshold": 8,      # cells
    "happiness_penalty": 0.40,    # 40% happiness reduction if isolated
    "penalty_score": -100,        # High penalty for isolated residential
}

# ============================================================================
# CONSTRAINT 4: Industrial Clustering Bonus
# ============================================================================
INDUSTRIAL_CLUSTER = {
    "name": "Industrial Clustering Bonus",
    "description": "Factories near each other gain efficiency (supply chain benefits)",
    "enabled": True,
    "industrial_types": [3],
    "distance_threshold": 4,      # cells
    "carbon_reduction": 0.05,     # 5% carbon reduction per nearby factory
    "bonus_score": 25,            # Fitness bonus per cluster pair
    "max_bonus_per_factory": 3,   # Maximum number of neighbors to count
}

# ============================================================================
# CONSTRAINT 5: Green Space Requirements
# ============================================================================
GREEN_SPACE_REQ = {
    "name": "Green Space Requirements",
    "description": "Adequate forest coverage required per residential zone",
    "enabled": True,
    "residential_types": [1, 2],
    "forest_types": [6],
    "ratio": 0.1,                 # 1 forest cell per 10 residential cells
    "search_radius": 6,           # cells
    "penalty_score": -150,        # Penalty per shortage
}

# ============================================================================
# CONSTRAINT 6: Energy Transmission Loss
# ============================================================================
ENERGY_TRANSMISSION = {
    "name": "Energy Transmission Loss",
    "description": "Power plants have distance-based effectiveness",
    "enabled": True,
    "power_plant_types": [4, 5],  # Coal, Solar
    "distance_ranges": {
        "full": 10,     # Full energy within 10 cells
        "half": 20,     # 50% energy within 20 cells
        "none": 21,     # No energy beyond 20 cells
    },
    "efficiency_full": 1.0,
    "efficiency_half": 0.5,
    "efficiency_none": 0.0,
    "penalty_score": -200,        # Penalty per energy shortage event
}

# ============================================================================
# CONSTRAINT 7: Pollution Dispersion
# ============================================================================
POLLUTION_DISPERSION = {
    "name": "Pollution Dispersion",
    "description": "Factories create pollution zones affecting residential health/happiness",
    "enabled": True,
    "pollution_sources": [3, 4],  # Factory, Coal plant
    "residential_types": [1, 2],
    "pollution_radius": 5,        # cells
    "happiness_penalty": 0.20,    # 20% happiness reduction
    "penalty_score": -30,         # Penalty per affected residential cell
}

# ============================================================================
# CONSTRAINT 8: Zoning Compatibility
# ============================================================================
ZONING_COMPAT = {
    "name": "Zoning Compatibility",
    "description": "Certain building combinations are prohibited (severe violations)",
    "enabled": True,
    "incompatible_pairs": [
        (1, 4, 8),  # Res_HD cannot be within 8 cells of Coal plant
        (2, 4, 8),  # Res_Eco cannot be within 8 cells of Coal plant
        (3, 6, 2),  # Factory cannot be within 2 cells of Forest (deforestation)
    ],  # Format: (type1, type2, min_distance)
    "penalty_score": -500,        # Severe penalty per violation
}

# ============================================================================
# Consolidated Constraints Dictionary
# ============================================================================
CONSTRAINTS = {
    "nimby": NIMBY,
    "heat_island": HEAT_ISLAND,
    "transit_access": TRANSIT_ACCESS,
    "industrial_cluster": INDUSTRIAL_CLUSTER,
    "green_space": GREEN_SPACE_REQ,
    "energy_transmission": ENERGY_TRANSMISSION,
    "pollution_dispersion": POLLUTION_DISPERSION,
    "zoning_compatibility": ZONING_COMPAT,
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_enabled_constraints():
    """Return list of enabled constraint names."""
    return [name for name, config in CONSTRAINTS.items() if config["enabled"]]

def enable_constraint(constraint_name):
    """Enable a specific constraint."""
    if constraint_name in CONSTRAINTS:
        CONSTRAINTS[constraint_name]["enabled"] = True
    else:
        raise ValueError(f"Unknown constraint: {constraint_name}")

def disable_constraint(constraint_name):
    """Disable a specific constraint."""
    if constraint_name in CONSTRAINTS:
        CONSTRAINTS[constraint_name]["enabled"] = False
    else:
        raise ValueError(f"Unknown constraint: {constraint_name}")

def get_constraint_config(constraint_name):
    """Get configuration for a specific constraint."""
    return CONSTRAINTS.get(constraint_name, None)

# ============================================================================
# Research Notes
# ============================================================================
CONSTRAINT_RESEARCH_NOTES = """
Spatial Constraint Design Rationale:

1. **NIMBY Effect**: 
   Based on documented property value decreases near industrial facilities.
   Real-world studies show 10-30% value drops, we use 30% happiness penalty.

2. **Heat Island Effect**:
   EPA studies document 1-7°F temperature increases in dense urban areas.
   Urban forests reduce local temperatures by 2-9°F. We model this as
   10% carbon increase for cooling when forests are absent.

3. **Transit Accessibility**:
   Essential for livability. Studies show car-dependent suburbs have
   higher per-capita carbon emissions. Strong penalty for isolation.

4. **Industrial Clustering**:
   Supply chain economics: Clusters reduce transportation, shared infrastructure.
   Efficiency gains modeled as 5% carbon reduction per proximate factory.

5. **Green Space Requirements**:
   WHO recommends minimum 9m² green space per capita. We simplify to
   1 forest cell per 10 residential cells as proxy metric.

6. **Energy Transmission Loss**:
   Simplified model of electrical grid distribution losses. Real-world
   transmission losses are 5-10% but highly dependent on voltage levels
   and infrastructure quality.

7. **Pollution Dispersion**:
   Simplified point-source pollution model. Real-world air quality modeling
   requires meteorological data (wind patterns, atmospheric stability).
   We use distance-based proxy.

8. **Zoning Compatibility**:
   Reflects real-world zoning laws preventing incompatible land uses.
   Coal plants near residential are strictly regulated in most jurisdictions.

Limitations:
- No temporal dynamics (wind direction changes, seasonal variations)
- No topographical effects (hills, water bodies affecting pollution/climate)
- Simplified distance metrics (Euclidean vs network distances)
- Binary effects (real-world effects are continuous gradients)

Future Enhancements:
- Integration with air quality models (AERMOD, CALPUFF)
- Weather-dependent constraints
- Noise pollution modeling
- Water resource constraints
- Traffic congestion modeling
"""

if __name__ == "__main__":
    # Print constraints summary
    print("=" * 80)
    print("SPATIAL CONSTRAINTS CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal Constraints Defined: {len(CONSTRAINTS)}\n")
    
    for name, config in CONSTRAINTS.items():
        status = "ENABLED" if config["enabled"] else "DISABLED"
        print(f"[{status}] {config['name']}")
        print(f"    Description: {config['description']}")
        print(f"    Penalty Score: {config.get('penalty_score', 'N/A')}")
        print()
    
    print("=" * 80)
    print(f"Active Constraints: {len(get_enabled_constraints())}/{len(CONSTRAINTS)}")
    print("=" * 80)
