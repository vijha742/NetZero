"""
Building Type Configuration for Zero-Carbon City Simulation

This configuration defines the characteristics of different building types
used in the urban planning simulation. Each building has attributes that
affect carbon emissions, energy production/consumption, cost, and population.

Note: Values are synthetic and designed to represent archetypes for research purposes.
Real-world emission factors vary significantly by region, building standards, and energy sources.

References:
- Residential carbon: EPA Household Carbon Footprint (5-15 tons CO2/year per household)
- Industrial carbon: Varies widely (100-10,000 tons CO2/year) based on industry type
- Power generation: IPCC emission factors (~1000 kg CO2/MWh for coal, ~50 for solar)
"""

# Building Type IDs and their characteristics
BUILDING_TYPES = {
    0: {
        "name": "Empty Land",
        "short_name": "Empty",
        "carbon": 0,           # No emissions
        "cost": 0,             # Free
        "pop": 0,              # No population
        "jobs": 0,             # No jobs
        "energy_demand": 0,    # No energy consumption
        "energy_supply": 0,    # No energy production
        "color": "#F5F5DC",    # Beige for visualization
        "category": "empty"
    },
    1: {
        "name": "Residential High Density",
        "short_name": "Res_HD",
        "carbon": 50,          # Higher emissions per unit (older buildings, more residents)
        "cost": 500,           # Lower construction cost
        "pop": 200,            # High population density
        "jobs": 0,
        "energy_demand": 100,  # High energy consumption
        "energy_supply": 0,
        "color": "#4A90E2",    # Blue
        "category": "residential"
    },
    2: {
        "name": "Residential Eco Housing",
        "short_name": "Res_Eco",
        "carbon": 10,          # Low emissions (energy-efficient, passive house standards)
        "cost": 800,           # Higher construction cost (green building materials)
        "pop": 100,            # Lower density (more space per resident)
        "jobs": 0,
        "energy_demand": 30,   # Much lower energy demand (efficient insulation, appliances)
        "energy_supply": 0,
        "color": "#7ED321",    # Light green
        "category": "residential"
    },
    3: {
        "name": "Industrial Factory",
        "short_name": "Factory",
        "carbon": 1000,        # High emissions (manufacturing processes)
        "cost": 2000,          # High construction cost
        "pop": 0,
        "jobs": 50,            # Creates jobs
        "energy_demand": 500,  # Very high energy consumption
        "energy_supply": 0,
        "color": "#8B8B8B",    # Gray
        "category": "industrial"
    },
    4: {
        "name": "Coal Power Plant",
        "short_name": "Coal",
        "carbon": 5000,        # Extremely high emissions
        "cost": 1500,          # Moderate cost (mature technology)
        "pop": 0,
        "jobs": 20,            # Some jobs
        "energy_demand": 0,
        "energy_supply": 5000, # High energy production
        "color": "#2C2C2C",    # Dark gray/black
        "category": "energy"
    },
    5: {
        "name": "Solar Power Farm",
        "short_name": "Solar",
        "carbon": 50,          # Low emissions (manufacturing and maintenance)
        "cost": 3000,          # High upfront cost
        "pop": 0,
        "jobs": 10,            # Fewer jobs than coal
        "energy_demand": 0,
        "energy_supply": 1000, # Lower energy output (intermittent)
        "color": "#F8E71C",    # Yellow
        "category": "energy"
    },
    6: {
        "name": "Forest/Green Space",
        "short_name": "Forest",
        "carbon": -200,        # Negative carbon (carbon sink)
        "cost": 100,           # Low cost (land maintenance)
        "pop": 0,
        "jobs": 5,             # Park maintenance jobs
        "energy_demand": 0,
        "energy_supply": 0,
        "color": "#228B22",    # Forest green
        "category": "green"
    },
    7: {
        "name": "Road Network",
        "short_name": "Road",
        "carbon": 5,           # Low emissions (construction and maintenance)
        "cost": 50,            # Relatively inexpensive
        "pop": 0,
        "jobs": 0,
        "energy_demand": 10,   # Street lighting
        "energy_supply": 0,
        "color": "#696969",    # Dim gray
        "category": "infrastructure"
    },
}

# Building categories for easy filtering
CATEGORIES = {
    "residential": [1, 2],
    "industrial": [3],
    "energy": [4, 5],
    "green": [6],
    "infrastructure": [7],
    "empty": [0]
}

# Building type counts (for initialization purposes)
NUM_BUILDING_TYPES = len(BUILDING_TYPES)

# Extensibility: Helper functions for adding new building types
def add_building_type(building_id, properties):
    """
    Add a new building type to the configuration.
    
    Args:
        building_id (int): Unique identifier for the building type
        properties (dict): Dictionary containing building attributes
    """
    required_keys = ["name", "short_name", "carbon", "cost", "pop", "jobs", 
                     "energy_demand", "energy_supply", "color", "category"]
    
    # Validate all required keys are present
    for key in required_keys:
        if key not in properties:
            raise ValueError(f"Missing required property '{key}' for building type {building_id}")
    
    BUILDING_TYPES[building_id] = properties
    
    # Update category mapping
    category = properties["category"]
    if category in CATEGORIES:
        if building_id not in CATEGORIES[category]:
            CATEGORIES[category].append(building_id)
    else:
        CATEGORIES[category] = [building_id]

def get_building_by_category(category):
    """Get all building type IDs in a specific category."""
    return CATEGORIES.get(category, [])

def get_building_info(building_id):
    """Get information about a specific building type."""
    return BUILDING_TYPES.get(building_id, None)

# Data gap documentation
DATA_GAPS = {
    "residential": "Real-world residential carbon emissions vary by region (climate), "
                   "building age, energy source mix, and household size. Standardized databases are lacking.",
    "industrial": "Industrial emissions highly depend on industry type (steel, chemicals, electronics). "
                  "Limited publicly available, standardized emission data per facility type.",
    "energy": "Power plant emissions depend on fuel type, efficiency, capacity factor. "
              "Grid mix complexity not captured in this model.",
    "spatial": "Real-world urban heat island effects, pollution dispersion models require "
               "detailed meteorological and topographical data.",
}

# Research notes for conference paper
RESEARCH_NOTES = """
Building Configuration Design Choices:

1. **Synthetic Dataset Rationale**: 
   Due to lack of standardized, open-access carbon emission databases for urban buildings,
   we created archetypes representing common building categories. Values are calibrated to
   reflect relative emission intensities rather than absolute measurements.

2. **Simplifying Assumptions**:
   - All buildings of the same type have identical characteristics (no variance)
   - Carbon emissions are static (no temporal variation, weather effects)
   - Energy demand is constant (no peak/off-peak modeling)
   - Construction costs are one-time (no operational costs)

3. **Future Work**:
   - Integration with EPA GHGRP (Greenhouse Gas Reporting Program) data
   - Building-specific emission factors from IPCC guidelines
   - Regional climate adjustment factors (heating/cooling degree days)
   - Life cycle assessment (LCA) data for construction materials

4. **Validation Approach**:
   Relative comparisons (high-density vs eco-housing) are more reliable than
   absolute carbon values. System demonstrates proof-of-concept for optimization
   methodology rather than precise emission accounting.
"""

if __name__ == "__main__":
    # Print building configuration summary
    print("=" * 70)
    print("BUILDING TYPE CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal Building Types: {NUM_BUILDING_TYPES}\n")
    
    for building_id, props in BUILDING_TYPES.items():
        print(f"[{building_id}] {props['name']}")
        print(f"    Carbon: {props['carbon']:>6} | Cost: {props['cost']:>5} | "
              f"Pop: {props['pop']:>3} | Jobs: {props['jobs']:>2}")
        print(f"    Energy: Demand={props['energy_demand']:>4}, Supply={props['energy_supply']:>4}")
        print(f"    Category: {props['category']}")
        print()
    
    print("=" * 70)
    print("CATEGORIES")
    print("=" * 70)
    for category, building_ids in CATEGORIES.items():
        print(f"{category:15s}: {building_ids}")
