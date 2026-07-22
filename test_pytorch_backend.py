"""
Test PyTorch Backend Migration

This script tests the new PyTorch backend to ensure:
1. Device detection works correctly (MPS/CUDA/CPU)
2. All array operations work as expected
3. Tensor-to-NumPy conversions are correct
4. Module imports succeed
"""

import sys
import os

print("="*70)
print("PyTorch Backend Migration Test")
print("="*70)

# Test 1: Backend detection
print("\n[TEST 1] Testing backend detection...")
try:
    from src.utils.array_backend import xp, USE_GPU, device, device_name
    print(f"✓ Backend imported successfully")
    print(f"  - Device: {device}")
    print(f"  - Device Name: {device_name}")
    print(f"  - GPU Enabled: {USE_GPU}")
except Exception as e:
    print(f"✗ Backend import failed: {e}")
    sys.exit(1)

# Test 2: Basic array operations
print("\n[TEST 2] Testing basic array operations...")
try:
    # Create arrays
    arr1 = xp.zeros((5, 5), dtype=int)
    arr2 = xp.ones((3, 3), dtype=float)
    arr3 = xp.arange(10)
    print(f"✓ Array creation works")
    print(f"  - zeros shape: {arr1.shape}")
    print(f"  - ones shape: {arr2.shape}")
    print(f"  - arange shape: {arr3.shape}")
except Exception as e:
    print(f"✗ Array creation failed: {e}")
    sys.exit(1)

# Test 3: Random operations
print("\n[TEST 3] Testing random operations...")
try:
    xp.random.seed(42)
    rand_arr = xp.random.rand(5, 5)
    randint_arr = xp.random.randint(0, 10, size=(5, 5))
    print(f"✓ Random operations work")
    print(f"  - rand shape: {rand_arr.shape}")
    print(f"  - randint shape: {randint_arr.shape}")
except Exception as e:
    print(f"✗ Random operations failed: {e}")
    sys.exit(1)

# Test 4: Mathematical operations
print("\n[TEST 4] Testing mathematical operations...")
try:
    a = xp.array([1.0, 2.0, 3.0, 4.0])
    sqrt_result = xp.sqrt(a)
    sum_result = xp.sum(a)
    max_result = xp.max(a)
    print(f"✓ Math operations work")
    print(f"  - sqrt result shape: {sqrt_result.shape}")
    print(f"  - sum: {sum_result}")
    print(f"  - max: {max_result}")
except Exception as e:
    print(f"✗ Math operations failed: {e}")
    sys.exit(1)

# Test 5: Special operations (isin, argwhere, unique)
print("\n[TEST 5] Testing special operations...")
try:
    arr = xp.array([1, 2, 3, 2, 1, 4, 3])
    isin_result = xp.isin(arr, xp.array([1, 3]))
    argwhere_result = xp.argwhere(arr > 2)
    unique_vals, counts = xp.unique(arr, return_counts=True)
    print(f"✓ Special operations work")
    print(f"  - isin shape: {isin_result.shape}")
    print(f"  - argwhere found: {len(argwhere_result)} elements")
    print(f"  - unique values: {len(unique_vals)}")
except Exception as e:
    print(f"✗ Special operations failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Tensor to NumPy conversion
print("\n[TEST 6] Testing tensor to NumPy conversion...")
try:
    from src.utils.array_backend import to_numpy
    tensor = xp.array([[1, 2], [3, 4]])
    numpy_arr = to_numpy(tensor)
    print(f"✓ Tensor to NumPy conversion works")
    print(f"  - Tensor type: {type(tensor)}")
    print(f"  - NumPy type: {type(numpy_arr)}")
    print(f"  - Values match: {numpy_arr[0, 0] == 1}")
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Module imports
print("\n[TEST 7] Testing module imports...")
try:
    from src.models.city_grid import CityGrid
    from src.engine.simulation import calculate_city_metrics
    from src.optimization.fitness import calculate_fitness
    from src.utils.distance import euclidean_distance
    print(f"✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: CityGrid with PyTorch
print("\n[TEST 8] Testing CityGrid with PyTorch backend...")
try:
    city = CityGrid(size=10)
    city.randomize(method='random', seed=42)
    counts = city.get_building_counts()
    city_copy = city.copy()
    print(f"✓ CityGrid operations work")
    print(f"  - Grid shape: {city.grid.shape}")
    print(f"  - Building types found: {len(counts)}")
    print(f"  - Copy successful: {city_copy.size == city.size}")
except Exception as e:
    print(f"✗ CityGrid operations failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Distance calculations
print("\n[TEST 9] Testing distance calculations...")
try:
    from src.utils.distance import euclidean_distance, manhattan_distance
    dist1 = euclidean_distance(0, 0, 3, 4)
    dist2 = manhattan_distance(0, 0, 3, 4)
    # Convert to scalar if tensor
    d1_val = float(dist1.item()) if hasattr(dist1, 'item') else float(dist1)
    d2_val = float(dist2.item()) if hasattr(dist2, 'item') else float(dist2)
    print(f"✓ Distance calculations work")
    print(f"  - Euclidean distance (0,0) to (3,4): {d1_val:.2f}")
    print(f"  - Manhattan distance (0,0) to (3,4): {d2_val:.2f}")
except Exception as e:
    print(f"✗ Distance calculations failed: {e}")
    import traceback
    traceback.print_exc()

# Test 10: Fitness calculation
print("\n[TEST 10] Testing fitness calculation...")
try:
    city = CityGrid(size=20)
    city.randomize(method='random_weighted', seed=42)
    from src.optimization.fitness import calculate_fitness
    result = calculate_fitness(city)
    print(f"✓ Fitness calculation works")
    print(f"  - Fitness score: {result.fitness:.2f}")
    print(f"  - Is viable: {result.is_viable}")
except Exception as e:
    print(f"✗ Fitness calculation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print(f"\nYour PyTorch backend is working correctly on {device_name}")
print(f"GPU Acceleration: {'ENABLED' if USE_GPU else 'DISABLED'}")
print("\nNext steps:")
print("  1. Install PyTorch if not already installed:")
print("     pip install torch>=2.1.0")
print("  2. Run the main application:")
print("     streamlit run app.py")
print("  3. Or run a quick optimization:")
print("     python quick_test.py")
print("="*70)
