"""
Array Backend Utility - Hardware Agnostic NumPy/CuPy Switching

This module provides a unified interface for array operations that can
transparently use either NumPy (CPU) or CuPy (GPU) based on availability
and configuration.

This allows the codebase to be hardware-agnostic while still benefiting
from GPU acceleration when available.
"""

import sys

# Try to import CuPy, fall back to NumPy
USE_GPU = False
try:
    import cupy as cp
    USE_GPU = True
    xp = cp
    print("GPU acceleration enabled (CuPy detected)", file=sys.stderr)
except ImportError:
    import numpy as np
    xp = np
    print("Using CPU computation (NumPy). Install CuPy for GPU acceleration.", file=sys.stderr)

# Export the array module
__all__ = ['xp', 'USE_GPU', 'to_numpy', 'to_gpu', 'get_array_module']


def to_numpy(array):
    """
    Convert array to NumPy array (from GPU to CPU if needed).
    
    Args:
        array: NumPy or CuPy array
        
    Returns:
        NumPy array
    """
    if USE_GPU and hasattr(array, 'get'):
        return array.get()  # CuPy to NumPy
    return array


def to_gpu(array):
    """
    Convert array to GPU array if GPU is available.
    
    Args:
        array: NumPy array
        
    Returns:
        CuPy array if GPU available, otherwise NumPy array
    """
    if USE_GPU:
        return cp.asarray(array)
    return array


def get_array_module(array):
    """
    Get the appropriate array module for a given array.
    
    Args:
        array: NumPy or CuPy array
        
    Returns:
        numpy or cupy module
    """
    if USE_GPU:
        return cp.get_array_module(array)
    return np


def set_gpu_enabled(enabled):
    """
    Enable or disable GPU usage (requires restart of modules).
    
    Args:
        enabled (bool): Whether to use GPU
    """
    global USE_GPU, xp
    
    if enabled and not USE_GPU:
        try:
            import cupy as cp
            USE_GPU = True
            xp = cp
            print("GPU acceleration enabled", file=sys.stderr)
        except ImportError:
            print("CuPy not available. Cannot enable GPU.", file=sys.stderr)
    elif not enabled and USE_GPU:
        import numpy as np
        USE_GPU = False
        xp = np
        print("GPU acceleration disabled, using CPU", file=sys.stderr)
