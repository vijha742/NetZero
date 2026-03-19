"""
Array Backend Utility - Hardware Agnostic PyTorch Backend

This module provides a unified interface for array operations that can
transparently use PyTorch with GPU acceleration (Apple Silicon MPS, NVIDIA CUDA)
or fall back to CPU-only computation.

Migrated from NumPy/CuPy to PyTorch for Apple Silicon M4 support.
Uses float32 for optimal GPU performance.
"""

import sys
import torch

# Verbose logging flag
VERBOSE_LOGGING = True

def log(message):
    """Log message if verbose logging is enabled."""
    if VERBOSE_LOGGING:
        print(f"[ArrayBackend] {message}", file=sys.stderr)

# Device detection with priority: MPS > CUDA > CPU
device = None
USE_GPU = False
device_name = "unknown"

if torch.backends.mps.is_available():
    device = torch.device("mps")
    USE_GPU = True
    device_name = "Apple Silicon MPS"
    log(f"✓ GPU acceleration enabled: {device_name}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    USE_GPU = True
    device_name = f"NVIDIA CUDA (GPU: {torch.cuda.get_device_name(0)})"
    log(f"✓ GPU acceleration enabled: {device_name}")
else:
    device = torch.device("cpu")
    USE_GPU = False
    device_name = "CPU-only"
    log(f"⚠ Using CPU computation. Install PyTorch with GPU support for acceleration.")

log(f"Device: {device}, GPU Available: {USE_GPU}")

# Default data types (float32 for performance)
DEFAULT_FLOAT_DTYPE = torch.float32
DEFAULT_INT_DTYPE = torch.int64


class TorchArrayModule:
    """
    Wrapper class to provide NumPy-like interface for PyTorch tensors.
    This allows minimal code changes in the rest of the codebase.
    """
    
    def __init__(self, device):
        self.device = device
        self.random = self.RandomModule(device)
    
    class RandomModule:
        """Handles random number generation."""
        def __init__(self, device):
            self.device = device
        
        def seed(self, seed):
            """Set random seed for reproducibility."""
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        def rand(self, *shape, **kwargs):
            """Generate random floats in [0, 1)."""
            return torch.rand(*shape, device=self.device, dtype=DEFAULT_FLOAT_DTYPE)
        
        def randint(self, low, high=None, size=None, **kwargs):
            """Generate random integers."""
            if high is None:
                high = low
                low = 0
            if size is None:
                return torch.randint(low, high, (1,), device=self.device, dtype=DEFAULT_INT_DTYPE).item()
            return torch.randint(low, high, size, device=self.device, dtype=DEFAULT_INT_DTYPE)
    
    def array(self, data, dtype=None):
        """Convert data to tensor."""
        if dtype is None:
            # Auto-detect: if data contains floats, use float32, else int64
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            # Try to infer type
            try:
                tensor = torch.tensor(data, device=self.device)
                if tensor.dtype in [torch.float64, torch.float32, torch.float16]:
                    return tensor.to(dtype=DEFAULT_FLOAT_DTYPE)
                return tensor
            except:
                return torch.tensor(data, device=self.device)
        else:
            if dtype == int:
                dtype = DEFAULT_INT_DTYPE
            elif dtype == float:
                dtype = DEFAULT_FLOAT_DTYPE
            return torch.tensor(data, dtype=dtype, device=self.device)
    
    def zeros(self, shape, dtype=None):
        """Create tensor of zeros."""
        if dtype is None:
            dtype = DEFAULT_FLOAT_DTYPE
        elif dtype == int:
            dtype = DEFAULT_INT_DTYPE
        elif dtype == float:
            dtype = DEFAULT_FLOAT_DTYPE
        return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def ones(self, shape, dtype=None):
        """Create tensor of ones."""
        if dtype is None:
            dtype = DEFAULT_FLOAT_DTYPE
        elif dtype == int:
            dtype = DEFAULT_INT_DTYPE
        elif dtype == float:
            dtype = DEFAULT_FLOAT_DTYPE
        return torch.ones(shape, dtype=dtype, device=self.device)
    
    def arange(self, *args, **kwargs):
        """Create range of values."""
        return torch.arange(*args, device=self.device, **kwargs)
    
    def meshgrid(self, *tensors, indexing='xy'):
        """Create coordinate matrices from coordinate vectors."""
        return torch.meshgrid(*tensors, indexing=indexing)
    
    def ogrid(self, *slices):
        """
        Open meshgrid - creates sparse coordinate arrays.
        Note: This is a simplified version for the most common use case.
        """
        # For now, return a simple implementation
        # Most common usage: ogrid[:n, :m] which returns (n,1) and (1,m) arrays
        if len(slices) == 2:
            slice1, slice2 = slices
            if isinstance(slice1, slice) and isinstance(slice2, slice):
                n = slice1.stop if slice1.stop else 0
                m = slice2.stop if slice2.stop else 0
                return (
                    torch.arange(n, device=self.device).reshape(-1, 1),
                    torch.arange(m, device=self.device).reshape(1, -1)
                )
        raise NotImplementedError("ogrid: Only 2D [:n, :m] slicing supported")
    
    def sqrt(self, x):
        """Square root."""
        return torch.sqrt(x.to(dtype=DEFAULT_FLOAT_DTYPE))
    
    def abs(self, x):
        """Absolute value."""
        return torch.abs(x)
    
    def exp(self, x):
        """Exponential."""
        return torch.exp(x.to(dtype=DEFAULT_FLOAT_DTYPE))
    
    def sum(self, x, *args, **kwargs):
        """Sum of array elements."""
        return torch.sum(x, *args, **kwargs)
    
    def max(self, x, *args, **kwargs):
        """Maximum value."""
        if len(args) == 0 and len(kwargs) == 0:
            return torch.max(x)
        return torch.max(x, *args, **kwargs)
    
    def min(self, x, *args, **kwargs):
        """Minimum value."""
        if len(args) == 0 and len(kwargs) == 0:
            return torch.min(x)
        return torch.min(x, *args, **kwargs)
    
    def any(self, x, *args, **kwargs):
        """Test whether any array element is True."""
        return torch.any(x, *args, **kwargs)
    
    def all(self, x, *args, **kwargs):
        """Test whether all array elements are True."""
        return torch.all(x, *args, **kwargs)
    
    def where(self, condition, x, y):
        """Return elements from x or y depending on condition."""
        return torch.where(condition, x, y)
    
    def isin(self, elements, test_elements):
        """
        Test whether each element of elements is in test_elements.
        PyTorch 1.10+ has torch.isin, but we implement it for compatibility.
        """
        if hasattr(torch, 'isin'):
            if isinstance(test_elements, (list, tuple)):
                test_elements = torch.tensor(test_elements, device=self.device, dtype=elements.dtype)
            return torch.isin(elements, test_elements)
        else:
            # Fallback implementation
            if isinstance(test_elements, (list, tuple)):
                test_elements = torch.tensor(test_elements, device=self.device, dtype=elements.dtype)
            mask = torch.zeros_like(elements, dtype=torch.bool)
            for val in test_elements:
                mask |= (elements == val)
            return mask
    
    def argwhere(self, condition):
        """Find indices where condition is True."""
        return torch.nonzero(condition, as_tuple=False)
    
    def unique(self, x, return_counts=False):
        """Find unique elements."""
        if return_counts:
            unique_vals, counts = torch.unique(x, return_counts=True)
            return unique_vals, counts
        return torch.unique(x)


# Create global xp module
xp = TorchArrayModule(device)

# Export the array module
__all__ = ['xp', 'USE_GPU', 'device', 'device_name', 'to_numpy', 'to_device', 
           'get_device', 'set_verbose_logging', 'DEFAULT_FLOAT_DTYPE', 'DEFAULT_INT_DTYPE']


def to_numpy(array):
    """
    Convert tensor to NumPy array (from GPU to CPU if needed).
    
    Args:
        array: PyTorch tensor or NumPy array
        
    Returns:
        NumPy array
    """
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


def to_device(array):
    """
    Convert array to tensor on the current device.
    
    Args:
        array: NumPy array or PyTorch tensor
        
    Returns:
        PyTorch tensor on device
    """
    if not isinstance(array, torch.Tensor):
        return torch.tensor(array, device=device)
    return array.to(device)


def get_device():
    """Get the current device being used."""
    return device


def set_verbose_logging(enabled):
    """Enable or disable verbose logging."""
    global VERBOSE_LOGGING
    VERBOSE_LOGGING = enabled
    log(f"Verbose logging {'enabled' if enabled else 'disabled'}")


def set_device(new_device):
    """
    Change the compute device (requires module reload to take full effect).
    
    Args:
        new_device (str): 'mps', 'cuda', or 'cpu'
    """
    global device, USE_GPU, device_name, xp
    
    if new_device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        USE_GPU = True
        device_name = "Apple Silicon MPS"
        log(f"✓ Switched to: {device_name}")
    elif new_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        USE_GPU = True
        device_name = f"NVIDIA CUDA (GPU: {torch.cuda.get_device_name(0)})"
        log(f"✓ Switched to: {device_name}")
    elif new_device == 'cpu':
        device = torch.device('cpu')
        USE_GPU = False
        device_name = "CPU"
        log(f"✓ Switched to: CPU-only mode")
    else:
        log(f"✗ Device '{new_device}' not available or invalid")
        return False
    
    # Recreate xp module with new device
    xp = TorchArrayModule(device)
    return True


# Log initialization summary
if VERBOSE_LOGGING:
    log(f"Backend initialized successfully")
    log(f"  - Device: {device}")
    log(f"  - GPU Acceleration: {USE_GPU}")
    log(f"  - Float dtype: {DEFAULT_FLOAT_DTYPE}")
    log(f"  - Int dtype: {DEFAULT_INT_DTYPE}")
