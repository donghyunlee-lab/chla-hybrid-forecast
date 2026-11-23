"""
PyTorch device selection (CUDA, MPS, CPU), cache cleanup and Matplotlib font setup.
"""

import gc
import torch
import platform

# ---- Device configuration ----

def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    
    Returns:
        torch.device: cuda, mps, or cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_mps() -> bool:
    """Check if running on Apple Silicon MPS."""
    return torch.backends.mps.is_available()


def is_cuda() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def sync_device():
    """
    Synchronize device for accurate timing.
    MPS and CUDA ops are async; call this before measuring elapsed time.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def get_optimal_workers() -> int:
    """
    Get optimal num_workers for DataLoader.
    
    - MPS (Apple Silicon): 2 (fork-safe, avoids overhead)
    - CUDA: 4
    - CPU: 0 (in-process)
    """
    if torch.backends.mps.is_available():
        return 2
    elif torch.cuda.is_available():
        return 4
    return 0


# Global device instance
device = get_device()

_device_info = f"Using device: {device}"
if is_mps():
    _device_info += " (Apple Silicon MPS)"
elif is_cuda():
    _device_info += f" ({torch.cuda.get_device_name(0)})"
print(_device_info)


# ---- Memory cleanup ----

def cleanup_memory():
    """Run garbage collection and empty active device caches."""
    gc.collect()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Matplotlib Korean font setup ----

def configure_matplotlib_fonts():
    """Configure matplotlib for Korean font support."""
    try:
        import matplotlib.pyplot as plt
        
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        elif platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:  # Linux
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    except Exception:
        pass  # Use default font if Korean font not available


# Auto-configure matplotlib fonts on import
configure_matplotlib_fonts()

