"""
Memory management utilities for efficient GPU/CPU memory usage.
"""

import gc
import torch
from typing import Optional


class MemoryManager:
    """Helper methods for device cache cleanup and memory reporting."""
    
    @staticmethod
    def cleanup():
        """Clean up memory (garbage collection + device cache)."""
        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info() -> dict:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory stats
        """
        info = {"device": "cpu"}
        
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            info["reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
            info["max_allocated"] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        elif torch.backends.mps.is_available():
            info["device"] = "mps"
            # MPS doesn't have detailed memory stats
            info["allocated"] = None
            info["reserved"] = None
        
        return info
    
    @staticmethod
    def optimize_batch_size(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        target_memory_usage: float = 0.8,
        max_batch_size: int = 256
    ) -> int:
        """
        Find optimal batch size for the given model and input.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor (batch_size=1)
            target_memory_usage: Target fraction of available memory
            max_batch_size: Maximum batch size to try
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return 32  # Default for non-CUDA
        
        # Try increasing batch sizes
        for batch_size in [8, 16, 32, 64, 128, 256]:
            if batch_size > max_batch_size:
                break
            
            try:
                MemoryManager.cleanup()
                test_input = sample_input.repeat(batch_size, 1, 1)
                with torch.no_grad():
                    _ = model(test_input)
                
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_reserved()
                
                if memory_used > target_memory_usage:
                    return max(8, batch_size // 2)
                    
            except RuntimeError:
                return max(8, batch_size // 2)
        
        return batch_size
