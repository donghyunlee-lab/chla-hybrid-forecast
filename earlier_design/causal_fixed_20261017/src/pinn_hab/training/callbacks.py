"""
Training callbacks for progress tracking and logging.
"""


class ProgressTracker:
    """
    Tracks and displays the overall progress of the pipeline.
    """
    
    def __init__(self):
        self.current_percent = 0
        
    def update(self, percent: int, message: str):
        """Update progress bar."""
        self.current_percent = max(self.current_percent, percent)
        GREEN = '\033[92m'
        RESET = '\033[0m'
        print(f"\n{GREEN}[PROGRESS] {self.current_percent}% - {message}{RESET}")
        
    def complete(self):
        """Mark as complete."""
        self.update(100, "Pipeline Completed!")


class TensorBoardCallback:
    """
    TensorBoard logging callback.
    """
    
    def __init__(self, log_dir: str = 'runs'):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("[WARNING] TensorBoard not available")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()
