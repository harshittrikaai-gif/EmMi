import json
import os
import time
from pathlib import Path
from typing import Dict, Any

class NeuralMetricsMonitor:
    """
    Lightweight monitor that logs training metrics to a shared file.
    Used for real-time observability in the Streamlit UI.
    """
    def __init__(self, log_dir: str = "outputs/monitor"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "latest_metrics.json"
        self.history_file = self.log_dir / "history.json"
        
        # Reset current run
        self._history = []
        self._save_json({"step": 0, "status": "initializing"}, self.metrics_file)
        self._save_json([], self.history_file)

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """Log a single training step."""
        data = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        # Save latest for quick polling
        self._save_json(data, self.metrics_file)
        
        # Update history (if step is a multiple of N to save memory)
        if step % 5 == 0:
            self._history.append(data)
            self._save_json(self._history, self.history_file)

    def _save_json(self, data: Any, path: Path):
        """Atomic write to prevent UI read-during-write issues."""
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, path)

    def mark_complete(self):
        """Finalize the run."""
        with open(self.metrics_file, "r") as f:
            data = json.load(f)
        data["status"] = "complete"
        self._save_json(data, self.metrics_file)
