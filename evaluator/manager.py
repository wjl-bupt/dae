# eval/checkpoint/manager.py
from pathlib import Path

class CheckpointManager:
    def __init__(self, root: str, pattern="*.pt"):
        self.root = Path(root)
        self.pattern = pattern

    def list_checkpoints(self):
        return sorted(self.root.glob(self.pattern))

    def load(self, ckpt_path, device):
        import torch
        return torch.load(ckpt_path, map_location=device)
