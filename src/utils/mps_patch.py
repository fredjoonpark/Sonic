"""
MPS compatibility helpers for Apple Silicon.

Conv3d is not supported on MPS and the PYTORCH_ENABLE_MPS_FALLBACK env var
is unreliable. Instead we provide utilities to move models that contain
Conv3d (like the temporal VAE decoder) to CPU for their forward passes.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager


def has_conv3d(module: nn.Module) -> bool:
    """Check if a module or any of its children contain Conv3d layers."""
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            return True
    return False


@contextmanager
def on_cpu_if_mps(module: nn.Module, device):
    """Context manager that temporarily moves a module to CPU if running on MPS.
    
    Moves the module to CPU on entry and back to the original device on exit.
    No-op if the device is not MPS.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "mps":
        original_dtype = next(module.parameters()).dtype
        module.to(device="cpu", dtype=torch.float32)
        try:
            yield torch.device("cpu")
        finally:
            module.to(device=dev, dtype=original_dtype)
    else:
        yield dev
