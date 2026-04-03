"""
MPS compatibility patch for Apple Silicon.

Conv3d is not supported on MPS in PyTorch <= 2.5.
This module provides two strategies:

1. (Preferred) Use the `mps-conv3d` package which provides a native Metal
   implementation — pip install mps-conv3d
2. (Fallback) Monkey-patch nn.Conv3d.forward to run on CPU using F.conv3d
   with detached weight copies, avoiding module-level .to() calls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_patched = False


def _conv3d_forward_cpu_fallback(self, input):
    """Run Conv3d on CPU when input is on MPS, without moving module params."""
    if input.device.type != "mps":
        return F.conv3d(input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    input_cpu = input.cpu()
    weight_cpu = self.weight.cpu()
    bias_cpu = self.bias.cpu() if self.bias is not None else None
    out = F.conv3d(input_cpu, weight_cpu, bias_cpu,
                   self.stride, self.padding, self.dilation, self.groups)
    return out.to("mps")


def patch_conv3d_for_mps():
    """Patch Conv3d to work on MPS. Safe to call multiple times."""
    global _patched
    if _patched:
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    # Check if current PyTorch already supports Conv3d on MPS
    try:
        x = torch.zeros(1, 1, 1, 1, 1, device="mps")
        w = torch.zeros(1, 1, 1, 1, 1, device="mps")
        F.conv3d(x, w)
        print("[MPS] Conv3d is natively supported — no patch needed")
        _patched = True
        return
    except (RuntimeError, NotImplementedError):
        pass

    # Strategy 1: try mps-conv3d (native Metal, full speed)
    try:
        from mps_conv3d import patch_conv3d
        patch_conv3d()
        print("[MPS] Using mps-conv3d native Metal implementation")
        _patched = True
        return
    except ImportError:
        pass

    # Strategy 2: CPU fallback (slow but works)
    print("[MPS] Conv3d not natively supported — using CPU fallback")
    print("[MPS] For better performance: pip install mps-conv3d")
    nn.Conv3d.forward = _conv3d_forward_cpu_fallback
    _patched = True
