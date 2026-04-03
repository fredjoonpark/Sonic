"""
MPS compatibility patch for Apple Silicon.

Conv3d is not supported on MPS and PYTORCH_ENABLE_MPS_FALLBACK is unreliable.
This module monkey-patches nn.Conv3d.forward to run the operation on CPU
using detached copies of weights — avoiding the constant .to() on the module
itself which causes deadlocks.
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

    # Move input and weight copies to CPU, run there, move result back
    input_cpu = input.cpu()
    weight_cpu = self.weight.cpu()
    bias_cpu = self.bias.cpu() if self.bias is not None else None
    out = F.conv3d(input_cpu, weight_cpu, bias_cpu,
                   self.stride, self.padding, self.dilation, self.groups)
    return out.to("mps")


def patch_conv3d_for_mps():
    """Apply the Conv3d MPS→CPU fallback. Safe to call multiple times."""
    global _patched
    if _patched:
        return
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        nn.Conv3d.forward = _conv3d_forward_cpu_fallback
        _patched = True
