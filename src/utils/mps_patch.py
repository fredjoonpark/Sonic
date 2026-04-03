"""
Monkey-patch torch.nn.Conv3d to work on MPS by falling back to CPU.

MPS (Apple Silicon Metal backend) does not support Conv3d, and the
PYTORCH_ENABLE_MPS_FALLBACK env var is unreliable for conv ops.
This patch wraps Conv3d.forward so that inputs are moved to CPU,
the convolution runs there, and the output is moved back to MPS.

Import this module before any model construction when running on MPS.
"""

import torch
import torch.nn as nn

_original_conv3d_forward = nn.Conv3d.forward


def _conv3d_forward_mps_fallback(self, input):
    if input.device.type == "mps":
        # Move weight/bias to CPU, run conv, move result back to MPS
        result = _original_conv3d_forward(
            self.cpu(), input.cpu()
        ).to("mps")
        # Move the module back to MPS so parameters stay on the right device
        self.to("mps")
        return result
    return _original_conv3d_forward(self, input)


def patch_conv3d_for_mps():
    """Apply the Conv3d MPS→CPU fallback patch."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        nn.Conv3d.forward = _conv3d_forward_mps_fallback
