"""CUDA utilities for Mojo GPU kernels.

Provides CUDA Graph capture/replay via FFI + dlsym interception.
All APIs are compile-time no-ops on non-NVIDIA platforms.
"""

from .graph import CUDAGraph, maybe_capture_replay
