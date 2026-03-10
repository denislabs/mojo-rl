"""Shared utilities for Atari game GPU environments.

This module provides common constants and helper types used by all
game-specific environment structs (PongEnv, BreakoutEnv, etc.).

Each game implements GPUDiscreteEnv + BoxDiscreteActionEnv + RenderableEnv
directly (like CartPole), with game physics inlined in the GPU kernels.
"""

from std.memory import alloc
from core import State, Action
from render import Renderer2D

from .colors import SCREEN_W, SCREEN_H, PIXEL_OBS_DIM

comptime gpu_dtype = DType.float32


# ============================================================================
# Simple State/Action types for trait conformance
# ============================================================================


@fieldwise_init
struct AtariGameState(Copyable, ImplicitlyCopyable, Movable, State):
    """Generic state wrapper — just an integer index (needed for Env trait)."""

    var index: Int

    fn __init__(out self, *, copy: Self):
        self.index = copy.index

    fn __init__(out self, *, deinit take: Self):
        self.index = take.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct AtariGameAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Generic discrete action wrapper."""

    var value: Int

    fn __init__(out self, *, copy: Self):
        self.value = copy.value

    fn __init__(out self, *, deinit take: Self):
        self.value = take.value


# ============================================================================
# Workspace sizes for pixel observation mode
# ============================================================================

# Per-env workspace layout (pixel mode):
#   [0 .. 8399]        160×210 grayscale framebuffer (packed: 4 bytes per float32)
#   [8400 .. 36623]    4 × 84×84 frame stack (as float32)
#   [36624]            frame_idx (ring buffer write position)
comptime FRAME_BUF_F32_SIZE: Int = (SCREEN_W * SCREEN_H + 3) // 4  # 8400
comptime FRAME_STACK_F32_SIZE: Int = 4 * 84 * 84  # 28224
comptime PIXEL_WS_PER_ENV: Int = FRAME_BUF_F32_SIZE + FRAME_STACK_F32_SIZE + 1
