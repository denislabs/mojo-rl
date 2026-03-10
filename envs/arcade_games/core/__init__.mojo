"""Atari Games — Core infrastructure for native GPU game engines."""

from .colors import (
    COLOR_BLACK,
    COLOR_WHITE,
    COLOR_RED,
    COLOR_ORANGE,
    COLOR_YELLOW,
    COLOR_GREEN,
    COLOR_AQUA,
    COLOR_BLUE,
    COLOR_GRAY,
)
from .gpu_renderer import draw_filled_rect, draw_hline, clear_frame
from .preprocessing import resize_160x210_to_84x84, push_frame_stack
