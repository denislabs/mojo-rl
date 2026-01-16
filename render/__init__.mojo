"""Render package - SDL2-based rendering infrastructure for RL environments.

This package provides:
- SDL2 FFI bindings (sdl2.mojo)
- High-level renderer with common functionality (renderer_base.mojo)
- Transform utilities: Vec2, Transform2D, Camera (transform.mojo)
- Color utilities and palettes (colors.mojo)
- Common shape vertex definitions (shapes.mojo)
"""

# SDL2 low-level bindings
from .sdl2 import SDL2, SDL_Event, SDL_Point, SDL_Rect, SDL_Color, SDLHandle

# Transform utilities
from .transform import (
    Vec2,
    Transform2D,
    Camera,
    RotatingCamera,
    make_centered_camera,
    make_offset_camera,
    make_rotating_camera,
)

# Color utilities and palettes
from .colors import (
    Colors,
    # Basic colors
    white,
    black,
    red,
    green,
    blue,
    yellow,
    cyan,
    magenta,
    orange,
    purple,
    gray,
    light_gray,
    dark_gray,
    transparent,
    # Environment colors
    sky_blue,
    night_sky,
    space_black,
    ground_brown,
    grass_green,
    dark_grass,
    sand,
    mountain_brown,
    moon_gray,
    cart_blue,
    pole_tan,
    axle_purple,
    car_red,
    lander_gray,
    hull_purple,
    contact_green,
    no_contact_blue,
    active_green,
    inactive_gray,
    warning_orange,
    danger_red,
    velocity_orange,
    force_red,
    torque_blue,
    target_gold,
    flag_red,
    helipad_yellow,
    track_gray,
    track_visited,
    curb_red,
    curb_white,
    # Color utilities
    rgb,
    rgba,
    with_alpha,
    lerp_color,
    brighten,
    darken,
    grayscale,
    # Particle/effect colors
    flame_color,
    smoke_color,
    spark_color,
    # Gradients
    heat_gradient,
    rainbow_gradient,
)

# Shape factories
from .shapes import (
    # Basic shapes
    make_rect,
    make_box,
    make_triangle,
    make_circle,
    make_regular_polygon,
    make_hexagon,
    # Arrows and indicators
    make_arrow,
    make_simple_arrow_head,
    make_chevron,
    # Vehicle parts
    make_wheel,
    make_capsule,
    make_car_body,
    make_lander_body,
    make_leg_box,
    # UI elements
    make_flag,
    make_star,
    make_cross,
    # Terrain
    make_filled_terrain,
    # Transform utilities
    offset_vertices,
    scale_vertices,
    rotate_vertices,
    flip_vertices_y,
    flip_vertices_x,
)

# Native SDL2 renderer base (shared infrastructure)
from .renderer_base import RendererBase
