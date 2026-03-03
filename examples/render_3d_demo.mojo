"""Simple 3D rendering demo using the Renderer3D GPU renderer.

Shows spheres, boxes, and capsules with lighting, shadows, and a ground grid.
The camera auto-orbits and can be controlled with mouse (drag to orbit,
scroll to zoom, right-drag to pan).

Run with:
    pixi run mojo run examples/render_3d_demo.mojo
"""

from math import sin, cos, pi
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from render import (
    Renderer3D,
    Camera3D,
    Color,
    Light,
    LightMode,
    blue,
    red,
    green,
    orange,
    yellow,
    cyan,
    magenta,
    white,
    gray,
)

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]

comptime WIDTH = 1280
comptime HEIGHT = 720
comptime TARGET_FPS = 60


fn main() raises:
    print("3D Rendering Demo")
    print("  Mouse left-drag : orbit camera")
    print("  Mouse right-drag: pan camera")
    print("  Scroll          : zoom")
    print("  R               : reset camera")
    print("  Escape          : quit")
    print()

    # -------------------------------------------------------------------------
    # Camera: positioned behind and above the scene, looking at the origin
    # -------------------------------------------------------------------------
    var camera = Camera3D(
        eye=Vec3(0.0, -6.0, 3.0),
        target=Vec3(0.0, 0.0, 0.5),
        up=Vec3(0.0, 0.0, 1.0),
        fov=60.0,
        aspect=Float64(WIDTH) / Float64(HEIGHT),
        near=0.1,
        far=100.0,
        screen_width=WIDTH,
        screen_height=HEIGHT,
    )

    # -------------------------------------------------------------------------
    # Renderer
    # -------------------------------------------------------------------------
    var renderer = Renderer3D(
        width=WIDTH,
        height=HEIGHT,
        camera=camera,
        draw_grid=True,
        draw_axes=True,
    )

    print("Renderer created, calling init...")
    var title = String("3D Rendering Demo")
    renderer.init(title)
    print("Init done, entering main loop...")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    var frame: Int = 0

    while not renderer.check_quit():
        var t = Float64(frame) * 0.02  # time in "seconds" at 50 steps/s visual

        renderer.begin_frame()
        if frame == 0:
            print("Frame 0: begin_frame done")

        # --- Bouncing blue sphere ---
        var sphere_z = 0.4 + 0.3 * sin(t * 2.0)
        renderer.draw_sphere(
            center=Vec3(0.0, 0.0, sphere_z),
            radius=0.35,
            color=blue(),
            shininess=0.8,
            specular=0.9,
        )

        # --- Spinning red box ---
        var box_angle = t * 1.5
        var box_q = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), box_angle)
        renderer.draw_box(
            center=Vec3(1.8, 0.0, 0.3),
            orientation=box_q,
            half_extents=Vec3(0.3, 0.3, 0.3),
            color=red(),
            shininess=0.4,
            specular=0.6,
        )

        # --- Tilted orange capsule ---
        var cap_q = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float64(pi) / 4.0)
        renderer.draw_capsule(
            center=Vec3(-1.8, 0.0, 0.5),
            orientation=cap_q,
            radius=0.2,
            half_height=0.4,
            axis=2,
            color=orange(),
            shininess=0.3,
            specular=0.5,
        )

        # --- Orbiting green sphere ---
        var orbit_r: Float64 = 1.2
        var orbit_x = orbit_r * cos(t)
        var orbit_y = orbit_r * sin(t)
        renderer.draw_sphere(
            center=Vec3(orbit_x, orbit_y, 0.25),
            radius=0.2,
            color=green(),
            shininess=0.6,
            specular=0.7,
        )

        # --- Yellow sphere (back) ---
        renderer.draw_sphere(
            center=Vec3(0.0, 2.0, 0.3),
            radius=0.25,
            color=yellow(),
        )

        # --- Cyan capsule (horizontal) ---
        var h_cap_q = Quat.from_axis_angle(
            Vec3(1.0, 0.0, 0.0), Float64(pi) / 2.0
        )
        renderer.draw_capsule(
            center=Vec3(0.0, -2.0, 0.25),
            orientation=h_cap_q,
            radius=0.15,
            half_height=0.5,
            axis=2,
            color=cyan(),
        )

        # --- Connection lines between spheres ---
        renderer.draw_line_3d(
            start=Vec3(0.0, 0.0, sphere_z),
            end=Vec3(orbit_x, orbit_y, 0.25),
            color=gray(),
        )

        # Render grid + axes, then submit
        renderer.render_scene()
        if frame == 0:
            print("Frame 0: before end_frame")
        renderer.end_frame()
        if frame == 0:
            print("Frame 0: end_frame done")

        frame += 1

    renderer.close()
    print("Demo finished.")
