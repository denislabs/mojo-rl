"""3D Rendering Demo.

Demonstrates the GPU-accelerated render module by rendering a simple 3D scene
with physics bodies (spheres, capsules, boxes) and ground plane.
Uses SDL3 GPU API with Metal (MSL) shaders.
"""

from render import (
    Renderer3D,
    Camera3D,
    Color,
)
from math3d import Vec3, Quat


fn main() raises:
    print("GPU 3D Rendering Demo")
    print("Press ESC to quit")
    print("=====================")

    # Create camera looking at the scene
    var camera = Camera3D(
        eye=Vec3(3.0, -6.0, 4.0),  # Camera position
        target=Vec3(0.0, 0.0, 1.0),  # Look at center
        up=Vec3(0.0, 0.0, 1.0),  # Z-up
        fov=60.0,
        screen_width=800,
        screen_height=450,
    )

    # Create renderer
    var renderer = Renderer3D(
        width=800,
        height=450,
        camera=camera,
        draw_grid=True,
        draw_axes=True,
    )

    # Initialize SDL3 GPU
    var title = String("GPU 3D Physics Demo")
    renderer.init(title)

    print("Rendering scene...")

    # Animation loop
    var frame = 0
    var angle = 0.0

    while not renderer.check_quit():
        # Begin frame
        renderer.begin_frame()

        # Draw scene elements (grid and axes)
        renderer.render_scene()

        # Draw a sphere
        renderer.draw_sphere(
            center=Vec3(0.0, 0.0, 1.0),
            radius=0.5,
            color=Color(0, 255, 255, 255),
        )

        # Draw a capsule (rotating)
        var capsule_quat = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), angle)
        renderer.draw_capsule(
            center=Vec3(2.0, 0.0, 0.5),
            orientation=capsule_quat,
            radius=0.15,
            half_height=0.3,
            axis=0,  # X-axis
            color=Color(255, 255, 0, 255),
        )

        # Draw a box (rotating)
        var box_quat = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), -angle * 0.5)
        renderer.draw_box(
            center=Vec3(-2.0, 0.0, 0.5),
            orientation=box_quat,
            half_extents=Vec3(0.4, 0.3, 0.3),
            color=Color(0, 255, 0, 255),
        )

        # Draw some more spheres as "bodies"
        for i in range(3):
            var x = -1.0 + Float64(i) * 1.0
            renderer.draw_sphere(
                center=Vec3(x, 2.0, 0.25),
                radius=0.25,
                color=Color(255, 0, 0, 255),
            )

        # Draw a line representing a link/connection
        renderer.draw_line_3d(
            Vec3(0.0, 0.0, 1.5),
            Vec3(2.0, 0.0, 0.5),
            Color(255, 255, 255, 255),
        )

        # End frame
        renderer.end_frame()

        # Update animation
        angle += 0.02
        frame += 1

        # Delay to cap at ~60 FPS
        renderer.delay_ms(16)

        # Print status every 60 frames
        if frame % 60 == 0:
            print("Frame:", frame)

    print("Demo finished.")
    renderer.close()
