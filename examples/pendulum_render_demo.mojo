"""Pendulum Render Demo.

Demonstrates the hinge joint constraint by rendering a swinging pendulum.
Uses physics3d_v2 for physics simulation and render3d for visualization.

Run with:
    cd mojo-rl
    pixi run mojo run examples/pendulum_render_demo.mojo
"""

from math import sin, cos, sqrt
from render3d import (
    Renderer3D,
    Camera3D,
    Color3D,
)
from render3d.shapes3d import WireframeLine
from math3d import Vec3, Quat
from physics3d_v2.types import Model, Data
from physics3d_v2.integrator import ImpulseIntegrator

# Physics configuration
comptime NUM_BODIES: Int = 1
comptime MAX_CONTACTS: Int = 5
comptime MAX_JOINTS: Int = 1
comptime DTYPE = DType.float64
comptime PI: Float64 = 3.14159265358979323846


fn main() raises:
    print("=" * 60)
    print("    Pendulum Render Demo (Hinge Joint)")
    print("=" * 60)
    print()
    print("Controls:")
    print("  ESC or Q - Quit")
    print()

    # Physics parameters
    var L: Float64 = 1.5  # Pendulum length (m)
    var mass: Float64 = 1.0
    var radius: Float64 = 0.15  # Bob radius for visualization
    var initial_angle_deg: Float64 = 45.0  # Starting angle
    var initial_angle = initial_angle_deg * PI / 180.0
    var dt: Float64 = 0.002  # Physics timestep
    var substeps = 4  # Physics substeps per frame

    print("Physics setup:")
    print("  Pendulum length:", L, "m")
    print("  Bob mass:", mass, "kg")
    print("  Initial angle:", initial_angle_deg, "degrees")
    print("  Timestep:", dt, "s")
    print()

    # Pivot position (where the pendulum is attached)
    var pivot_x: Float64 = 0.0
    var pivot_y: Float64 = 0.0
    var pivot_z: Float64 = 2.0  # Height of pivot

    # Create physics model
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](-10.0),  # Ground far below
        restitution=Scalar[DTYPE](0.0),
    )
    model.set_body(0, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))

    # Add hinge joint
    # anchor_parent = pivot location in world
    # anchor_child = offset from bob center to attachment point (L units up)
    _ = model.add_hinge_joint(
        parent=-1,  # World anchor
        child=0,
        anchor_parent=(Scalar[DTYPE](pivot_x), Scalar[DTYPE](pivot_y), Scalar[DTYPE](pivot_z)),
        anchor_child=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](L)),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),  # Y-axis rotation
    )

    # Initialize physics data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Calculate initial bob position (swung to initial_angle)
    var bob_x = L * sin(initial_angle)
    var bob_z = pivot_z - L * cos(initial_angle)
    data.set_body_position(0, Scalar[DTYPE](bob_x), Scalar[DTYPE](0.0), Scalar[DTYPE](bob_z))
    data.set_body_velocity(0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0))

    # Set initial quaternion: rotation by -initial_angle around Y-axis
    # This aligns the anchor_child vector toward the pivot
    var half_angle = initial_angle / 2.0
    data.quaternions[0] = Scalar[DTYPE](0.0)  # qx
    data.quaternions[1] = Scalar[DTYPE](-sin(half_angle))  # qy
    data.quaternions[2] = Scalar[DTYPE](0.0)  # qz
    data.quaternions[3] = Scalar[DTYPE](cos(half_angle))  # qw

    print("Initial bob position: (", bob_x, ", 0,", bob_z, ")")

    # Create camera looking at the pendulum
    var camera = Camera3D(
        eye=Vec3(5.0, -5.0, 3.0),  # Camera position
        target=Vec3(0.0, 0.0, 1.5),  # Look at pendulum area
        up=Vec3(0.0, 0.0, 1.0),  # Z-up
        fov=60.0,
        screen_width=800,
        screen_height=600,
    )

    # Create renderer
    var renderer = Renderer3D(
        width=800,
        height=600,
        camera=camera,
        draw_grid=True,
        draw_axes=True,
    )

    # Initialize SDL2
    var title = String("Pendulum Demo - Hinge Joint")
    renderer.init(title)

    print("\nStarting simulation...")

    # Animation loop
    var frame = 0
    var sim_time: Float64 = 0.0

    while not renderer.check_quit():
        # Physics substeps
        for _ in range(substeps):
            ImpulseIntegrator.step(model, data)
            sim_time += dt

        # Get current bob position
        var pos_x = Float64(data.positions[0])
        var pos_y = Float64(data.positions[1])
        var pos_z = Float64(data.positions[2])

        # Begin frame
        renderer.begin_frame()

        # Draw scene elements (grid and axes)
        renderer.render_scene()

        # Draw pivot point (small sphere)
        renderer.draw_sphere(
            center=Vec3(pivot_x, pivot_y, pivot_z),
            radius=0.05,
            color=Color3D.white(),
            segments=8,
            rings=6,
        )

        # Draw pendulum rod (line from pivot to bob)
        var rod = WireframeLine(
            Vec3(pivot_x, pivot_y, pivot_z),
            Vec3(pos_x, pos_y, pos_z),
        )
        renderer.draw_line_3d(rod, Color3D.yellow())

        # Draw bob (sphere)
        renderer.draw_sphere(
            center=Vec3(pos_x, pos_y, pos_z),
            radius=radius,
            color=Color3D.cyan(),
            segments=16,
            rings=12,
        )

        # Draw a shadow on the ground (XY plane at z=0)
        renderer.draw_sphere(
            center=Vec3(pos_x, pos_y, 0.01),
            radius=radius * 0.5,
            color=Color3D(40, 40, 40),
            segments=8,
            rings=4,
        )

        # Draw trail effect (optional - shows motion history)
        # We'll draw some ghost spheres at previous positions
        # This is a simple way to show the pendulum's path

        # End frame
        renderer.end_frame()

        frame += 1

        # Delay to cap at ~60 FPS
        renderer.delay(16)

        # Print status periodically
        if frame % 120 == 0:
            var vel_x = Float64(data.velocities[0])
            var vel_z = Float64(data.velocities[2])
            var speed = sqrt(vel_x * vel_x + vel_z * vel_z)
            print(
                "Frame:",
                frame,
                " t =",
                sim_time,
                "s  pos = (",
                pos_x,
                ",",
                pos_z,
                ")  speed =",
                speed,
                "m/s",
            )

    print()
    print("Demo finished after", frame, "frames")
    print("Simulation time:", sim_time, "s")
    renderer.close()
