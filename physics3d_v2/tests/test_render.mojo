"""Visual demo for Physics3D v2 renderer.

Shows a ball dropping and bouncing on the ground.
Press ESC or close window to exit.

Run with:
    pixi run mojo run physics3d_v2/tests/test_render.mojo

Time limit: Set MAX_DURATION_SECONDS to limit demo duration (0 = infinite).
"""

from physics3d_v2.types import Body, Geom, Model, Data
from physics3d_v2 import step
from physics3d_v2.render import Physics3DRenderer
from time import perf_counter_ns

# Configuration
comptime MAX_DURATION_SECONDS: Float64 = 30.0  # Set to 0 for infinite


fn main() raises:
    """Run the visual physics demo."""
    print("=" * 60)
    print("Physics3D v2 - Visual Demo")
    print("=" * 60)
    print()
    print("A ball will drop and bounce on the ground.")
    print("Close the window or press ESC to exit.")
    if MAX_DURATION_SECONDS > 0:
        print("Auto-exit after", MAX_DURATION_SECONDS, "seconds.")
    print()

    # Physics setup - bouncy ball
    var radius: Float64 = 0.15
    var body = Body.create_sphere(mass=1.0, radius=radius)
    var geom = Geom.sphere(radius)
    var model = Model.create(
        body,
        geom,
        timestep=0.002,  # Small timestep for smooth animation
        gravity_z=-9.81,
        ground_z=0.0,
        restitution=0.7,  # Bouncy!
    )

    var data = Data[DType.float64]()
    data.set_position(0, 0, 2.0)  # Start 2m above ground

    # Renderer setup
    var renderer = Physics3DRenderer(
        width=1024,
        height=768,
        show_velocity=True,
        show_shadows=True,
        show_contact=True,
    )
    renderer.init()

    print("Simulation running...")
    print("  Initial height: 2.0 m")
    print("  Ball radius:", radius, "m")
    print("  Restitution: 0.7")
    print()

    # Simulation loop
    var frame_count = 0
    var physics_steps_per_frame = 5  # Multiple physics steps per render frame
    var start_time_ns = perf_counter_ns()

    while not renderer.check_quit():
        # Check time limit
        if MAX_DURATION_SECONDS > 0:
            var elapsed_ns = perf_counter_ns() - start_time_ns
            var elapsed_seconds = Float64(elapsed_ns) / 1_000_000_000.0
            if elapsed_seconds >= MAX_DURATION_SECONDS:
                print("  Time limit reached (", MAX_DURATION_SECONDS, "s)")
                break

        # Run multiple physics steps per frame
        for _ in range(physics_steps_per_frame):
            step(model, data)

        # Render
        renderer.render(model, data)

        # Frame delay for ~60 FPS rendering
        renderer.delay(16)

        frame_count += 1

        # Print status occasionally
        if frame_count % 120 == 0:
            var z = data.get_z()
            var vz = data.get_vz()
            print("  z =", z, "m, vz =", vz, "m/s")

    renderer.close()

    print()
    print("Demo finished after", frame_count, "frames.")
    print("=" * 60)
