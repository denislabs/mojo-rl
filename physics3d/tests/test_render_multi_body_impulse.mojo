"""Visual demo for Physics3D v2 multi-body renderer.

Shows multiple spheres dropping, bouncing on ground and colliding with each other.
Press ESC or close window to exit.

Run with:
    pixi run mojo run physics3d/tests/test_render_multi_body.mojo

Time limit: Set MAX_DURATION_SECONDS to limit demo duration (0 = infinite).
"""

from physics3d.types import Model, Data
from physics3d.integrator import ImpulseIntegrator
from physics3d.render import Physics3DRenderer
from time import perf_counter_ns

# Configuration
comptime MAX_DURATION_SECONDS: Float64 = 30.0  # Set to 0 for infinite
comptime NUM_BODIES: Int = 5
comptime MAX_CONTACTS: Int = 20
comptime DTYPE = DType.float64


fn main() raises:
    """Run the multi-body visual physics demo."""
    print("=" * 60)
    print("Physics3D v2 - Multi-Body Visual Demo")
    print("=" * 60)
    print()
    print("Multiple spheres will drop, bounce, and collide.")
    print("Close the window or press ESC to exit.")
    if MAX_DURATION_SECONDS > 0:
        print("Auto-exit after", MAX_DURATION_SECONDS, "seconds.")
    print()

    # Physics setup - multiple bouncy balls
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.002,  # Small timestep for smooth animation
        ground_z=0.0,
        restitution=0.6,  # Bouncy!
        friction=0.3,
    )

    # Configure bodies with different sizes
    var radii = List[Float64]()
    radii.append(0.12)  # Body 0
    radii.append(0.10)  # Body 1
    radii.append(0.15)  # Body 2
    radii.append(0.08)  # Body 3
    radii.append(0.11)  # Body 4

    for i in range(NUM_BODIES):
        model.set_body(i, mass=1.0, radius=radii[i])

    # Initialize data with staggered positions
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Arrange spheres in different starting positions
    # Body 0: Center, high
    data.set_body_position(0, 0.0, 0.0, 2.0)

    # Body 1: Left, medium height
    data.set_body_position(1, -0.3, 0.0, 1.5)

    # Body 2: Right, low (will be hit by others)
    data.set_body_position(2, 0.25, 0.0, 0.5)

    # Body 3: Far left, high
    data.set_body_position(3, -0.5, 0.2, 2.5)

    # Body 4: Back, medium
    data.set_body_position(4, 0.1, 0.4, 1.8)

    # Give some initial velocities for more interesting motion
    data.set_body_velocity(1, 0.3, 0.0, 0.0)  # Moving right
    data.set_body_velocity(3, 0.2, -0.1, 0.0)  # Moving diagonally

    # Renderer setup
    var renderer = Physics3DRenderer(
        width=1024,
        height=768,
        show_velocity=True,
        show_contacts=True,
    )
    renderer.init()

    print("Simulation running...")
    print("  Number of bodies:", NUM_BODIES)
    print("  Restitution: 0.6")
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
            ImpulseIntegrator.step(model, data)

        # Render
        renderer.render(model, data)

        # Frame delay for ~60 FPS rendering
        renderer.delay(16)

        frame_count += 1

        # Print status occasionally
        if frame_count % 120 == 0:
            # Count contact types
            var ground_contacts = 0
            var sphere_contacts = 0
            for c in range(data.num_contacts):
                if data.contacts[c].body_b < 0:
                    ground_contacts += 1
                else:
                    sphere_contacts += 1
            print(
                "  Frame",
                frame_count,
                "- ground contacts:",
                ground_contacts,
                "sphere contacts:",
                sphere_contacts,
            )
            for i in range(NUM_BODIES):
                var x = data.positions[i * 3 + 0]
                var y = data.positions[i * 3 + 1]
                var z = data.positions[i * 3 + 2]
                print("    Body", i, ": pos=(", x, ",", y, ",", z, ")")

    renderer.close()

    print()
    print("Demo finished after", frame_count, "frames.")
    print("=" * 60)
