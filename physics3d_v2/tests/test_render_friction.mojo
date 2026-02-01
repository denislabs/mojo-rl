"""Visual demo for Physics3D v2 friction implementation.

Shows spheres with different friction behaviors:
- Sphere sliding to stop due to friction
- Spheres colliding with friction
- Comparison of high vs low friction

Press ESC or close window to exit.

Run with:
    pixi run mojo run physics3d_v2/tests/test_render_friction.mojo
"""

from physics3d_v2.types import Model, Data
from physics3d_v2.integrator import ImpulseIntegrator
from physics3d_v2.render import Physics3DRenderer
from time import perf_counter_ns

# Configuration
comptime MAX_DURATION_SECONDS: Float64 = 30.0  # Set to 0 for infinite
comptime NUM_BODIES: Int = 4
comptime MAX_CONTACTS: Int = 20
comptime DTYPE = DType.float64


fn main() raises:
    """Run the friction visual physics demo."""
    print("=" * 60)
    print("Physics3D v2 - Friction Visual Demo (Phase 6)")
    print("=" * 60)
    print()
    print("Demonstration of Coulomb friction:")
    print("  - Spheres slide and slow down due to friction")
    print("  - Two spheres will collide in the center")
    print("  - Watch how friction affects their motion")
    print()
    print("Close the window or press ESC to exit.")
    if MAX_DURATION_SECONDS > 0:
        print("Auto-exit after", MAX_DURATION_SECONDS, "seconds.")
    print()

    # Physics setup - lower friction to allow collisions
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.002,  # Small timestep for smooth animation
        ground_z=0.0,
        restitution=0.7,  # Higher bounce to see collisions
        friction=0.2,     # Lower friction so spheres can reach each other
    )

    # Configure bodies - all same size for clarity
    var radius: Float64 = 0.1
    for i in range(NUM_BODIES):
        model.set_body(i, mass=1.0, radius=radius)

    # Initialize data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Sphere 0: Left side, sliding right (will collide with sphere 1)
    data.set_body_position(0, -0.5, 0.0, radius)  # On ground, closer to center
    data.set_body_velocity(0, 3.0, 0.0, 0.0)      # Moving right at 3 m/s

    # Sphere 1: Right side, sliding left (will collide with sphere 0)
    data.set_body_position(1, 0.5, 0.0, radius)   # On ground, closer to center
    data.set_body_velocity(1, -3.0, 0.0, 0.0)     # Moving left at 3 m/s

    # Sphere 2: Back, sliding forward (independent sliding demo)
    data.set_body_position(2, 0.0, 0.8, radius)   # On ground
    data.set_body_velocity(2, 0.0, -2.0, 0.0)     # Moving forward faster

    # Sphere 3: Drops from height with horizontal velocity
    data.set_body_position(3, -0.3, -0.5, 0.8)    # Above ground
    data.set_body_velocity(3, 1.5, 0.5, 0.0)      # Moving diagonally

    # Renderer setup
    var renderer = Physics3DRenderer(
        width=1024,
        height=768,
        show_velocity=True,
        show_shadows=True,
        show_contacts=True,
    )
    renderer.init()

    print("Simulation running...")
    print("  Number of bodies:", NUM_BODIES)
    print("  Friction coefficient: 0.2")
    print("  Restitution: 0.7")
    print()
    print("Watch the spheres collide, bounce, and slow down due to friction!")
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

        # Print status occasionally - show velocities to demonstrate friction
        if frame_count % 60 == 0:
            print("  Frame", frame_count, "- Velocities:")
            for i in range(NUM_BODIES):
                var vx = data.velocities[i * 3 + 0]
                var vy = data.velocities[i * 3 + 1]
                var vz = data.velocities[i * 3 + 2]
                var speed = (vx * vx + vy * vy).__pow__(0.5)
                print("    Body", i, ": v=(", vx, ",", vy, ",", vz, ") speed=", speed)

    renderer.close()

    print()
    print("Demo finished after", frame_count, "frames.")
    print()

    # Final state summary
    print("Final state:")
    for i in range(NUM_BODIES):
        var x = data.positions[i * 3 + 0]
        var y = data.positions[i * 3 + 1]
        var z = data.positions[i * 3 + 2]
        var vx = data.velocities[i * 3 + 0]
        var vy = data.velocities[i * 3 + 1]
        var vz = data.velocities[i * 3 + 2]
        var speed = (vx * vx + vy * vy).__pow__(0.5)
        print("  Body", i, ": pos=(", x, ",", y, ",", z, ") speed=", speed, "m/s")

    print("=" * 60)
