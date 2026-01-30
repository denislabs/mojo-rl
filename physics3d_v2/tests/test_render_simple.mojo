"""Simple visual demo for Physics3D v2 - Two spheres collision.

Shows two spheres approaching each other and bouncing.
This is a simpler scenario to verify the physics visually.

Run with:
    pixi run mojo run physics3d_v2/tests/test_render_simple.mojo
"""

from physics3d_v2.types import MultiBodyModel, MultiBodyData
from physics3d_v2.multi_body_step import step_multi_body
from physics3d_v2.render_multi_body import MultiBodyRenderer
from time import perf_counter_ns

comptime MAX_DURATION_SECONDS: Float64 = 30.0
comptime NUM_BODIES: Int = 2
comptime MAX_CONTACTS: Int = 10
comptime DTYPE = DType.float64


fn main() raises:
    print("=" * 60)
    print("Physics3D v2 - Simple Two-Sphere Demo")
    print("=" * 60)
    print()
    print("Two spheres approach and bounce off each other.")
    print("Watch them separate after collision.")
    print()

    # Setup: Two spheres approaching each other
    var model = MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=0.0,  # No gravity for cleaner demo
        timestep=0.002,
        ground_z=-10.0,  # Ground far below (no ground contact)
        restitution=0.9,  # High bounce
        friction=0.0,
    )

    model.set_body(0, mass=1.0, radius=0.15)
    model.set_body(1, mass=1.0, radius=0.15)

    var data = MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Start spheres apart, moving toward each other
    data.set_body_position(0, -0.5, 0.0, 0.5)
    data.set_body_position(1, 0.5, 0.0, 0.5)
    data.set_body_velocity(0, 0.5, 0.0, 0.0)  # Moving right
    data.set_body_velocity(1, -0.5, 0.0, 0.0)  # Moving left

    var renderer = MultiBodyRenderer(
        width=1024,
        height=768,
        show_velocity=True,
        show_shadows=False,
        show_contacts=True,
    )
    renderer.init()

    print("Simulation running...")
    print("  Sphere 0: starts at x=-0.5, moving right")
    print("  Sphere 1: starts at x=+0.5, moving left")
    print("  Restitution: 0.9 (high bounce)")
    print()

    var frame_count = 0
    var physics_steps_per_frame = 5
    var start_time_ns = perf_counter_ns()
    var collision_reported = False

    while not renderer.check_quit():
        if MAX_DURATION_SECONDS > 0:
            var elapsed_ns = perf_counter_ns() - start_time_ns
            var elapsed_seconds = Float64(elapsed_ns) / 1_000_000_000.0
            if elapsed_seconds >= MAX_DURATION_SECONDS:
                print("Time limit reached.")
                break

        for _ in range(physics_steps_per_frame):
            step_multi_body(model, data)

        renderer.render(model, data)
        renderer.delay(16)

        frame_count += 1

        # Report collision
        if not collision_reported and data.num_contacts > 0:
            for c in range(data.num_contacts):
                if data.contacts[c].body_a >= 0 and data.contacts[c].body_b >= 0:
                    print("  Collision detected!")
                    collision_reported = True
                    break

        # Report positions periodically
        if frame_count % 60 == 0:
            var x0 = data.positions[0 * 3 + 0]
            var x1 = data.positions[1 * 3 + 0]
            var v0 = data.velocities[0 * 3 + 0]
            var v1 = data.velocities[1 * 3 + 0]
            print("  Frame", frame_count, ": x0=", x0, " x1=", x1, " sep=", x1 - x0)

    renderer.close()
    print()
    print("Demo finished.")
