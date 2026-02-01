"""Double Pendulum Visual Demonstration.

Renders a double pendulum (two-link chain) simulation using SDL2.
Shows the chaotic motion characteristic of double pendulums.

Run with:
    cd mojo-rl
    pixi run mojo run physics3d_v2/examples/double_pendulum_render_demo.mojo

Requirements: SDL2 installed (brew install sdl2 sdl2_ttf on macOS)
"""

from math import sin, cos
from physics3d_v2.types import Model, Data
from physics3d_v2.integrator import ImpulseIntegrator
from physics3d_v2.render import Physics3DRenderer

# Configuration
comptime NUM_BODIES: Int = 2
comptime MAX_CONTACTS: Int = 10
comptime MAX_JOINTS: Int = 2
comptime DTYPE = DType.float64
comptime PI: Float64 = 3.14159265358979323846


fn main() raises:
    print("=" * 60)
    print("    Double Pendulum Visual Demonstration")
    print("=" * 60)
    print()
    print("Close the window to exit.")
    print()

    # Physics parameters
    var L1: Float64 = 1.0  # Length of first link
    var L2: Float64 = 1.0  # Length of second link
    var mass: Float64 = 1.0
    var radius: Float64 = 0.1  # Larger for visibility
    var initial_angle_deg: Float64 = 120.0  # Large angle for chaotic motion
    var initial_angle = initial_angle_deg * PI / 180.0
    var dt: Float64 = 0.001
    var pivot_z = L1

    print("Setup:")
    print("  L1 =", L1, "m, L2 =", L2, "m")
    print("  Initial angle:", initial_angle_deg, "degrees")
    print("  Timestep:", dt, "s")
    print()

    # Create model with 2 bodies and 2 joints
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](-3.0),  # Ground visible but below pendulum
        restitution=Scalar[DTYPE](0.0),
    )
    model.set_body(0, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))
    model.set_body(1, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))

    # Joint 0: World -> Body 0
    _ = model.add_hinge_joint(
        parent=-1,  # World anchor
        child=0,
        anchor_parent=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](pivot_z),
        ),
        anchor_child=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](L1),
        ),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )

    # Joint 1: Body 0 -> Body 1
    _ = model.add_hinge_joint(
        parent=0,
        child=1,
        anchor_parent=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
        ),
        anchor_child=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](L2),
        ),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )

    # Initialize data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Body 0 at initial angle
    var body0_x = L1 * sin(initial_angle)
    var body0_z = pivot_z - L1 * cos(initial_angle)
    data.set_body_position(
        0, Scalar[DTYPE](body0_x), Scalar[DTYPE](0.0), Scalar[DTYPE](body0_z)
    )

    # Body 1 at same angle from body 0
    var body1_x = body0_x + L2 * sin(initial_angle)
    var body1_z = body0_z - L2 * cos(initial_angle)
    data.set_body_position(
        1, Scalar[DTYPE](body1_x), Scalar[DTYPE](0.0), Scalar[DTYPE](body1_z)
    )

    # Set initial quaternions
    var half_angle = initial_angle / 2.0
    data.quaternions[0 * 4 + 0] = Scalar[DTYPE](0.0)
    data.quaternions[0 * 4 + 1] = Scalar[DTYPE](-sin(half_angle))
    data.quaternions[0 * 4 + 2] = Scalar[DTYPE](0.0)
    data.quaternions[0 * 4 + 3] = Scalar[DTYPE](cos(half_angle))
    data.quaternions[1 * 4 + 0] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 1] = Scalar[DTYPE](-sin(half_angle))
    data.quaternions[1 * 4 + 2] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 3] = Scalar[DTYPE](cos(half_angle))

    # Create renderer
    var renderer = Physics3DRenderer(
        width=1024,
        height=768,
        show_velocity=False,  # Disable for cleaner visualization
        show_shadows=True,
        show_contacts=False,
    )
    renderer.init()

    print("Simulation running...")
    print("Observe the chaotic motion of the double pendulum.")
    print()

    # Simulation loop
    var steps_per_frame = 10  # Run 10 physics steps per rendered frame
    var frame_count = 0

    while not renderer.check_quit():
        # Run physics
        for _ in range(steps_per_frame):
            ImpulseIntegrator.step(model, data)

        # Render
        renderer.render_with_joints(model, data)

        # Frame delay for ~60 FPS
        renderer.delay(16)

        frame_count += 1
        if frame_count % 100 == 0:
            var p0_x = Float64(data.positions[0])
            var p0_z = Float64(data.positions[2])
            var p1_x = Float64(data.positions[3])
            var p1_z = Float64(data.positions[5])
            print(
                "Frame",
                frame_count,
                ": Body0 = (",
                p0_x,
                ",",
                p0_z,
                "), Body1 = (",
                p1_x,
                ",",
                p1_z,
                ")",
            )

    renderer.close()
    print()
    print("Demo finished.")
