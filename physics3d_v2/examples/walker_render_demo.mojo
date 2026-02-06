"""Bipedal Walker Visual Demonstration (Phase 10a).

Renders a bipedal walker simulation using SDL2.
Shows the 3-body walker with torso and two legs.

Controls:
- Close window to exit
- Walker automatically applies alternating torques for walking motion

Run with:
    cd mojo-rl
    pixi run mojo run physics3d_v2/examples/walker_render_demo.mojo

Requirements: SDL2 installed (brew install sdl2 sdl2_ttf on macOS)
"""

from physics3d_v2.types import Model, Data
from physics3d_v2.integrator import ImpulseIntegrator
from physics3d_v2.render import Physics3DRenderer

# Configuration
comptime NUM_BODIES: Int = 3
comptime MAX_CONTACTS: Int = 15
comptime MAX_JOINTS: Int = 2
comptime DTYPE = DType.float64
comptime PI: Float64 = 3.14159265358979323846


fn main() raises:
    print("=" * 60)
    print("    Bipedal Walker Visual Demonstration (Phase 10a)")
    print("=" * 60)
    print()
    print("Close the window to exit.")
    print()

    # Physics parameters (matching WalkerEnv but with better visual proportions)
    var torso_mass: Float64 = 1.0
    var torso_radius: Float64 = 0.08     # Smaller torso for better proportions
    var leg_mass: Float64 = 0.3
    var leg_radius: Float64 = 0.04       # Thin leg capsule
    var leg_half_length: Float64 = 0.12  # Slightly longer legs
    var hip_offset_x: Float64 = 0.06     # Narrower stance
    var hip_offset_z: Float64 = 0.08     # Hip closer to torso center
    # Hip is at top of leg capsule: half_length + radius above center
    var hip_height = leg_half_length + leg_radius  # 0.16
    var dt: Float64 = 0.005
    var torque_limit: Float64 = 15.0

    print("Setup:")
    print("  Torso: mass =", torso_mass, "kg, radius =", torso_radius, "m")
    print("  Legs: mass =", leg_mass, "kg, capsule radius =", leg_radius, "m, half_length =", leg_half_length, "m")
    print("  Timestep:", dt, "s")
    print("  Torque limit:", torque_limit, "N*m")
    print()

    # Create model with 3 bodies and 2 joints
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](0.0),
        restitution=Scalar[DTYPE](0.0),
        friction=Scalar[DTYPE](0.8),
    )

    # Torso (body 0) - sphere
    model.set_body(0, mass=Scalar[DTYPE](torso_mass), radius=Scalar[DTYPE](torso_radius))

    # Left Leg (body 1) - vertical capsule
    model.set_body_capsule(
        1,
        mass=Scalar[DTYPE](leg_mass),
        radius=Scalar[DTYPE](leg_radius),
        half_length=Scalar[DTYPE](leg_half_length),
    )

    # Right Leg (body 2) - vertical capsule
    model.set_body_capsule(
        2,
        mass=Scalar[DTYPE](leg_mass),
        radius=Scalar[DTYPE](leg_radius),
        half_length=Scalar[DTYPE](leg_half_length),
    )

    # Left Hip: Torso -> Left Leg
    _ = model.add_hinge_joint(
        parent=0,
        child=1,
        anchor_parent=(
            Scalar[DTYPE](-hip_offset_x),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](-hip_offset_z),
        ),
        anchor_child=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](hip_height),  # Top of leg capsule
        ),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )
    model.joints[0].torque_limit = Scalar[DTYPE](torque_limit)

    # Right Hip: Torso -> Right Leg
    _ = model.add_hinge_joint(
        parent=0,
        child=2,
        anchor_parent=(
            Scalar[DTYPE](hip_offset_x),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](-hip_offset_z),
        ),
        anchor_child=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](hip_height),  # Top of leg capsule
        ),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )
    model.joints[1].torque_limit = Scalar[DTYPE](torque_limit)

    # Initialize data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Calculate initial positions (matching WalkerEnv)
    # Leg capsule center: positioned so lowest point touches ground
    # Lowest point at z = center_z - half_length - radius = 0
    # So center_z = half_length + radius
    var leg_z = leg_half_length + leg_radius  # 0.14

    # Torso z: above legs, connected by hips
    # Hip is at top of leg capsule: leg_z + hip_height
    # Then add hip_offset_z to get to torso center
    var torso_z = leg_z + hip_height + hip_offset_z  # 0.48

    # Torso at center
    data.set_body_position(
        0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](torso_z)
    )

    # Left leg offset left
    data.set_body_position(
        1, Scalar[DTYPE](-hip_offset_x), Scalar[DTYPE](0.0), Scalar[DTYPE](leg_z)
    )

    # Right leg offset right
    data.set_body_position(
        2, Scalar[DTYPE](hip_offset_x), Scalar[DTYPE](0.0), Scalar[DTYPE](leg_z)
    )

    # Create renderer
    var renderer = Physics3DRenderer(
        width=1024,
        height=768,
        show_velocity=False,
        show_contacts=True,
    )
    renderer.init()

    print("Simulation running...")
    print("Walker with ZERO torque - should stand or fall naturally.")
    print()

    # Simulation loop
    var steps_per_frame = 10  # Run 10 physics steps per rendered frame
    var frame_count = 0
    var sim_time: Float64 = 0.0

    # No torque applied - natural behavior
    model.joints[0].target_torque = Scalar[DTYPE](0.0)
    model.joints[1].target_torque = Scalar[DTYPE](0.0)

    while not renderer.check_quit():
        # Run physics with zero torque
        for _ in range(steps_per_frame):
            ImpulseIntegrator.step(model, data)
            sim_time += dt

        # Render using standard multi-body render with joints
        renderer.render_with_joints(model, data)

        # Frame delay for ~60 FPS
        renderer.delay(16)

        frame_count += 1
        if frame_count % 100 == 0:
            var torso_pos = data.get_body_position(0)
            var left_pos = data.get_body_position(1)
            var right_pos = data.get_body_position(2)
            print(
                "Frame", frame_count,
                ": Torso = (", Float64(torso_pos[0]), ",", Float64(torso_pos[2]),
                "), Left = (", Float64(left_pos[0]), ",", Float64(left_pos[2]),
                "), Right = (", Float64(right_pos[0]), ",", Float64(right_pos[2]), ")",
            )

    renderer.close()
    print()
    print("Demo finished.")
