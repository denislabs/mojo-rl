"""4-Body Hopper Visual Demonstration.

Renders a realistic 4-body hopper simulation using SDL2.
Shows the hopper with:
- Torso: Capsule (vertical)
- Thigh: Capsule (vertical)
- Leg: Capsule (vertical)
- Foot: Capsule (horizontal, rotated 90° around Y-axis)

Controls:
- Close window to exit
- Hopper stands with zero torque (natural behavior)

Run with:
    cd mojo-rl
    pixi run mojo run physics3d_v2/examples/hopper_render_demo.mojo

Requirements: SDL2 installed (brew install sdl2 sdl2_ttf on macOS)
"""

from physics3d_v2.types import Model, Data
from physics3d_v2.integrator import ImpulseIntegrator
from physics3d_v2.render import Physics3DRenderer

# Configuration: 4 bodies, 20 max contacts, 3 joints
comptime NUM_BODIES: Int = 4
comptime MAX_CONTACTS: Int = 20
comptime MAX_JOINTS: Int = 3
comptime DTYPE = DType.float64


fn main() raises:
    print("=" * 60)
    print("    4-Body Hopper Visual Demonstration")
    print("=" * 60)
    print()
    print("Close the window to exit.")
    print()

    # MuJoCo-like body dimensions
    var torso_mass: Float64 = 1.0
    var torso_radius: Float64 = 0.05
    var torso_half_length: Float64 = 0.2

    var thigh_mass: Float64 = 0.5
    var thigh_radius: Float64 = 0.05
    var thigh_half_length: Float64 = 0.225

    var leg_mass: Float64 = 0.3
    var leg_radius: Float64 = 0.04
    var leg_half_length: Float64 = 0.25

    var foot_mass: Float64 = 0.2
    var foot_radius: Float64 = 0.06
    var foot_half_length: Float64 = 0.195

    var dt: Float64 = 0.002
    var torque_limit: Float64 = 200.0

    print("Setup (MuJoCo-like dimensions):")
    print("  Torso: mass =", torso_mass, "kg, radius =", torso_radius, "m, half_length =", torso_half_length, "m")
    print("  Thigh: mass =", thigh_mass, "kg, radius =", thigh_radius, "m, half_length =", thigh_half_length, "m")
    print("  Leg: mass =", leg_mass, "kg, radius =", leg_radius, "m, half_length =", leg_half_length, "m")
    print("  Foot: mass =", foot_mass, "kg, radius =", foot_radius, "m, half_length =", foot_half_length, "m (HORIZONTAL)")
    print("  Timestep:", dt, "s")
    print("  Torque limit:", torque_limit, "N*m")
    print()

    # Create model with 4 bodies and 3 joints
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](0.0),
        restitution=Scalar[DTYPE](0.0),
        friction=Scalar[DTYPE](0.9),
    )

    # Body 0: Torso (vertical capsule)
    model.set_body_capsule(
        0,
        mass=Scalar[DTYPE](torso_mass),
        radius=Scalar[DTYPE](torso_radius),
        half_length=Scalar[DTYPE](torso_half_length),
    )

    # Body 1: Thigh (vertical capsule)
    model.set_body_capsule(
        1,
        mass=Scalar[DTYPE](thigh_mass),
        radius=Scalar[DTYPE](thigh_radius),
        half_length=Scalar[DTYPE](thigh_half_length),
    )

    # Body 2: Leg (vertical capsule)
    model.set_body_capsule(
        2,
        mass=Scalar[DTYPE](leg_mass),
        radius=Scalar[DTYPE](leg_radius),
        half_length=Scalar[DTYPE](leg_half_length),
    )

    # Body 3: Foot (horizontal capsule - will be rotated 90° around Y-axis)
    model.set_body_capsule(
        3,
        mass=Scalar[DTYPE](foot_mass),
        radius=Scalar[DTYPE](foot_radius),
        half_length=Scalar[DTYPE](foot_half_length),
    )

    # Joint 0: Hip (Torso -> Thigh)
    _ = model.add_hinge_joint(
        parent=0,
        child=1,
        anchor_parent=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](-torso_half_length),  # Bottom of torso
        ),
        anchor_child=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](thigh_half_length),  # Top of thigh
        ),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )
    model.joints[0].torque_limit = Scalar[DTYPE](torque_limit)

    # Joint 1: Knee (Thigh -> Leg)
    _ = model.add_hinge_joint(
        parent=1,
        child=2,
        anchor_parent=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](-thigh_half_length),  # Bottom of thigh
        ),
        anchor_child=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](leg_half_length),  # Top of leg
        ),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )
    model.joints[1].torque_limit = Scalar[DTYPE](torque_limit)

    # Joint 2: Ankle (Leg -> Foot)
    _ = model.add_hinge_joint(
        parent=2,
        child=3,
        anchor_parent=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](-leg_half_length),  # Bottom of leg
        ),
        anchor_child=(
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.0),  # Center of foot
        ),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )
    model.joints[2].torque_limit = Scalar[DTYPE](torque_limit)

    # Initialize data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Calculate initial positions from ground up (matching MuJoCo)
    # Foot is horizontal, so its height is just its radius
    var foot_z = foot_radius  # ~0.06

    # Leg center: above foot
    var leg_z = foot_z + leg_radius + leg_half_length  # ~0.35

    # Thigh center: above leg
    var thigh_z = leg_z + leg_half_length + thigh_half_length  # ~0.825

    # Torso center: above thigh
    var torso_z = thigh_z + thigh_half_length + torso_half_length  # ~1.25

    print("Initial positions:")
    print("  Foot z:", foot_z, "(horizontal capsule, height = radius)")
    print("  Leg z:", leg_z)
    print("  Thigh z:", thigh_z)
    print("  Torso z:", torso_z)
    print()

    # Set body positions
    data.set_body_position(0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](torso_z))
    data.set_body_position(1, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](thigh_z))
    data.set_body_position(2, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](leg_z))
    data.set_body_position(3, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](foot_z))

    # Set quaternions: identity for vertical bodies (torso, thigh, leg)
    for i in range(3):
        data.quaternions[i * 4 + 0] = 0.0  # qx
        data.quaternions[i * 4 + 1] = 0.0  # qy
        data.quaternions[i * 4 + 2] = 0.0  # qz
        data.quaternions[i * 4 + 3] = 1.0  # qw

    # Foot quaternion: 90° rotation around Y-axis (horizontal capsule)
    # Quaternion for Y-axis rotation: qx=0, qy=sin(θ/2), qz=0, qw=cos(θ/2)
    # θ = 90° = π/2, so sin(π/4) ≈ 0.70710678, cos(π/4) ≈ 0.70710678
    data.quaternions[3 * 4 + 0] = 0.0           # qx
    data.quaternions[3 * 4 + 1] = 0.70710678    # qy (sin(π/4))
    data.quaternions[3 * 4 + 2] = 0.0           # qz
    data.quaternions[3 * 4 + 3] = 0.70710678    # qw (cos(π/4))

    print("Foot quaternion set to 90° Y rotation (horizontal capsule)")
    print()

    # Create renderer
    var renderer = Physics3DRenderer(
        width=1024,
        height=768,
        show_velocity=False,
        show_contacts=True,
    )
    renderer.init()

    print("Simulation running...")
    print("4-body hopper with ZERO torque - should stand naturally.")
    print()

    # Simulation loop
    var steps_per_frame = 25  # Run more physics steps per frame due to smaller timestep
    var frame_count = 0
    var sim_time: Float64 = 0.0

    # No torque applied - natural behavior
    for j in range(3):
        model.joints[j].target_torque = Scalar[DTYPE](0.0)

    # Damping coefficient to reduce oscillations at rest
    var linear_damping: Float64 = 0.995
    var angular_damping: Float64 = 0.99

    while not renderer.check_quit():
        # Run physics with zero torque
        for _ in range(steps_per_frame):
            ImpulseIntegrator.step(model, data)
            sim_time += dt

            # Apply damping to reduce movement at rest
            for i in range(NUM_BODIES):
                data.velocities[i * 3 + 0] *= Scalar[DTYPE](linear_damping)
                data.velocities[i * 3 + 1] *= Scalar[DTYPE](linear_damping)
                data.velocities[i * 3 + 2] *= Scalar[DTYPE](linear_damping)
                data.angular_velocities[i * 3 + 0] *= Scalar[DTYPE](angular_damping)
                data.angular_velocities[i * 3 + 1] *= Scalar[DTYPE](angular_damping)
                data.angular_velocities[i * 3 + 2] *= Scalar[DTYPE](angular_damping)

        # Render using standard multi-body render with joints
        renderer.render_with_joints(model, data)

        # Frame delay for ~60 FPS
        renderer.delay(16)

        frame_count += 1
        if frame_count % 100 == 0:
            var torso_pos = data.get_body_position(0)
            var foot_pos = data.get_body_position(3)
            print(
                "Frame", frame_count,
                ": Torso z =", Float64(torso_pos[2]),
                ", Foot z =", Float64(foot_pos[2]),
            )

    renderer.close()
    print()
    print("Demo finished.")
