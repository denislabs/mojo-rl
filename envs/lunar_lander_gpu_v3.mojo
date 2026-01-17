"""GPU-compatible LunarLander environment V3 - Improved contact physics.

Key changes from V2:
1. Contact detection accounts for leg shape (not just a point)
2. Impulse-based contact response matching Box2D
3. Better matching of CPU normalization formulas

State: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
Actions: 0=nop, 1=left_engine, 2=main_engine, 3=right_engine
"""

from math import cos, sin, sqrt, tanh, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random.philox import Random as PhiloxRandom

from deep_rl.gpu import random_range, xorshift32
from core import GPUDiscreteEnv


# =============================================================================
# Physics Constants - Matched exactly to CPU (lunar_lander.mojo)
# =============================================================================

comptime gpu_dtype = DType.float32

# Core physics
comptime GRAVITY: Float64 = -10.0
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SIDE_ENGINE_POWER: Float64 = 0.6
comptime SCALE: Float64 = 30.0
comptime FPS: Int = 50
comptime TAU: Float64 = 0.02  # 1/FPS

# Viewport/world dimensions
comptime VIEWPORT_W: Float64 = 600.0
comptime VIEWPORT_H: Float64 = 400.0
comptime W_UNITS: Float64 = VIEWPORT_W / SCALE  # 20.0
comptime H_UNITS: Float64 = VIEWPORT_H / SCALE  # 13.333...

# Helipad position (matches CPU: H/4)
comptime HELIPAD_Y: Float64 = H_UNITS / 4.0  # ~3.333
comptime HELIPAD_X: Float64 = W_UNITS / 2.0  # 10.0

# Initial spawn position (matches CPU)
comptime INITIAL_Y: Float64 = H_UNITS  # Top of viewport
comptime INITIAL_X: Float64 = W_UNITS / 2.0  # Center

# Lander geometry (from CPU lunar_lander.mojo)
comptime LEG_AWAY: Float64 = 20.0 / SCALE  # ~0.667 - horizontal distance from center
comptime LEG_DOWN: Float64 = 18.0 / SCALE  # 0.6 - vertical distance below center
comptime LEG_W: Float64 = 2.0 / SCALE  # Leg width (half-extent)
comptime LEG_H: Float64 = 8.0 / SCALE  # Leg height (half-extent)
comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / SCALE  # ~0.567

# Engine mount positions (from CPU)
comptime MAIN_ENGINE_Y_OFFSET: Float64 = 4.0 / SCALE
comptime SIDE_ENGINE_HEIGHT: Float64 = 14.0 / SCALE
comptime SIDE_ENGINE_AWAY: Float64 = 12.0 / SCALE

# Lander mass/inertia (from CPU: density=5.0 on polygon)
comptime LANDER_MASS: Float64 = 5.0
comptime LANDER_INERTIA: Float64 = 2.0

# Contact physics (from CPU Box2D settings)
comptime CONTACT_FRICTION: Float64 = 0.1
comptime CONTACT_RESTITUTION: Float64 = 0.0

# Reward constants
comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03


# =============================================================================
# GPU LunarLander Environment V3
# =============================================================================


struct LunarLanderGPUv3(GPUDiscreteEnv):
    """GPU-compatible LunarLander with improved contact physics.

    Key improvements over V2:
    - Leg contact considers leg box shape, not just tip point
    - Impulse-based contact response (j = -m*v, not velocity damping)
    - Matches CPU's semi-implicit Euler integration order
    """

    comptime STATE_SIZE: Int = 8  # No hidden state needed for simpler physics
    comptime OBS_DIM: Int = 8
    comptime NUM_ACTIONS: Int = 4
    comptime TPB: Int = 256

    @staticmethod
    @always_inline
    fn step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ],
        rewards: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        rng_seed: Scalar[DType.uint64],
    ):
        """GPU kernel for LunarLander physics step."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        # =====================================================================
        # Extract state (normalized observations)
        # =====================================================================
        var x_obs = states[i, 0]
        var y_obs = states[i, 1]
        var vx_obs = states[i, 2]
        var vy_obs = states[i, 3]
        var angle_obs = states[i, 4]
        var angular_vel_obs = states[i, 5]
        var left_contact = states[i, 6]
        var right_contact = states[i, 7]

        # Denormalize to world coordinates
        # CPU normalization (lunar_lander.mojo:914-931):
        #   x_norm = (pos.x - W/2) / (W/2)
        #   y_norm = (pos.y - (helipad_y + LEG_DOWN)) / (H/2)
        #   vx_norm = vel.x * (W/2) / FPS
        #   vy_norm = vel.y * (H/2) / FPS
        #   angle_norm = angle (raw radians)
        #   angular_vel_norm = 20 * angular_vel / FPS
        var x = x_obs * Scalar[gpu_dtype](W_UNITS / 2.0) + Scalar[gpu_dtype](HELIPAD_X)
        var y = y_obs * Scalar[gpu_dtype](H_UNITS / 2.0) + Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)
        var vx = vx_obs / Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy = vy_obs / Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        var angle = angle_obs
        var angular_vel = angular_vel_obs / Scalar[gpu_dtype](20.0 / Float64(FPS))

        # =====================================================================
        # Compute prev_shaping BEFORE physics
        # =====================================================================
        var dist_prev = sqrt(x_obs * x_obs + y_obs * y_obs)
        var speed_prev = sqrt(vx_obs * vx_obs + vy_obs * vy_obs)
        var angle_abs_prev = angle
        if angle_abs_prev < Scalar[gpu_dtype](0.0):
            angle_abs_prev = -angle_abs_prev

        var prev_shaping = (
            Scalar[gpu_dtype](-100.0) * dist_prev
            + Scalar[gpu_dtype](-100.0) * speed_prev
            + Scalar[gpu_dtype](-100.0) * angle_abs_prev
            + Scalar[gpu_dtype](10.0) * left_contact
            + Scalar[gpu_dtype](10.0) * right_contact
        )

        # =====================================================================
        # Apply engine forces
        # =====================================================================
        var action = Int(actions[i])
        var m_power = Scalar[gpu_dtype](0.0)
        var s_power = Scalar[gpu_dtype](0.0)

        # Direction vectors
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        # Random dispersion (matches CPU)
        var rng = PhiloxRandom(seed=Int(rng_seed[0]), offset=UInt64(i))
        var rand_vals = rng.step_uniform()
        var dispersion_x = (
            Scalar[gpu_dtype](rand_vals[0]) * Scalar[gpu_dtype](2.0)
            - Scalar[gpu_dtype](1.0)
        ) / Scalar[gpu_dtype](SCALE)
        var dispersion_y = (
            Scalar[gpu_dtype](rand_vals[1]) * Scalar[gpu_dtype](2.0)
            - Scalar[gpu_dtype](1.0)
        ) / Scalar[gpu_dtype](SCALE)

        if action == 2:
            # Main engine
            m_power = Scalar[gpu_dtype](1.0)
            var ox = (
                tip_x * (Scalar[gpu_dtype](MAIN_ENGINE_Y_OFFSET) + Scalar[gpu_dtype](2.0) * dispersion_x)
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y * (Scalar[gpu_dtype](MAIN_ENGINE_Y_OFFSET) + Scalar[gpu_dtype](2.0) * dispersion_x)
                - side_y * dispersion_y
            )
            var impulse_x = -ox * Scalar[gpu_dtype](MAIN_ENGINE_POWER)
            var impulse_y = -oy * Scalar[gpu_dtype](MAIN_ENGINE_POWER)
            vx = vx + impulse_x / Scalar[gpu_dtype](LANDER_MASS)
            vy = vy + impulse_y / Scalar[gpu_dtype](LANDER_MASS)
            var torque = ox * impulse_y - oy * impulse_x
            angular_vel = angular_vel + torque / Scalar[gpu_dtype](LANDER_INERTIA)

        elif action == 1:
            # Left engine
            s_power = Scalar[gpu_dtype](1.0)
            var direction = Scalar[gpu_dtype](-1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[gpu_dtype](3.0) * dispersion_y + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[gpu_dtype](3.0) * dispersion_y + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var impulse_x = -ox * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            var impulse_y = -oy * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            vx = vx + impulse_x / Scalar[gpu_dtype](LANDER_MASS)
            vy = vy + impulse_y / Scalar[gpu_dtype](LANDER_MASS)
            var r_x = ox - tip_x * Scalar[gpu_dtype](LANDER_HALF_HEIGHT)
            var r_y = oy + tip_y * Scalar[gpu_dtype](SIDE_ENGINE_HEIGHT)
            var torque = r_x * impulse_y - r_y * impulse_x
            angular_vel = angular_vel + torque / Scalar[gpu_dtype](LANDER_INERTIA)

        elif action == 3:
            # Right engine
            s_power = Scalar[gpu_dtype](1.0)
            var direction = Scalar[gpu_dtype](1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[gpu_dtype](3.0) * dispersion_y + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[gpu_dtype](3.0) * dispersion_y + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var impulse_x = -ox * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            var impulse_y = -oy * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            vx = vx + impulse_x / Scalar[gpu_dtype](LANDER_MASS)
            vy = vy + impulse_y / Scalar[gpu_dtype](LANDER_MASS)
            var r_x = ox - tip_x * Scalar[gpu_dtype](LANDER_HALF_HEIGHT)
            var r_y = oy + tip_y * Scalar[gpu_dtype](SIDE_ENGINE_HEIGHT)
            var torque = r_x * impulse_y - r_y * impulse_x
            angular_vel = angular_vel + torque / Scalar[gpu_dtype](LANDER_INERTIA)

        # =====================================================================
        # Semi-implicit Euler (matches Box2D order)
        # Step 1: Apply gravity to velocity
        # =====================================================================
        vy = vy + Scalar[gpu_dtype](GRAVITY * TAU)

        # =====================================================================
        # Step 2: Integrate position
        # =====================================================================
        x = x + vx * Scalar[gpu_dtype](TAU)
        y = y + vy * Scalar[gpu_dtype](TAU)
        angle = angle + angular_vel * Scalar[gpu_dtype](TAU)

        # Normalize angle to [-pi, pi]
        var two_pi = Scalar[gpu_dtype](2.0 * pi)
        var pi_val = Scalar[gpu_dtype](pi)
        if angle > Scalar[gpu_dtype](100.0) or angle < Scalar[gpu_dtype](-100.0):
            angle = Scalar[gpu_dtype](0.0)
        if angle > pi_val:
            angle = angle - two_pi
        if angle > pi_val:
            angle = angle - two_pi
        if angle < -pi_val:
            angle = angle + two_pi
        if angle < -pi_val:
            angle = angle + two_pi

        # =====================================================================
        # Step 3: Contact detection and response
        # =====================================================================
        var cos_angle = cos(angle)
        var sin_angle = sin(angle)
        var terrain_y = Scalar[gpu_dtype](HELIPAD_Y)

        # Compute leg attachment points (where joint connects to lander)
        # These are the TOP of the legs, attached to the lander body
        var left_attach_x = x - Scalar[gpu_dtype](LEG_AWAY) * cos_angle
        var left_attach_y = y - Scalar[gpu_dtype](LEG_AWAY) * sin_angle

        var right_attach_x = x + Scalar[gpu_dtype](LEG_AWAY) * cos_angle
        var right_attach_y = y + Scalar[gpu_dtype](LEG_AWAY) * sin_angle

        # Compute leg TIP positions (bottom of legs)
        # Legs extend downward (in lander's local frame) by LEG_DOWN
        # The leg box has half-height LEG_H, so the tip is at attachment - LEG_DOWN - LEG_H
        # But for simplicity, we use the full LEG_DOWN as the offset to the contact point
        var leg_length = Scalar[gpu_dtype](LEG_DOWN) + Scalar[gpu_dtype](LEG_H)

        var left_tip_x = left_attach_x - leg_length * sin_angle
        var left_tip_y = left_attach_y - leg_length * cos_angle

        var right_tip_x = right_attach_x - leg_length * sin_angle
        var right_tip_y = right_attach_y - leg_length * cos_angle

        # Contact detection: tip below terrain
        # Add a small threshold to account for leg width
        var contact_threshold = Scalar[gpu_dtype](LEG_W)

        left_contact = Scalar[gpu_dtype](0.0)
        right_contact = Scalar[gpu_dtype](0.0)

        var left_penetration = terrain_y - Scalar[gpu_dtype](left_tip_y[0])
        var right_penetration = terrain_y - Scalar[gpu_dtype](right_tip_y[0])

        if left_penetration > Scalar[gpu_dtype](0.0):
            left_contact = Scalar[gpu_dtype](1.0)

        if right_penetration > Scalar[gpu_dtype](0.0):
            right_contact = Scalar[gpu_dtype](1.0)

        # =====================================================================
        # Contact response (impulse-based, matches Box2D)
        # =====================================================================
        var any_contact = (left_contact > Scalar[gpu_dtype](0.5)) or (right_contact > Scalar[gpu_dtype](0.5))

        if any_contact:
            # Count contacts
            var num_contacts = Scalar[gpu_dtype](0.0)
            var total_penetration = Scalar[gpu_dtype](0.0)

            if left_contact > Scalar[gpu_dtype](0.5):
                num_contacts = num_contacts + Scalar[gpu_dtype](1.0)
                total_penetration = total_penetration + left_penetration
            if right_contact > Scalar[gpu_dtype](0.5):
                num_contacts = num_contacts + Scalar[gpu_dtype](1.0)
                total_penetration = total_penetration + right_penetration

            # Position correction (Baumgarte stabilization)
            # Box2D uses ~0.2 factor, applied over iterations
            var baumgarte = Scalar[gpu_dtype](0.2)
            var slop = Scalar[gpu_dtype](0.005)  # Penetration allowance
            var avg_penetration = total_penetration / num_contacts

            if avg_penetration > slop:
                y = y + baumgarte * (avg_penetration - slop)

            # Normal impulse (stop downward motion)
            # j = -(1 + e) * v_n * m for restitution e=0
            if vy < Scalar[gpu_dtype](0.0):
                # Apply normal impulse to stop penetration
                var normal_impulse = -vy * Scalar[gpu_dtype](LANDER_MASS)
                vy = Scalar[gpu_dtype](0.0)  # Stop vertical motion (restitution = 0)

                # Friction impulse (Coulomb friction)
                # |j_t| <= mu * |j_n|
                var max_friction_impulse = Scalar[gpu_dtype](CONTACT_FRICTION) * normal_impulse
                var friction_impulse = -vx * Scalar[gpu_dtype](LANDER_MASS)

                # Clamp friction
                if friction_impulse > max_friction_impulse:
                    friction_impulse = max_friction_impulse
                elif friction_impulse < -max_friction_impulse:
                    friction_impulse = -max_friction_impulse

                vx = vx + friction_impulse / Scalar[gpu_dtype](LANDER_MASS)

                # Angular damping from contact (simplified joint friction)
                var angular_damping = Scalar[gpu_dtype](0.95)
                angular_vel = angular_vel * angular_damping

        # =====================================================================
        # Velocity clamping
        # =====================================================================
        var max_vel = Scalar[gpu_dtype](100.0)
        var max_ang_vel = Scalar[gpu_dtype](20.0)
        if vx > max_vel:
            vx = max_vel
        if vx < -max_vel:
            vx = -max_vel
        if vy > max_vel:
            vy = max_vel
        if vy < -max_vel:
            vy = -max_vel
        if angular_vel > max_ang_vel:
            angular_vel = max_ang_vel
        if angular_vel < -max_ang_vel:
            angular_vel = -max_ang_vel

        # =====================================================================
        # Termination conditions
        # =====================================================================
        var x_norm = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](W_UNITS / 2.0)
        var terminated = False
        var crashed = False
        var landed = False

        # Out of bounds
        if x_norm >= Scalar[gpu_dtype](1.0) or x_norm <= Scalar[gpu_dtype](-1.0):
            terminated = True
            crashed = True

        # Lander body touches ground (crash if legs not down)
        var lander_bottom_y = y - Scalar[gpu_dtype](LANDER_HALF_HEIGHT) * cos_angle
        var both_legs = (left_contact > Scalar[gpu_dtype](0.5)) and (right_contact > Scalar[gpu_dtype](0.5))

        if lander_bottom_y <= terrain_y and not both_legs:
            terminated = True
            crashed = True

        # Successful landing
        if both_legs:
            var vx_abs = vx if vx >= Scalar[gpu_dtype](0.0) else -vx
            var vy_abs = vy if vy >= Scalar[gpu_dtype](0.0) else -vy
            var angvel_abs = angular_vel if angular_vel >= Scalar[gpu_dtype](0.0) else -angular_vel

            if (
                vx_abs < Scalar[gpu_dtype](0.5)
                and vy_abs < Scalar[gpu_dtype](0.5)
                and angvel_abs < Scalar[gpu_dtype](0.5)
            ):
                terminated = True
                landed = True

        # =====================================================================
        # Compute reward
        # =====================================================================
        var y_norm = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[gpu_dtype](H_UNITS / 2.0)
        var vx_norm = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy_norm = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))

        var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
        var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
        var angle_abs = angle if angle >= Scalar[gpu_dtype](0.0) else -angle

        var new_shaping = (
            Scalar[gpu_dtype](-100.0) * dist
            + Scalar[gpu_dtype](-100.0) * speed
            + Scalar[gpu_dtype](-100.0) * angle_abs
            + Scalar[gpu_dtype](10.0) * left_contact
            + Scalar[gpu_dtype](10.0) * right_contact
        )

        var reward = new_shaping - prev_shaping
        reward = reward - m_power * Scalar[gpu_dtype](MAIN_ENGINE_FUEL_COST)
        reward = reward - s_power * Scalar[gpu_dtype](SIDE_ENGINE_FUEL_COST)

        if crashed:
            reward = Scalar[gpu_dtype](-100.0)
        elif landed:
            reward = Scalar[gpu_dtype](100.0)

        # =====================================================================
        # NaN/Inf safety
        # =====================================================================
        var state_corrupted = False
        if x != x or y != y or vx != vx or vy != vy or angle != angle or angular_vel != angular_vel:
            state_corrupted = True
        var extreme = Scalar[gpu_dtype](1e6)
        if x > extreme or x < -extreme or y > extreme or y < -extreme:
            state_corrupted = True

        if state_corrupted:
            x = Scalar[gpu_dtype](INITIAL_X)
            y = Scalar[gpu_dtype](INITIAL_Y)
            vx = Scalar[gpu_dtype](0.0)
            vy = Scalar[gpu_dtype](0.0)
            angle = Scalar[gpu_dtype](0.0)
            angular_vel = Scalar[gpu_dtype](0.0)
            left_contact = Scalar[gpu_dtype](0.0)
            right_contact = Scalar[gpu_dtype](0.0)
            reward = Scalar[gpu_dtype](-100.0)
            terminated = True

        # =====================================================================
        # Write back normalized state
        # =====================================================================
        states[i, 0] = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](W_UNITS / 2.0)
        states[i, 1] = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[gpu_dtype](H_UNITS / 2.0)
        states[i, 2] = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        states[i, 3] = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        states[i, 4] = angle
        states[i, 5] = angular_vel * Scalar[gpu_dtype](20.0 / Float64(FPS))
        states[i, 6] = left_contact
        states[i, 7] = right_contact

        rewards[i] = reward
        dones[i] = Scalar[gpu_dtype](1.0) if terminated else Scalar[gpu_dtype](0.0)

    @staticmethod
    @always_inline
    fn reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
    ):
        """GPU kernel to reset all environments."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var rng = xorshift32(Scalar[DType.uint32](i * 2654435761 + 12345))

        # Initial position
        var x = Scalar[gpu_dtype](INITIAL_X)
        var y = Scalar[gpu_dtype](INITIAL_Y)

        # Random initial velocity (matches CPU's initial random force effect)
        var result_vx = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-4.0), Scalar[gpu_dtype](4.0)
        )
        var vx = result_vx[0]
        rng = result_vx[1]

        var result_vy = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-4.0), Scalar[gpu_dtype](4.0)
        )
        var vy = result_vy[0]
        rng = result_vy[1]

        var result_angle = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.1), Scalar[gpu_dtype](0.1)
        )
        var angle = result_angle[0]
        rng = result_angle[1]

        var result_angvel = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.5), Scalar[gpu_dtype](0.5)
        )
        var angular_vel = result_angvel[0]

        # Write normalized state
        states[i, 0] = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](W_UNITS / 2.0)
        states[i, 1] = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[gpu_dtype](H_UNITS / 2.0)
        states[i, 2] = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        states[i, 3] = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        states[i, 4] = angle
        states[i, 5] = angular_vel * Scalar[gpu_dtype](20.0 / Float64(FPS))
        states[i, 6] = Scalar[gpu_dtype](0.0)  # left_contact
        states[i, 7] = Scalar[gpu_dtype](0.0)  # right_contact

    @staticmethod
    @always_inline
    fn selective_reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        rng_seed: Scalar[DType.uint32],
    ):
        """GPU kernel to reset only done environments."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        if dones[i] < Scalar[gpu_dtype](0.5):
            return

        var rng = xorshift32(Scalar[DType.uint32](i * 2654435761) + rng_seed)

        var x = Scalar[gpu_dtype](INITIAL_X)
        var y = Scalar[gpu_dtype](INITIAL_Y)

        var result_vx = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-4.0), Scalar[gpu_dtype](4.0)
        )
        var vx = result_vx[0]
        rng = result_vx[1]

        var result_vy = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-4.0), Scalar[gpu_dtype](4.0)
        )
        var vy = result_vy[0]
        rng = result_vy[1]

        var result_angle = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.1), Scalar[gpu_dtype](0.1)
        )
        var angle = result_angle[0]
        rng = result_angle[1]

        var result_angvel = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.5), Scalar[gpu_dtype](0.5)
        )
        var angular_vel = result_angvel[0]

        states[i, 0] = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](W_UNITS / 2.0)
        states[i, 1] = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[gpu_dtype](H_UNITS / 2.0)
        states[i, 2] = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        states[i, 3] = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        states[i, 4] = angle
        states[i, 5] = angular_vel * Scalar[gpu_dtype](20.0 / Float64(FPS))
        states[i, 6] = Scalar[gpu_dtype](0.0)
        states[i, 7] = Scalar[gpu_dtype](0.0)

        dones[i] = Scalar[gpu_dtype](0.0)

    # =========================================================================
    # Host Launcher Methods
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Launch step kernel on GPU."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint64](rng_seed)

        @always_inline
        fn step_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            rewards: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            rng_seed: Scalar[DType.uint64],
        ):
            Self.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones, rng_seed
            )

        ctx.enqueue_function[step_wrapper, step_wrapper](
            states,
            actions,
            rewards,
            dones,
            seed,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[gpu_dtype]) raises:
        """Launch reset kernel on GPU."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            Self.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt32,
    ) raises:
        """Launch selective reset kernel on GPU."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint32](rng_seed)

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            rng_seed: Scalar[DType.uint32],
        ):
            Self.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                states, dones, rng_seed
            )

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states,
            dones,
            seed,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )
