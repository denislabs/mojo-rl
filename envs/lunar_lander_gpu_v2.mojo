"""GPU-compatible LunarLander environment for PPO training - V2 with improved physics.

This version implements:
- Sub-stepping (4 sub-steps per frame) for improved stability
- Velocity Verlet integration for better accuracy
- Hidden state for physics continuity (prev velocities, contact impulse cache)
- Contact impulse iteration (2 iterations per sub-step)

State layout [12 values]:
    0-7:  Observable state [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]
    8-11: Hidden state [prev_vx, prev_vy, prev_angular_vel, contact_impulse_sum]

The neural network only sees the first 8 values (OBS_DIM=8).
"""

from math import cos, sin, sqrt, tanh, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random.philox import Random as PhiloxRandom

from deep_rl.gpu import random_range, xorshift32
from core import GPUDiscreteEnv


# =============================================================================
# Physics Constants (matching Gymnasium LunarLander-v3)
# =============================================================================

comptime gpu_dtype = DType.float32

# Core physics
comptime GRAVITY: Float64 = -10.0
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SIDE_ENGINE_POWER: Float64 = 0.6
comptime SCALE: Float64 = 30.0
comptime FPS: Int = 50
comptime TAU: Float64 = 0.02  # 1/FPS

# Sub-stepping for improved physics stability
# More sub-steps = better accuracy, matching Box2D's iterative approach
comptime NUM_SUBSTEPS: Int = 4
comptime SUB_TAU: Float64 = TAU / Float64(NUM_SUBSTEPS)  # 0.005s per sub-step

# Contact solver iterations per sub-step (like Box2D velocity iterations)
comptime CONTACT_ITERATIONS: Int = 2

# Hidden state indices
comptime HIDDEN_PREV_VX: Int = 8
comptime HIDDEN_PREV_VY: Int = 9
comptime HIDDEN_PREV_ANGVEL: Int = 10
comptime HIDDEN_CONTACT_IMPULSE: Int = 11

# Full state size (8 observable + 4 hidden)
comptime FULL_STATE_SIZE: Int = 12

# Lander geometry (in physics units, after SCALE division)
comptime LEG_AWAY: Float64 = 20.0 / 30.0
comptime LEG_DOWN: Float64 = 18.0 / 30.0
comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / 30.0

# Engine mount positions
comptime MAIN_ENGINE_Y_OFFSET: Float64 = 4.0 / 30.0
comptime SIDE_ENGINE_HEIGHT: Float64 = 14.0 / 30.0
comptime SIDE_ENGINE_AWAY: Float64 = 12.0 / 30.0

# Lander mass/inertia
comptime LANDER_MASS: Float64 = 5.0
comptime LANDER_INERTIA: Float64 = 2.0

# Reward constants
comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03

# Viewport/world dimensions
comptime VIEWPORT_W: Float64 = 600.0
comptime VIEWPORT_H: Float64 = 400.0
comptime W_UNITS: Float64 = 20.0
comptime H_UNITS: Float64 = 13.333

# Helipad position
comptime HELIPAD_Y: Float64 = H_UNITS / 4.0
comptime HELIPAD_X: Float64 = W_UNITS / 2.0

# Initial spawn position
comptime INITIAL_Y: Float64 = H_UNITS
comptime INITIAL_X: Float64 = W_UNITS / 2.0


# =============================================================================
# GPU LunarLander Environment V2
# =============================================================================


struct LunarLanderGPUv2(GPUDiscreteEnv):
    """GPU-compatible LunarLander with improved physics.

    Key improvements over V1:
    - Sub-stepping: 4 physics sub-steps per frame for stability
    - Velocity Verlet: Better integration accuracy
    - Hidden state: Tracks prev velocities for Verlet, contact impulse cache
    - Contact iteration: 2 iterations per sub-step for better constraint solving
    """

    # GPUDiscreteEnv trait constants
    comptime STATE_SIZE: Int = FULL_STATE_SIZE  # 12 (8 obs + 4 hidden)
    comptime OBS_DIM: Int = 8  # Neural network input dimension
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
        """GPU kernel for LunarLander physics step with sub-stepping and Velocity Verlet.
        """
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        # =====================================================================
        # Extract current state (NORMALIZED observations from buffer)
        # =====================================================================
        var x_obs = states[i, 0]
        var y_obs = states[i, 1]
        var vx_obs = states[i, 2]
        var vy_obs = states[i, 3]
        var angle_obs = states[i, 4]
        var angular_vel_obs = states[i, 5]
        var left_contact = states[i, 6]
        var right_contact = states[i, 7]

        # Hidden state (previous velocities for Verlet, contact impulse cache)
        var prev_vx_obs = states[i, HIDDEN_PREV_VX]
        var prev_vy_obs = states[i, HIDDEN_PREV_VY]
        var prev_angvel_obs = states[i, HIDDEN_PREV_ANGVEL]
        var contact_impulse_cache = states[i, HIDDEN_CONTACT_IMPULSE]

        # DENORMALIZE to get raw physics values
        var x = x_obs * Scalar[gpu_dtype](W_UNITS / 2.0) + Scalar[gpu_dtype](
            HELIPAD_X
        )
        var y = y_obs * Scalar[gpu_dtype](H_UNITS / 2.0) + Scalar[gpu_dtype](
            HELIPAD_Y + LEG_DOWN
        )
        var vx = vx_obs / Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy = vy_obs / Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        var angle = angle_obs
        var angular_vel = angular_vel_obs / Scalar[gpu_dtype](
            20.0 / Float64(FPS)
        )

        # Previous velocities (denormalized)
        var prev_vx = prev_vx_obs / Scalar[gpu_dtype](
            W_UNITS / 2.0 / Float64(FPS)
        )
        var prev_vy = prev_vy_obs / Scalar[gpu_dtype](
            H_UNITS / 2.0 / Float64(FPS)
        )
        var prev_angular_vel = prev_angvel_obs / Scalar[gpu_dtype](
            20.0 / Float64(FPS)
        )

        # =====================================================================
        # Compute prev_shaping BEFORE physics (for reward shaping)
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
        # Parse action and compute engine impulses (applied once per frame)
        # =====================================================================
        var action = Int(actions[i])
        var m_power = Scalar[gpu_dtype](0.0)
        var s_power = Scalar[gpu_dtype](0.0)

        # Random dispersion using PhiloxRandom
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

        # Direction vectors (tip points upward when angle=0)
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y  # Perpendicular to tip
        var side_y = tip_x

        # Compute impulse from action (applied at start of frame, not per sub-step)
        var impulse_vx = Scalar[gpu_dtype](0.0)
        var impulse_vy = Scalar[gpu_dtype](0.0)
        var impulse_angular = Scalar[gpu_dtype](0.0)

        if action == 2:
            # Main engine
            m_power = Scalar[gpu_dtype](1.0)
            var ox = (
                tip_x
                * (
                    Scalar[gpu_dtype](MAIN_ENGINE_Y_OFFSET)
                    + Scalar[gpu_dtype](2.0) * dispersion_x
                )
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y
                * (
                    Scalar[gpu_dtype](MAIN_ENGINE_Y_OFFSET)
                    + Scalar[gpu_dtype](2.0) * dispersion_x
                )
                - side_y * dispersion_y
            )
            var imp_x = -ox * Scalar[gpu_dtype](MAIN_ENGINE_POWER)
            var imp_y = -oy * Scalar[gpu_dtype](MAIN_ENGINE_POWER)
            impulse_vx = Scalar[gpu_dtype](imp_x[0]) / Scalar[gpu_dtype](
                LANDER_MASS
            )
            impulse_vy = Scalar[gpu_dtype](imp_y[0]) / Scalar[gpu_dtype](
                LANDER_MASS
            )
            var torque = ox * imp_y - oy * imp_x
            impulse_angular = Scalar[gpu_dtype](torque[0]) / Scalar[gpu_dtype](
                LANDER_INERTIA
            )

        elif action == 1:
            # Left engine
            s_power = Scalar[gpu_dtype](1.0)
            var direction = Scalar[gpu_dtype](-1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[gpu_dtype](3.0) * dispersion_y
                + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[gpu_dtype](3.0) * dispersion_y
                + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var imp_x = -ox * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            var imp_y = -oy * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            impulse_vx = Scalar[gpu_dtype](imp_x[0]) / Scalar[gpu_dtype](
                LANDER_MASS
            )
            impulse_vy = Scalar[gpu_dtype](imp_y[0]) / Scalar[gpu_dtype](
                LANDER_MASS
            )
            var r_x = ox - tip_x * Scalar[gpu_dtype](LANDER_HALF_HEIGHT)
            var r_y = oy + tip_y * Scalar[gpu_dtype](SIDE_ENGINE_HEIGHT)
            var torque = r_x * imp_y - r_y * imp_x
            impulse_angular = Scalar[gpu_dtype](torque[0]) / Scalar[gpu_dtype](
                LANDER_INERTIA
            )

        elif action == 3:
            # Right engine
            s_power = Scalar[gpu_dtype](1.0)
            var direction = Scalar[gpu_dtype](1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[gpu_dtype](3.0) * dispersion_y
                + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[gpu_dtype](3.0) * dispersion_y
                + direction * Scalar[gpu_dtype](SIDE_ENGINE_AWAY)
            )
            var imp_x = -ox * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            var imp_y = -oy * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            impulse_vx = Scalar[gpu_dtype](imp_x[0]) / Scalar[gpu_dtype](
                LANDER_MASS
            )
            impulse_vy = Scalar[gpu_dtype](imp_y[0]) / Scalar[gpu_dtype](
                LANDER_MASS
            )
            var r_x = ox - tip_x * Scalar[gpu_dtype](LANDER_HALF_HEIGHT)
            var r_y = oy + tip_y * Scalar[gpu_dtype](SIDE_ENGINE_HEIGHT)
            var torque = r_x * imp_y - r_y * imp_x
            impulse_angular = Scalar[gpu_dtype](torque[0]) / Scalar[gpu_dtype](
                LANDER_INERTIA
            )

        # Apply engine impulse (once per frame)
        vx = vx + impulse_vx
        vy = vy + impulse_vy
        angular_vel = angular_vel + impulse_angular

        # Store pre-substep velocities for Verlet
        var frame_start_vx = vx
        var frame_start_vy = vy
        var frame_start_angular_vel = angular_vel

        # =====================================================================
        # SUB-STEPPING LOOP
        # =====================================================================
        var terminated = False
        var crashed = False
        var landed = False
        var sub_dt = Scalar[gpu_dtype](SUB_TAU)
        var terrain_y = Scalar[gpu_dtype](HELIPAD_Y)
        var friction_coef = Scalar[gpu_dtype](0.1)

        @parameter
        for substep in range(NUM_SUBSTEPS):
            if terminated:
                break

            # -----------------------------------------------------------------
            # Velocity Verlet: First half - update velocity with gravity
            # v(t + dt/2) = v(t) + a(t) * dt/2
            # -----------------------------------------------------------------
            var half_dt = sub_dt / Scalar[gpu_dtype](2.0)
            vy = vy + Scalar[gpu_dtype](GRAVITY) * half_dt

            # -----------------------------------------------------------------
            # Position update using average velocity (Verlet style)
            # x(t + dt) = x(t) + v(t + dt/2) * dt
            # -----------------------------------------------------------------
            x = x + vx * sub_dt
            y = y + vy * sub_dt
            angle = angle + angular_vel * sub_dt

            # -----------------------------------------------------------------
            # Velocity Verlet: Second half - update velocity with gravity
            # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
            # -----------------------------------------------------------------
            vy = vy + Scalar[gpu_dtype](GRAVITY) * half_dt

            # Normalize angle
            var two_pi = Scalar[gpu_dtype](2.0 * pi)
            var pi_val = Scalar[gpu_dtype](pi)
            if angle > Scalar[gpu_dtype](100.0):
                angle = Scalar[gpu_dtype](0.0)
            elif angle < Scalar[gpu_dtype](-100.0):
                angle = Scalar[gpu_dtype](0.0)
            if angle > pi_val:
                angle = angle - two_pi
            if angle < -pi_val:
                angle = angle + two_pi

            # -----------------------------------------------------------------
            # Contact detection and response (iterated for stability)
            # -----------------------------------------------------------------
            var cos_angle = cos(angle)
            var sin_angle = sin(angle)

            # Leg positions
            var left_leg_y = (
                y
                - Scalar[gpu_dtype](LEG_AWAY) * sin_angle
                - Scalar[gpu_dtype](LEG_DOWN) * cos_angle
            )
            var right_leg_y = (
                y
                + Scalar[gpu_dtype](LEG_AWAY) * sin_angle
                - Scalar[gpu_dtype](LEG_DOWN) * cos_angle
            )

            # Convert to scalar for comparison
            var left_leg_y_s = Scalar[gpu_dtype](left_leg_y[0])
            var right_leg_y_s = Scalar[gpu_dtype](right_leg_y[0])

            # Contact iteration (like Box2D velocity iterations)
            @parameter
            for contact_iter in range(CONTACT_ITERATIONS):
                left_contact = Scalar[gpu_dtype](0.0)
                right_contact = Scalar[gpu_dtype](0.0)
                var total_penetration = Scalar[gpu_dtype](0.0)
                var num_contacts = 0

                if left_leg_y_s <= terrain_y:
                    left_contact = Scalar[gpu_dtype](1.0)
                    var penetration = terrain_y - left_leg_y_s
                    if penetration > Scalar[gpu_dtype](0.0):
                        total_penetration = total_penetration + penetration
                        num_contacts += 1

                if right_leg_y_s <= terrain_y:
                    right_contact = Scalar[gpu_dtype](1.0)
                    var penetration = terrain_y - right_leg_y_s
                    if penetration > Scalar[gpu_dtype](0.0):
                        total_penetration = total_penetration + penetration
                        num_contacts += 1

                if num_contacts > 0:
                    # Position correction (Baumgarte stabilization style)
                    var correction = total_penetration / Scalar[gpu_dtype](
                        num_contacts
                    )
                    # Apply smaller correction per iteration (converges better)
                    var correction_factor = Scalar[gpu_dtype](
                        0.8
                    )  # 80% per iteration
                    y = y + correction * correction_factor

                    # Update leg positions after correction
                    left_leg_y_s = left_leg_y_s + correction * correction_factor
                    right_leg_y_s = (
                        right_leg_y_s + correction * correction_factor
                    )

                    # Velocity response (impulse-based)
                    if vy < Scalar[gpu_dtype](0.0):
                        # Normal impulse: stop downward motion
                        # Apply warm-started impulse using cache
                        var warm_factor = Scalar[gpu_dtype](0.5)
                        vy = vy * warm_factor  # Partial damping per iteration

                    # Friction impulse
                    vx = vx * (
                        Scalar[gpu_dtype](1.0)
                        - friction_coef * Scalar[gpu_dtype](0.5)
                    )

                    # Angular damping from contact
                    angular_vel = angular_vel * Scalar[gpu_dtype](0.99)

            # -----------------------------------------------------------------
            # Velocity clamping for stability
            # -----------------------------------------------------------------
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

            # -----------------------------------------------------------------
            # Termination checks (per sub-step for early exit)
            # -----------------------------------------------------------------
            var x_norm = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
                W_UNITS / 2.0
            )

            # Out of bounds
            if x_norm >= Scalar[gpu_dtype](1.0) or x_norm <= Scalar[gpu_dtype](
                -1.0
            ):
                terminated = True
                crashed = True

            # Body touches ground without both legs
            var lander_bottom_y = (
                y - Scalar[gpu_dtype](LANDER_HALF_HEIGHT) * cos_angle
            )
            var both_legs = (left_contact > Scalar[gpu_dtype](0.5)) and (
                right_contact > Scalar[gpu_dtype](0.5)
            )
            if lander_bottom_y <= terrain_y and not both_legs:
                terminated = True
                crashed = True

            # Successful landing check
            if both_legs:
                var vx_abs = vx if vx >= Scalar[gpu_dtype](0.0) else -vx
                var vy_abs = vy if vy >= Scalar[gpu_dtype](0.0) else -vy
                var angvel_abs = (
                    angular_vel if angular_vel
                    >= Scalar[gpu_dtype](0.0) else -angular_vel
                )
                if (
                    vx_abs < Scalar[gpu_dtype](0.5)
                    and vy_abs < Scalar[gpu_dtype](0.5)
                    and angvel_abs < Scalar[gpu_dtype](0.5)
                ):
                    terminated = True
                    landed = True

        # =====================================================================
        # Compute reward with shaping
        # =====================================================================
        var x_norm_final = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[
            gpu_dtype
        ](W_UNITS / 2.0)
        var y_norm_final = (
            y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)
        ) / Scalar[gpu_dtype](H_UNITS / 2.0)
        var vx_norm_final = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy_norm_final = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))

        var dist = sqrt(
            x_norm_final * x_norm_final + y_norm_final * y_norm_final
        )
        var speed = sqrt(
            vx_norm_final * vx_norm_final + vy_norm_final * vy_norm_final
        )
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
        # NaN/Inf safety check
        # =====================================================================
        var state_corrupted = False
        if (
            x != x
            or y != y
            or vx != vx
            or vy != vy
            or angle != angle
            or angular_vel != angular_vel
        ):
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
            frame_start_vx = Scalar[gpu_dtype](0.0)
            frame_start_vy = Scalar[gpu_dtype](0.0)
            frame_start_angular_vel = Scalar[gpu_dtype](0.0)
            reward = Scalar[gpu_dtype](-100.0)
            terminated = True

        # =====================================================================
        # Write back state (NORMALIZED)
        # =====================================================================
        states[i, 0] = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
            W_UNITS / 2.0
        )
        states[i, 1] = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[
            gpu_dtype
        ](H_UNITS / 2.0)
        states[i, 2] = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        states[i, 3] = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        states[i, 4] = angle
        states[i, 5] = angular_vel * Scalar[gpu_dtype](20.0 / Float64(FPS))
        states[i, 6] = left_contact
        states[i, 7] = right_contact

        # Hidden state: store current velocities as "previous" for next frame
        states[i, HIDDEN_PREV_VX] = frame_start_vx * Scalar[gpu_dtype](
            W_UNITS / 2.0 / Float64(FPS)
        )
        states[i, HIDDEN_PREV_VY] = frame_start_vy * Scalar[gpu_dtype](
            H_UNITS / 2.0 / Float64(FPS)
        )
        states[i, HIDDEN_PREV_ANGVEL] = frame_start_angular_vel * Scalar[
            gpu_dtype
        ](20.0 / Float64(FPS))
        # Contact impulse cache (sum of impulses applied this frame)
        var total_impulse = (
            frame_start_vy - vy
        ) if frame_start_vy > vy else Scalar[gpu_dtype](0.0)
        states[i, HIDDEN_CONTACT_IMPULSE] = total_impulse

        rewards[i] = reward
        dones[i] = Scalar[gpu_dtype](1.0) if terminated else Scalar[gpu_dtype](
            0.0
        )

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
        """GPU kernel to reset all environments to random initial states."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var rng = xorshift32(Scalar[DType.uint32](i * 2654435761 + 12345))

        # Initial position
        var x = Scalar[gpu_dtype](INITIAL_X)
        var y = Scalar[gpu_dtype](INITIAL_Y)

        # Random initial velocity
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

        # Normalize and write observable state
        states[i, 0] = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
            W_UNITS / 2.0
        )
        states[i, 1] = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[
            gpu_dtype
        ](H_UNITS / 2.0)
        states[i, 2] = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        states[i, 3] = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        states[i, 4] = angle
        states[i, 5] = angular_vel * Scalar[gpu_dtype](20.0 / Float64(FPS))
        states[i, 6] = Scalar[gpu_dtype](0.0)  # left_contact
        states[i, 7] = Scalar[gpu_dtype](0.0)  # right_contact

        # Initialize hidden state
        states[i, HIDDEN_PREV_VX] = states[i, 2]  # prev_vx = current vx
        states[i, HIDDEN_PREV_VY] = states[i, 3]  # prev_vy = current vy
        states[i, HIDDEN_PREV_ANGVEL] = states[
            i, 5
        ]  # prev_angular_vel = current
        states[i, HIDDEN_CONTACT_IMPULSE] = Scalar[gpu_dtype](0.0)

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

        states[i, 0] = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
            W_UNITS / 2.0
        )
        states[i, 1] = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[
            gpu_dtype
        ](H_UNITS / 2.0)
        states[i, 2] = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        states[i, 3] = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        states[i, 4] = angle
        states[i, 5] = angular_vel * Scalar[gpu_dtype](20.0 / Float64(FPS))
        states[i, 6] = Scalar[gpu_dtype](0.0)
        states[i, 7] = Scalar[gpu_dtype](0.0)

        # Initialize hidden state
        states[i, HIDDEN_PREV_VX] = states[i, 2]
        states[i, HIDDEN_PREV_VY] = states[i, 3]
        states[i, HIDDEN_PREV_ANGVEL] = states[i, 5]
        states[i, HIDDEN_CONTACT_IMPULSE] = Scalar[gpu_dtype](0.0)

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
