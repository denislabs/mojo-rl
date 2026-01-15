"""GPU-compatible LunarLander environment for PPO training.

This is a simplified physics implementation optimized for GPU parallelism:
- Single rigid body dynamics (no joints/constraints)
- Implicit leg contact detection
- Fixed flat helipad terrain
- Sinusoidal wind/turbulence effects

State: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
Actions: 0=nop, 1=left_engine, 2=main_engine, 3=right_engine

Usage:
    from envs.lunar_lander_gpu import LunarLanderGPU
    from deep_agents.ppo import DeepPPOAgent

    var agent = DeepPPOAgent[8, 4, 128, 128, 256, 512]()
    var metrics = agent.train_gpu[LunarLanderGPU](ctx, num_episodes=50000)
"""

from math import cos, sin, sqrt, tanh, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

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

# Lander geometry (in physics units, after SCALE division)
comptime LEG_AWAY: Float64 = 20.0 / 30.0  # ~0.667 - horizontal distance from center
comptime LEG_DOWN: Float64 = 18.0 / 30.0  # ~0.6 - vertical distance below center
comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / 30.0  # ~0.567 - from center to top/bottom

# Engine mount positions
comptime MAIN_ENGINE_Y_OFFSET: Float64 = 4.0 / 30.0  # Below center
comptime SIDE_ENGINE_HEIGHT: Float64 = 14.0 / 30.0  # Height of side engines
comptime SIDE_ENGINE_AWAY: Float64 = 12.0 / 30.0  # Horizontal distance

# Lander mass/inertia (approximated from original Box2D setup)
# Original: density=5.0 on a ~1.1x0.9 unit polygon
comptime LANDER_MASS: Float64 = 5.0  # Approximate mass
comptime LANDER_INERTIA: Float64 = 2.0  # Approximate moment of inertia

# Reward constants
comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03

# Wind constants
comptime WIND_POWER: Float64 = 15.0
comptime TURBULENCE_POWER: Float64 = 1.5

# Viewport/world dimensions (for normalization)
comptime VIEWPORT_W: Float64 = 600.0
comptime VIEWPORT_H: Float64 = 400.0
comptime W_UNITS: Float64 = 20.0  # VIEWPORT_W / SCALE
comptime H_UNITS: Float64 = 13.333  # VIEWPORT_H / SCALE

# Helipad position
comptime HELIPAD_Y: Float64 = H_UNITS / 4.0  # ~3.33 units
comptime HELIPAD_X: Float64 = W_UNITS / 2.0  # Center of viewport ~10 units

# Initial spawn position
comptime INITIAL_Y: Float64 = H_UNITS  # Top of viewport ~13.33 units
comptime INITIAL_X: Float64 = W_UNITS / 2.0  # Center

# Initial random force (for variety)
comptime INITIAL_RANDOM: Float64 = 1000.0

# Termination thresholds
comptime MAX_STEPS: Int = 1000

# =============================================================================
# GPU LunarLander Environment
# =============================================================================


struct LunarLanderGPU(GPUDiscreteEnv):
    """GPU-compatible LunarLander with simplified physics.

    Implements GPUDiscreteEnv trait for use with PPO GPU training.

    State layout [8 values]:
        0: x position (world units, ~0-20)
        1: y position (world units, ~0-13.33)
        2: vx velocity
        3: vy velocity
        4: angle (radians)
        5: angular velocity
        6: left leg contact (0.0 or 1.0)
        7: right leg contact (0.0 or 1.0)

    Actions:
        0: No-op
        1: Fire left engine (pushes right)
        2: Fire main engine (pushes up)
        3: Fire right engine (pushes left)
    """

    # GPUDiscreteEnv trait constants
    comptime STATE_SIZE: Int = 8
    comptime OBS_DIM: Int = 8
    comptime NUM_ACTIONS: Int = 4
    comptime TPB: Int = 256

    # =========================================================================
    # GPU Kernels (Static Inline Methods)
    # =========================================================================

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
    ):
        """GPU kernel for LunarLander physics step."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        # Extract current state (NORMALIZED observations from buffer)
        var x_obs = states[i, 0]
        var y_obs = states[i, 1]
        var vx_obs = states[i, 2]
        var vy_obs = states[i, 3]
        var angle_obs = states[i, 4]
        var angular_vel_obs = states[i, 5]
        var left_contact = states[i, 6]
        var right_contact = states[i, 7]

        # DENORMALIZE to get raw physics values
        var x = x_obs * Scalar[gpu_dtype](W_UNITS / 2.0) + Scalar[gpu_dtype](
            HELIPAD_X
        )
        var y = y_obs * Scalar[gpu_dtype](H_UNITS / 2.0) + Scalar[gpu_dtype](
            HELIPAD_Y + LEG_DOWN
        )
        var vx = vx_obs / Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy = vy_obs / Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        var angle = angle_obs * Scalar[gpu_dtype](pi)
        var angular_vel = angular_vel_obs / Scalar[gpu_dtype](
            20.0 / Float64(FPS)
        )

        # =====================================================================
        # Compute prev_shaping BEFORE physics (for reward shaping)
        # Already have normalized values from state
        # =====================================================================
        var x_norm_prev = x_obs
        var y_norm_prev = y_obs
        var vx_norm_prev = vx_obs
        var vy_norm_prev = vy_obs

        var dist_prev = sqrt(x_norm_prev * x_norm_prev + y_norm_prev * y_norm_prev)
        var speed_prev = sqrt(
            vx_norm_prev * vx_norm_prev + vy_norm_prev * vy_norm_prev
        )
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
        # Apply wind (only when not in contact with ground)
        # =====================================================================
        var in_contact = (left_contact > Scalar[gpu_dtype](0.5)) or (
            right_contact > Scalar[gpu_dtype](0.5)
        )
        if not in_contact:
            # Wind index based on thread for variation
            var wind_idx = Scalar[gpu_dtype](i * 7919)
            var wind_mag = tanh(
                sin(Scalar[gpu_dtype](0.02) * wind_idx)
                + sin(Scalar[gpu_dtype](pi * 0.01) * wind_idx)
            ) * Scalar[gpu_dtype](WIND_POWER)

            # Apply wind as acceleration
            vx = vx + wind_mag / Scalar[gpu_dtype](LANDER_MASS) * Scalar[gpu_dtype](
                TAU
            )

            # Turbulence torque
            var torque_mag = tanh(
                sin(Scalar[gpu_dtype](0.02) * wind_idx + Scalar[gpu_dtype](1000.0))
                + sin(
                    Scalar[gpu_dtype](pi * 0.01) * wind_idx
                    + Scalar[gpu_dtype](1000.0)
                )
            ) * Scalar[gpu_dtype](TURBULENCE_POWER)
            angular_vel = angular_vel + torque_mag / Scalar[gpu_dtype](
                LANDER_INERTIA
            ) * Scalar[gpu_dtype](TAU)

        # =====================================================================
        # Parse action and apply engine forces
        # =====================================================================
        var action = Int(actions[i])
        var m_power = Scalar[gpu_dtype](0.0)
        var s_power = Scalar[gpu_dtype](0.0)

        # Direction vectors (tip points upward when angle=0)
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y  # Perpendicular to tip
        var side_y = tip_x

        if action == 2:
            # Main engine - fire downward (thrust upward)
            m_power = Scalar[gpu_dtype](1.0)

            # Impulse applied at engine location (below center)
            var impulse_x = tip_x * Scalar[gpu_dtype](MAIN_ENGINE_POWER)
            var impulse_y = tip_y * Scalar[gpu_dtype](MAIN_ENGINE_POWER)

            # Apply as velocity change (impulse / mass)
            vx = vx + impulse_x / Scalar[gpu_dtype](LANDER_MASS)
            vy = vy + impulse_y / Scalar[gpu_dtype](LANDER_MASS)

            # Angular effect from off-center thrust
            # Torque = r × F (simplified: engine offset creates rotation)
            var engine_offset = Scalar[gpu_dtype](MAIN_ENGINE_Y_OFFSET)
            var torque = -engine_offset * impulse_x * Scalar[gpu_dtype](0.1)
            angular_vel = angular_vel + torque / Scalar[gpu_dtype](LANDER_INERTIA)

        elif action == 1:
            # Left engine - fire left (pushes lander right, rotates CCW)
            s_power = Scalar[gpu_dtype](1.0)
            var direction = Scalar[gpu_dtype](-1.0)

            var impulse_x = side_x * direction * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            var impulse_y = side_y * direction * Scalar[gpu_dtype](SIDE_ENGINE_POWER)

            vx = vx + impulse_x / Scalar[gpu_dtype](LANDER_MASS)
            vy = vy + impulse_y / Scalar[gpu_dtype](LANDER_MASS)

            # Side engines create more torque (lever arm from engine position)
            var torque = direction * Scalar[gpu_dtype](
                SIDE_ENGINE_POWER * SIDE_ENGINE_HEIGHT
            )
            angular_vel = angular_vel + torque / Scalar[gpu_dtype](LANDER_INERTIA)

        elif action == 3:
            # Right engine - fire right (pushes lander left, rotates CW)
            s_power = Scalar[gpu_dtype](1.0)
            var direction = Scalar[gpu_dtype](1.0)

            var impulse_x = side_x * direction * Scalar[gpu_dtype](SIDE_ENGINE_POWER)
            var impulse_y = side_y * direction * Scalar[gpu_dtype](SIDE_ENGINE_POWER)

            vx = vx + impulse_x / Scalar[gpu_dtype](LANDER_MASS)
            vy = vy + impulse_y / Scalar[gpu_dtype](LANDER_MASS)

            var torque = direction * Scalar[gpu_dtype](
                SIDE_ENGINE_POWER * SIDE_ENGINE_HEIGHT
            )
            angular_vel = angular_vel + torque / Scalar[gpu_dtype](LANDER_INERTIA)

        # =====================================================================
        # Apply gravity
        # =====================================================================
        vy = vy + Scalar[gpu_dtype](GRAVITY * TAU)

        # =====================================================================
        # Clamp velocities for numerical stability
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
        # Euler integration
        # =====================================================================
        x = x + vx * Scalar[gpu_dtype](TAU)
        y = y + vy * Scalar[gpu_dtype](TAU)
        angle = angle + angular_vel * Scalar[gpu_dtype](TAU)

        # Normalize angle to [-pi, pi] - bounded approach (max 4 iterations, handles most cases)
        var two_pi = Scalar[gpu_dtype](2.0 * pi)
        var pi_val = Scalar[gpu_dtype](pi)
        # Clamp extreme angles first (prevents any possibility of infinite loop or NaN)
        if angle > Scalar[gpu_dtype](100.0):
            angle = Scalar[gpu_dtype](0.0)
        elif angle < Scalar[gpu_dtype](-100.0):
            angle = Scalar[gpu_dtype](0.0)
        # Now safely normalize with bounded iterations
        if angle > pi_val:
            angle = angle - two_pi
        if angle > pi_val:
            angle = angle - two_pi
        if angle < -pi_val:
            angle = angle + two_pi
        if angle < -pi_val:
            angle = angle + two_pi

        # =====================================================================
        # Compute leg positions and detect contact
        # =====================================================================
        var cos_angle = cos(angle)
        var sin_angle = sin(angle)

        # Left leg tip position
        var left_leg_x = x - Scalar[gpu_dtype](LEG_AWAY) * cos_angle + Scalar[
            gpu_dtype
        ](LEG_DOWN) * sin_angle
        var left_leg_y = y - Scalar[gpu_dtype](LEG_AWAY) * sin_angle - Scalar[
            gpu_dtype
        ](LEG_DOWN) * cos_angle

        # Right leg tip position
        var right_leg_x = x + Scalar[gpu_dtype](LEG_AWAY) * cos_angle + Scalar[
            gpu_dtype
        ](LEG_DOWN) * sin_angle
        var right_leg_y = y + Scalar[gpu_dtype](LEG_AWAY) * sin_angle - Scalar[
            gpu_dtype
        ](LEG_DOWN) * cos_angle

        # Contact detection (leg tip at or below helipad)
        left_contact = Scalar[gpu_dtype](0.0)
        right_contact = Scalar[gpu_dtype](0.0)

        if left_leg_y <= Scalar[gpu_dtype](HELIPAD_Y):
            left_contact = Scalar[gpu_dtype](1.0)
            # Prevent penetration - push leg up
            if left_leg_y < Scalar[gpu_dtype](HELIPAD_Y):
                var penetration = Scalar[gpu_dtype](HELIPAD_Y) - left_leg_y
                y = y + penetration * Scalar[gpu_dtype](0.5)
                if vy < Scalar[gpu_dtype](0.0):
                    vy = vy * Scalar[gpu_dtype](-0.1)  # Bounce damping

        if right_leg_y <= Scalar[gpu_dtype](HELIPAD_Y):
            right_contact = Scalar[gpu_dtype](1.0)
            if right_leg_y < Scalar[gpu_dtype](HELIPAD_Y):
                var penetration = Scalar[gpu_dtype](HELIPAD_Y) - right_leg_y
                y = y + penetration * Scalar[gpu_dtype](0.5)
                if vy < Scalar[gpu_dtype](0.0):
                    vy = vy * Scalar[gpu_dtype](-0.1)

        # =====================================================================
        # Check termination conditions
        # =====================================================================
        var x_norm = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
            W_UNITS / 2.0
        )
        var terminated = False
        var crashed = False
        var landed = False

        # Out of bounds
        if x_norm >= Scalar[gpu_dtype](1.0) or x_norm <= Scalar[gpu_dtype](-1.0):
            terminated = True
            crashed = True

        # Check if lander body (not legs) touches ground
        var lander_bottom_y = y - Scalar[gpu_dtype](
            LANDER_HALF_HEIGHT
        ) * cos_angle
        var both_legs = (left_contact > Scalar[gpu_dtype](0.5)) and (
            right_contact > Scalar[gpu_dtype](0.5)
        )
        if lander_bottom_y <= Scalar[gpu_dtype](HELIPAD_Y) and not both_legs:
            terminated = True
            crashed = True

        # Check for successful landing (both legs down, low velocity)
        if both_legs:
            var vx_abs = vx
            if vx_abs < Scalar[gpu_dtype](0.0):
                vx_abs = -vx_abs
            var vy_abs = vy
            if vy_abs < Scalar[gpu_dtype](0.0):
                vy_abs = -vy_abs
            var angvel_abs = angular_vel
            if angvel_abs < Scalar[gpu_dtype](0.0):
                angvel_abs = -angvel_abs

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
        var y_norm = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[
            gpu_dtype
        ](H_UNITS / 2.0)
        var vx_norm = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy_norm = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))

        var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
        var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
        var angle_abs = angle
        if angle_abs < Scalar[gpu_dtype](0.0):
            angle_abs = -angle_abs

        var new_shaping = (
            Scalar[gpu_dtype](-100.0) * dist
            + Scalar[gpu_dtype](-100.0) * speed
            + Scalar[gpu_dtype](-100.0) * angle_abs
            + Scalar[gpu_dtype](10.0) * left_contact
            + Scalar[gpu_dtype](10.0) * right_contact
        )

        # Shaped reward + fuel costs
        var reward = new_shaping - prev_shaping
        reward = reward - m_power * Scalar[gpu_dtype](MAIN_ENGINE_FUEL_COST)
        reward = reward - s_power * Scalar[gpu_dtype](SIDE_ENGINE_FUEL_COST)

        # Terminal rewards
        if crashed:
            reward = Scalar[gpu_dtype](-100.0)
        elif landed:
            reward = Scalar[gpu_dtype](100.0)

        # =====================================================================
        # NaN/Inf safety check - force termination if state is corrupted
        # =====================================================================
        var state_corrupted = False
        # Check for NaN (NaN != NaN is true)
        if x != x or y != y or vx != vx or vy != vy or angle != angle or angular_vel != angular_vel:
            state_corrupted = True
        # Check for extreme values (likely Inf or overflow)
        var extreme = Scalar[gpu_dtype](1e6)
        if x > extreme or x < -extreme or y > extreme or y < -extreme:
            state_corrupted = True
        if vx > extreme or vx < -extreme or vy > extreme or vy < -extreme:
            state_corrupted = True

        if state_corrupted:
            # Reset to safe initial state and terminate
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
        # Write back state (NORMALIZED for neural network stability)
        # Matches Gymnasium LunarLander observation format
        # =====================================================================
        # Re-normalize positions after physics update
        x_obs = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
            W_UNITS / 2.0
        )
        y_obs = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[
            gpu_dtype
        ](H_UNITS / 2.0)
        # Re-normalize velocities
        vx_obs = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        vy_obs = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        # Re-normalize angle
        angle_obs = angle / Scalar[gpu_dtype](pi)
        # Re-normalize angular velocity
        angular_vel_obs = angular_vel * Scalar[gpu_dtype](
            20.0 / Float64(FPS)
        )

        # Clamp all observations to [-10, 10] for neural network safety
        var obs_max = Scalar[gpu_dtype](10.0)
        if x_obs > obs_max:
            x_obs = obs_max
        if x_obs < -obs_max:
            x_obs = -obs_max
        if y_obs > obs_max:
            y_obs = obs_max
        if y_obs < -obs_max:
            y_obs = -obs_max
        if vx_obs > obs_max:
            vx_obs = obs_max
        if vx_obs < -obs_max:
            vx_obs = -obs_max
        if vy_obs > obs_max:
            vy_obs = obs_max
        if vy_obs < -obs_max:
            vy_obs = -obs_max
        if angular_vel_obs > obs_max:
            angular_vel_obs = obs_max
        if angular_vel_obs < -obs_max:
            angular_vel_obs = -obs_max

        states[i, 0] = x_obs
        states[i, 1] = y_obs
        states[i, 2] = vx_obs
        states[i, 3] = vy_obs
        states[i, 4] = angle_obs
        states[i, 5] = angular_vel_obs
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
        """GPU kernel to reset all environments to random initial states."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        # Initialize RNG with thread-specific seed
        var rng = xorshift32(Scalar[DType.uint32](i * 2654435761 + 12345))

        # Random initial position near top-center
        var result_x = random_range[gpu_dtype](
            rng,
            Scalar[gpu_dtype](INITIAL_X - 0.5),
            Scalar[gpu_dtype](INITIAL_X + 0.5),
        )
        var x = result_x[0]
        rng = result_x[1]

        var result_y = random_range[gpu_dtype](
            rng,
            Scalar[gpu_dtype](INITIAL_Y - 0.5),
            Scalar[gpu_dtype](INITIAL_Y),
        )
        var y = result_y[0]
        rng = result_y[1]

        # Random initial velocity (small)
        var result_vx = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.5), Scalar[gpu_dtype](0.5)
        )
        var vx = result_vx[0]
        rng = result_vx[1]

        var result_vy = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.5), Scalar[gpu_dtype](0.5)
        )
        var vy = result_vy[0]
        rng = result_vy[1]

        # Random initial angle (small)
        var result_angle = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.1), Scalar[gpu_dtype](0.1)
        )
        var angle = result_angle[0]
        rng = result_angle[1]

        # Random angular velocity (small)
        var result_angvel = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.1), Scalar[gpu_dtype](0.1)
        )
        var angular_vel = result_angvel[0]

        # Write initial state (NORMALIZED for neural network)
        var x_obs = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
            W_UNITS / 2.0
        )
        var y_obs = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[
            gpu_dtype
        ](H_UNITS / 2.0)
        var vx_obs = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy_obs = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        var angle_obs = angle / Scalar[gpu_dtype](pi)
        var angular_vel_obs = angular_vel * Scalar[gpu_dtype](
            20.0 / Float64(FPS)
        )

        states[i, 0] = x_obs
        states[i, 1] = y_obs
        states[i, 2] = vx_obs
        states[i, 3] = vy_obs
        states[i, 4] = angle_obs
        states[i, 5] = angular_vel_obs
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

        # Only reset if done
        if dones[i] < Scalar[gpu_dtype](0.5):
            return

        # Initialize RNG with thread + external seed for variety
        var rng = xorshift32(Scalar[DType.uint32](i * 2654435761) + rng_seed)

        # Random initial position
        var result_x = random_range[gpu_dtype](
            rng,
            Scalar[gpu_dtype](INITIAL_X - 0.5),
            Scalar[gpu_dtype](INITIAL_X + 0.5),
        )
        var x = result_x[0]
        rng = result_x[1]

        var result_y = random_range[gpu_dtype](
            rng,
            Scalar[gpu_dtype](INITIAL_Y - 0.5),
            Scalar[gpu_dtype](INITIAL_Y),
        )
        var y = result_y[0]
        rng = result_y[1]

        # Random initial velocity
        var result_vx = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.5), Scalar[gpu_dtype](0.5)
        )
        var vx = result_vx[0]
        rng = result_vx[1]

        var result_vy = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.5), Scalar[gpu_dtype](0.5)
        )
        var vy = result_vy[0]
        rng = result_vy[1]

        # Random initial angle
        var result_angle = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.1), Scalar[gpu_dtype](0.1)
        )
        var angle = result_angle[0]
        rng = result_angle[1]

        # Random angular velocity
        var result_angvel = random_range[gpu_dtype](
            rng, Scalar[gpu_dtype](-0.1), Scalar[gpu_dtype](0.1)
        )
        var angular_vel = result_angvel[0]

        # Write initial state (NORMALIZED for neural network)
        var x_obs = (x - Scalar[gpu_dtype](HELIPAD_X)) / Scalar[gpu_dtype](
            W_UNITS / 2.0
        )
        var y_obs = (y - Scalar[gpu_dtype](HELIPAD_Y + LEG_DOWN)) / Scalar[
            gpu_dtype
        ](H_UNITS / 2.0)
        var vx_obs = vx * Scalar[gpu_dtype](W_UNITS / 2.0 / Float64(FPS))
        var vy_obs = vy * Scalar[gpu_dtype](H_UNITS / 2.0 / Float64(FPS))
        var angle_obs = angle / Scalar[gpu_dtype](pi)
        var angular_vel_obs = angular_vel * Scalar[gpu_dtype](
            20.0 / Float64(FPS)
        )

        states[i, 0] = x_obs
        states[i, 1] = y_obs
        states[i, 2] = vx_obs
        states[i, 3] = vy_obs
        states[i, 4] = angle_obs
        states[i, 5] = angular_vel_obs
        states[i, 6] = Scalar[gpu_dtype](0.0)  # left_contact
        states[i, 7] = Scalar[gpu_dtype](0.0)  # right_contact

        # Clear done flag
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
    ) raises:
        """Launch step kernel on GPU."""
        # Create tensor views
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

        @always_inline
        fn step_wrapper(
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
        ):
            Self.step_kernel[BATCH_SIZE, STATE_SIZE](states, actions, rewards, dones)

        ctx.enqueue_function[step_wrapper, step_wrapper](
            states,
            actions,
            rewards,
            dones,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Launch reset kernel on GPU."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
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
                gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            rng_seed: Scalar[DType.uint32],
        ):
            Self.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                states, dones, rng_seed
            )

        ctx.enqueue_function[
            selective_reset_wrapper, selective_reset_wrapper
        ](
            states,
            dones,
            seed,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )
