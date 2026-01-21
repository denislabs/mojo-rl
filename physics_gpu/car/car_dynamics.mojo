"""Full car dynamics simulation step.

This module combines all car physics components into a complete simulation step:
1. Update steering joints (front wheels track steering input)
2. Get friction limits from tile collision
3. Compute wheel forces via slip-based friction model
4. Integrate hull dynamics (Euler integration)
5. Update observation vector

The simulation is top-down (no gravity) with slip-based tire friction.
This matches the physics model in Gymnasium's car_dynamics.py.

Reference: gymnasium/envs/box2d/car_dynamics.py
"""

from math import cos, sin, sqrt, pi
from layout import LayoutTensor, Layout

from .constants import (
    # Physics constants
    HULL_MASS,
    HULL_INV_MASS,
    HULL_INERTIA,
    HULL_INV_INERTIA,
    STEERING_LIMIT,
    STEERING_MOTOR_SPEED,
    STEERING_GAIN,
    FRICTION_LIMIT,
    ROAD_FRICTION,
    CAR_DT,
    CAR_PI,
    CAR_TWO_PI,
    # State indices
    HULL_X,
    HULL_Y,
    HULL_ANGLE,
    HULL_VX,
    HULL_VY,
    HULL_OMEGA,
    HULL_STATE_SIZE,
    WHEEL_OMEGA,
    WHEEL_JOINT_ANGLE,
    WHEEL_PHASE,
    WHEEL_STATE_SIZE,
    CTRL_STEERING,
    CTRL_GAS,
    CTRL_BRAKE,
    # Wheel indices
    WHEEL_FL,
    WHEEL_FR,
    NUM_WHEELS,
    MAX_TRACK_TILES,
    TILE_DATA_SIZE,
)
from .layout import CarRacingLayout
from .wheel_friction import WheelFriction
from .tile_collision import TileCollision

from ..constants import dtype


struct CarDynamics:
    """Top-down car dynamics: hull + 4 wheels with tire friction.

    This struct provides static methods for simulating car physics.
    The physics is stateless - all state is passed via LayoutTensor.

    Simulation steps per time step:
    1. Update steering (front wheel joint angles)
    2. Compute wheel friction limits from track tiles
    3. Compute wheel forces (updates wheel omegas internally)
    4. Apply forces to hull (integrate dynamics)
    5. Update observation vector for RL
    """

    # =========================================================================
    # Steering Update
    # =========================================================================

    @staticmethod
    @always_inline
    fn update_steering[
        BATCH: Int,
        STATE_SIZE: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """Update front wheel steering angles.

        Front wheels smoothly rotate to match the steering input.
        Steering is limited by STEERING_LIMIT and the rotation rate
        is limited by STEERING_MOTOR_SPEED.

        From car_dynamics.py:
            dir = sign(steer - joint.angle)
            val = abs(steer - joint.angle)
            joint.motorSpeed = dir * min(50 * val, 3.0)

        Args:
            env: Environment index.
            state: State tensor.
            dt: Time step.
        """
        var steering_input = rebind[Scalar[dtype]](state[env, CONTROLS_OFFSET + CTRL_STEERING])

        # Clamp steering input to limits
        var steer_limit = Scalar[dtype](STEERING_LIMIT)
        var target_angle = steering_input * steer_limit  # Map [-1,1] to [-limit, limit]

        # Update front-left wheel
        var fl_off = WHEELS_OFFSET + WHEEL_FL * WHEEL_STATE_SIZE
        var fl_angle = rebind[Scalar[dtype]](state[env, fl_off + WHEEL_JOINT_ANGLE])
        fl_angle = CarDynamics._update_steering_angle(fl_angle, target_angle, dt)
        state[env, fl_off + WHEEL_JOINT_ANGLE] = fl_angle

        # Update front-right wheel (same steering angle)
        var fr_off = WHEELS_OFFSET + WHEEL_FR * WHEEL_STATE_SIZE
        var fr_angle = rebind[Scalar[dtype]](state[env, fr_off + WHEEL_JOINT_ANGLE])
        fr_angle = CarDynamics._update_steering_angle(fr_angle, target_angle, dt)
        state[env, fr_off + WHEEL_JOINT_ANGLE] = fr_angle

    @staticmethod
    @always_inline
    fn _update_steering_angle(
        current: Scalar[dtype],
        target: Scalar[dtype],
        dt: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Update a single steering angle towards target.

        Args:
            current: Current joint angle.
            target: Target joint angle.
            dt: Time step.

        Returns:
            New joint angle.
        """
        var diff = target - current
        var gain = Scalar[dtype](STEERING_GAIN)
        var max_speed = Scalar[dtype](STEERING_MOTOR_SPEED)
        var zero = Scalar[dtype](0.0)

        # Compute motor speed (proportional to error, capped)
        var speed = diff * gain
        if speed > max_speed:
            speed = max_speed
        elif speed < -max_speed:
            speed = -max_speed

        # Integrate angle
        var new_angle = current + speed * dt

        # Clamp to steering limits
        var limit = Scalar[dtype](STEERING_LIMIT)
        if new_angle > limit:
            new_angle = limit
        elif new_angle < -limit:
            new_angle = -limit

        return new_angle

    # =========================================================================
    # Hull Dynamics Integration
    # =========================================================================

    @staticmethod
    @always_inline
    fn integrate_hull[
        BATCH: Int,
        STATE_SIZE: Int,
        HULL_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        fx: Scalar[dtype],
        fy: Scalar[dtype],
        torque: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Integrate hull dynamics using semi-implicit Euler.

        Updates hull velocity from forces, then updates position from velocity.

        Args:
            env: Environment index.
            state: State tensor.
            fx, fy: Total force on hull (world frame).
            torque: Total torque on hull.
            dt: Time step.
        """
        # Read current state
        var x = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_X])
        var y = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_Y])
        var angle = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_ANGLE])
        var vx = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VX])
        var vy = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VY])
        var omega = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_OMEGA])

        var inv_mass = Scalar[dtype](HULL_INV_MASS)
        var inv_inertia = Scalar[dtype](HULL_INV_INERTIA)

        # Semi-implicit Euler: update velocity first
        vx = vx + fx * inv_mass * dt
        vy = vy + fy * inv_mass * dt
        omega = omega + torque * inv_inertia * dt

        # Then update position using new velocity
        x = x + vx * dt
        y = y + vy * dt
        angle = angle + omega * dt

        # Normalize angle to [-pi, pi]
        var pi_val = Scalar[dtype](CAR_PI)
        var two_pi = Scalar[dtype](CAR_TWO_PI)
        if angle > Scalar[dtype](100.0) or angle < Scalar[dtype](-100.0):
            angle = Scalar[dtype](0.0)
        elif angle > pi_val:
            angle = angle - two_pi
            if angle > pi_val:
                angle = angle - two_pi
        elif angle < -pi_val:
            angle = angle + two_pi
            if angle < -pi_val:
                angle = angle + two_pi

        # Write back
        state[env, HULL_OFFSET + HULL_X] = x
        state[env, HULL_OFFSET + HULL_Y] = y
        state[env, HULL_OFFSET + HULL_ANGLE] = angle
        state[env, HULL_OFFSET + HULL_VX] = vx
        state[env, HULL_OFFSET + HULL_VY] = vy
        state[env, HULL_OFFSET + HULL_OMEGA] = omega

    # =========================================================================
    # Single Environment Step
    # =========================================================================

    @staticmethod
    @always_inline
    fn step_single_env[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        tiles: LayoutTensor[
            dtype,
            Layout.row_major(MAX_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """Perform one physics step for a single environment.

        This is the core simulation step that can be called from:
        - CPU batch loop
        - GPU kernel (one thread per environment)

        Args:
            env: Environment index.
            state: State tensor [BATCH, STATE_SIZE].
            tiles: Track tile data [MAX_TILES, TILE_DATA_SIZE].
            num_active_tiles: Number of valid tiles.
            dt: Time step.
        """
        # Step 1: Update steering joints
        CarDynamics.update_steering[BATCH, STATE_SIZE, WHEELS_OFFSET, CONTROLS_OFFSET](
            env, state, dt
        )

        # Step 2: Get friction limits from track tiles
        var hull_x = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_X])
        var hull_y = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_Y])
        var hull_angle = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_ANGLE])

        var friction_limits = TileCollision.get_wheel_friction_limits[MAX_TILES](
            hull_x, hull_y, hull_angle, tiles, num_active_tiles
        )

        # Step 3: Compute wheel forces (also updates wheel omegas)
        var forces = WheelFriction.compute_all_wheels_forces[
            BATCH, STATE_SIZE, HULL_OFFSET, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, friction_limits, dt)

        var fx = forces[0]
        var fy = forces[1]
        var torque = forces[2]

        # Step 4: Integrate hull dynamics
        CarDynamics.integrate_hull[BATCH, STATE_SIZE, HULL_OFFSET](
            env, state, fx, fy, torque, dt
        )

    # =========================================================================
    # Observation Update
    # =========================================================================

    @staticmethod
    @always_inline
    fn update_observation[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
    ):
        """Update observation vector from physics state.

        Observation format (13 values):
        - [0-5]: Hull state (x, y, angle, vx, vy, omega)
        - [6-7]: Front wheel angles (FL, FR)
        - [8-11]: Wheel angular velocities (FL, FR, RL, RR)
        - [12]: (reserved for speed indicator or other)

        Args:
            env: Environment index.
            state: State tensor.
        """
        # Hull state (6 values)
        state[env, OBS_OFFSET + 0] = state[env, HULL_OFFSET + HULL_X]
        state[env, OBS_OFFSET + 1] = state[env, HULL_OFFSET + HULL_Y]
        state[env, OBS_OFFSET + 2] = state[env, HULL_OFFSET + HULL_ANGLE]
        state[env, OBS_OFFSET + 3] = state[env, HULL_OFFSET + HULL_VX]
        state[env, OBS_OFFSET + 4] = state[env, HULL_OFFSET + HULL_VY]
        state[env, OBS_OFFSET + 5] = state[env, HULL_OFFSET + HULL_OMEGA]

        # Front wheel angles (2 values)
        var fl_off = WHEELS_OFFSET + WHEEL_FL * WHEEL_STATE_SIZE
        var fr_off = WHEELS_OFFSET + WHEEL_FR * WHEEL_STATE_SIZE
        state[env, OBS_OFFSET + 6] = state[env, fl_off + WHEEL_JOINT_ANGLE]
        state[env, OBS_OFFSET + 7] = state[env, fr_off + WHEEL_JOINT_ANGLE]

        # Wheel angular velocities (4 values)
        @parameter
        for wheel in range(NUM_WHEELS):
            var wheel_off = WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE
            state[env, OBS_OFFSET + 8 + wheel] = state[env, wheel_off + WHEEL_OMEGA]

        # Speed indicator (computed from vx, vy)
        var vx = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VX])
        var vy = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VY])
        var speed = sqrt(vx * vx + vy * vy)
        state[env, OBS_OFFSET + 12] = speed

    # =========================================================================
    # Full Step with Observation Update
    # =========================================================================

    @staticmethod
    @always_inline
    fn step_with_obs[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        tiles: LayoutTensor[
            dtype,
            Layout.row_major(MAX_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """Perform physics step and update observations.

        This combines step_single_env and update_observation into a single
        method for convenience.

        Args:
            env: Environment index.
            state: State tensor.
            tiles: Track tile data.
            num_active_tiles: Number of valid tiles.
            dt: Time step.
        """
        # Physics step
        CarDynamics.step_single_env[
            BATCH, STATE_SIZE, OBS_OFFSET, HULL_OFFSET, WHEELS_OFFSET,
            CONTROLS_OFFSET, MAX_TILES
        ](env, state, tiles, num_active_tiles, dt)

        # Update observations
        CarDynamics.update_observation[
            BATCH, STATE_SIZE, OBS_OFFSET, OBS_DIM, HULL_OFFSET, WHEELS_OFFSET
        ](env, state)

    @staticmethod
    @always_inline
    fn step_trackless[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """Perform physics step without track (uses default road friction).

        This version assumes all wheels are on road (friction = 1.0).
        Useful for simplified training without track tile collision.

        Args:
            env: Environment index.
            state: State tensor.
            dt: Time step.
        """
        # Step 1: Update steering joints
        CarDynamics.update_steering[BATCH, STATE_SIZE, WHEELS_OFFSET, CONTROLS_OFFSET](
            env, state, dt
        )

        # Step 2: Use default road friction for all wheels (no tile lookup)
        # friction_limits = [fl, fr, rl, rr] all set to FRICTION_LIMIT * ROAD_FRICTION
        var default_friction = Scalar[dtype](FRICTION_LIMIT * ROAD_FRICTION)
        var friction_limits = InlineArray[Scalar[dtype], NUM_WHEELS](
            default_friction, default_friction, default_friction, default_friction
        )

        # Step 3: Compute wheel forces (also updates wheel omegas)
        var forces = WheelFriction.compute_all_wheels_forces[
            BATCH, STATE_SIZE, HULL_OFFSET, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, friction_limits, dt)

        var fx = forces[0]
        var fy = forces[1]
        var torque = forces[2]

        # Step 4: Integrate hull dynamics
        CarDynamics.integrate_hull[BATCH, STATE_SIZE, HULL_OFFSET](
            env, state, fx, fy, torque, dt
        )

        # Step 5: Update observations
        CarDynamics.update_observation[
            BATCH, STATE_SIZE, OBS_OFFSET, OBS_DIM, HULL_OFFSET, WHEELS_OFFSET
        ](env, state)

    # =========================================================================
    # CPU Batch Processing
    # =========================================================================

    @staticmethod
    fn step_batch_cpu[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        tiles: LayoutTensor[
            dtype,
            Layout.row_major(MAX_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """Step all environments (CPU).

        Args:
            state: State tensor [BATCH, STATE_SIZE].
            tiles: Track tile data [MAX_TILES, TILE_DATA_SIZE].
            num_active_tiles: Number of valid tiles.
            dt: Time step.
        """
        for env in range(BATCH):
            CarDynamics.step_with_obs[
                BATCH, STATE_SIZE, OBS_OFFSET, OBS_DIM, HULL_OFFSET,
                WHEELS_OFFSET, CONTROLS_OFFSET, MAX_TILES
            ](env, state, tiles, num_active_tiles, dt)
