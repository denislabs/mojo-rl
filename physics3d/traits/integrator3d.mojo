"""Integrator3D trait for 3D physics integration.

Defines the interface for numerical integrators that advance
body state forward in time.
"""

from layout import LayoutTensor, Layout
from ..constants import dtype


trait Integrator3D:
    """Interface for 3D physics integrators.

    Integrators take the current body state (position, orientation, velocity)
    and accumulated forces/torques, then compute the next state.
    """

    @staticmethod
    fn integrate_velocities[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        gravity: SIMD[dtype, 4],  # (gx, gy, gz, 0)
        dt: Scalar[dtype],
    ):
        """Integrate velocities using forces/torques.

        Updates linear and angular velocities based on accumulated forces.
        Does NOT update positions yet.

        v(t+dt) = v(t) + (F/m + g) * dt
        ω(t+dt) = ω(t) + I^-1 * τ * dt
        """
        ...

    @staticmethod
    fn integrate_positions[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        dt: Scalar[dtype],
    ):
        """Integrate positions using velocities.

        Updates positions and orientations based on current velocities.

        x(t+dt) = x(t) + v * dt
        q(t+dt) = q(t) + 0.5 * ω_quat * q(t) * dt  (then normalize)
        """
        ...

    @staticmethod
    fn integrate[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        gravity: SIMD[dtype, 4],
        dt: Scalar[dtype],
    ):
        """Full integration step (velocities then positions).

        This is a convenience method that calls integrate_velocities
        followed by integrate_positions.
        """
        ...
