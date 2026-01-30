"""Physics3D v2 dynamics - Acceleration computation.

Computes accelerations from forces using Newton's second law.
For a free body: a = F/m + g (linear), α = I⁻¹·τ (angular).
"""

from .types import Model, Data


fn compute_acceleration[
    DTYPE: DType
](model: Model[DTYPE], mut data: Data[DTYPE]):
    """Compute qacc from forces (Newton's 2nd law).

    Linear: a = F/m + g
    Angular: α = I⁻¹·τ (diagonal inertia assumption)
    """
    var inv_mass = Scalar[dtype](1.0) / model.body.mass

    # Linear acceleration: a = F/m + g
    # Note: gravity is [gx, gy, gz], so we add it component-wise
    data.qacc[0] = data.qfrc_applied[0] * inv_mass
    data.qacc[1] = data.qfrc_applied[1] * inv_mass
    data.qacc[2] = data.qfrc_applied[2] * inv_mass + model.gravity_z

    # Angular acceleration: α = I⁻¹·τ (diagonal inertia)
    # For diagonal inertia tensor, each component is independent
    data.qacc[3] = data.qfrc_applied[3] / model.body.inertia_xx
    data.qacc[4] = data.qfrc_applied[4] / model.body.inertia_yy
    data.qacc[5] = data.qfrc_applied[5] / model.body.inertia_zz
