"""Forward-Backward representations — zero-shot RL.

`docs/BFM_ZERO_SHOT_RL.md`. One `B: S -> R^d` and one `F: S x A x Z -> R^d`
approximate the successor measure of a whole family of policies at once, so a
reward supplied AFTER training picks a policy by linear algebra
(`z = E_rho[B(s)·r(s)]`) instead of by another training run.

Milestone 1 scope: the pieces, each separately gated. `point_mass` first and
walker second — on `point_mass` (nq = 2) the successor measure is traceable by
hand, and a collapsed `B` and a correct one produce the same loss curve on
walker.
"""

from .z_sampler import (
    sample_z,
    sample_z_uniform,
    z_from_b,
    z_from_reward,
)
