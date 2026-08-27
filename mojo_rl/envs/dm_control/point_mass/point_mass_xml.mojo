"""`dm_control` `point_mass` model — port of `dm_control/suite/point_mass.xml`.

Verbatim apart from the `<include>` lines. Serves BOTH tasks: dm_control uses
one XML for `easy` and `hard`, which differ only in `initialize_episode`.

THE TENDONS. The mass is driven through two fixed tendons rather than two
joint motors:

    <tendon>
      <fixed name="t1"><joint joint="root_x" coef="1"/>
                       <joint joint="root_y" coef="0"/></fixed>
      ...
    </tendon>
    <actuator><motor name="t1" tendon="t1"/> ... </actuator>

A fixed tendon's actuator moment arm on joint j IS its `coef`, so with the
identity coefficient matrix above this is exactly two joint motors — `t1`
drives `root_x`, `t2` drives `root_y`, both at the default `gear=".1"`.

This port used to WRITE THAT EQUIVALENCE OUT, substituting `<motor joint=...>`
for the tendons, because the engine resolved an actuator to a single joint.
That substitution is gone: `ModelDefFromXML.apply_actions` now walks the
comptime transmission triples (`_acd.motor_trn_qadr/dadr/coef`), of which a
joint transmission is the degenerate one-triple coef-1 case, so the real
tendons parse and actuate directly.

Removing it matters for `hard`, which overwrites `model.wrap_prm` — i.e. these
very coefs — each episode with a random mixing matrix. A substituted model has
nowhere to put that. `DMPointMassHardConfig` therefore drives the DOFs itself
from the RUNTIME tendon records (`Model.tendons`), which the comptime tables
above cannot see; see the notes there.

Note the joints are `limited="true" range="-.29 .29"` SLIDE joints, so the
degree->radian conversion must not touch them — see the `<compiler angle>`
notes in docs/DM_CONTROL_PORT.md.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.point_mass.point_mass_dims import (
    DM_POINT_MASS_DIMS,
)





comptime pmp = DM_POINT_MASS_DIMS

# obs = position (qpos, 2) + velocity (qvel, 2) = 4
comptime DMPointMassModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/point_mass.xml",
    nbody=pmp.NBODY, njoint=pmp.NJOINT, nq=pmp.NQ, nv=pmp.NV,
    ngeom=pmp.NGEOM, nact=pmp.NACT, ntex=pmp.NTEX, nmat=pmp.NMAT,
    nlight=pmp.NLIGHT, ncam=pmp.NCAM, nsite=pmp.NSITE,
    max_contacts=1,
    max_tendon=pmp.NTENDON,
    obs_dim_override=4,
    timestep=pmp.TIMESTEP,
]

# Tendon indices, in declaration order. `t1` mixes into root_x, `t2` into
# root_y — under the XML's identity coefs; `hard` replaces both rows.
comptime T1_TENDON_IDX: Int = 0
comptime T2_TENDON_IDX: Int = 1

# Geom indices in OUR ordering, which is worldbody text (DFS) order.
#
# These ARE MuJoCo's geom indices, as of the element-order fix (2026-08-03).
#
# MuJoCo groups geoms by body id: the five world geoms, then `target` (also
# world, but declared AFTER the `<body>` in the XML), then `pointmass` on body
# 1. Our parser used to number them in XML TEXT order, which put `pointmass`
# at 5 and `target` at 6 — the reverse — and this file carried a note saying
# so, treating the divergence as a property to work around.
#
# It was a bug, and dog is what proved it: the same text-vs-body ordering
# permutes JOINTS, and `fields_build` derives `qpos_adr`/`dof_adr` as running
# counters over the joint array, so the whole `qpos` layout goes with it.
# `full_parser` now groups joints, geoms and sites by body
# (`_stable_group_by_body_*`), gated by
# `tests/physics3d/test_element_order_vs_mujoco.mojo`.
#
# point_mass was the ONLY previously ported domain that interleaved, which is
# why it is the only one whose constants moved.
comptime POINTMASS_GEOM_IDX: Int = 6
comptime TARGET_GEOM_IDX: Int = 5

# `named.model.geom_size['target', 0]` — the target sphere's radius. Geom sizes
# are not carried in a form the reward hook reads, so it is lifted from the XML
# here and asserted against `model.geom_size` in the parity test.
comptime TARGET_SIZE: Float64 = 0.015
