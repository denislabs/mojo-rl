"""`manipulation/reach_site_features` as a COMPTIME model def.

The Phase 7 gates all drove Jaco through the RUNTIME path — `parse_xml_full`
plus `build_model_fields_from_flat`, over an XML baked into a temp directory by
Python. That path proved the model, forward kinematics, site IK and the contact
set, and it cannot do any more than that: the fields `Model` carries NO
ACTUATORS at all. `apply_actions` is a method on the comptime model def, so
`Phyics3dEnv` — and therefore any stepping of this task — needs this route.

⚠ THE TWO PATHS SHARE A PARSER BUT NOT ALL OF ONE. `ModelDefFromXML.init_fields`
calls `parse_xml_full`, so the Phase 7 parser fixes (the default joint axis, the
default-class geom attributes) apply here unchanged. The COUNTS below come from
`parse_xml`, which is the separate COMPTIME counter — a fix in one is not a fix
in the other, so `test_manipulation_reach_def` asserts every count against
MuJoCo rather than trusting that they agree.

⚠ `max_contacts` is 128 against a measured worst case of 48 over 60 in-range
poses (`test_jaco_contacts_vs_mujoco`). It is NOT the 256 the gates use: that
number was chosen while plane-mesh was emitting a contact per hull vertex and a
single pair could saturate the buffer. With `maxplanemesh` in place the real
ceiling is small, and an oversized contact buffer costs memory in every env.

ELLIPTIC CONE + 5 NOSLIP ITERATIONS IS SUPPORTED as of 2026-08-13 (task #53).
The baked XML carries `<option cone="elliptic" noslip_iterations="5"
noslip_tolerance="0">`, and `solver/noslip.mojo` now has the elliptic branch of
`mj_solNoSlip` with `_newton_solve_env` dispatching to it. This file used to
pass `allow_missing_noslip=True` to get past a comptime guard, and to state
that the resulting rollout would NOT match MuJoCo; both the guard and the flag
are gone.

That mattered here more than anywhere: measured on this model, MuJoCo against
itself from the same state with only `noslip_iterations` changed, `max|d(qacc)|`
is **7.4e+2 on step 1** against a `|qacc|` of 1.7e+4 — 4.2%, at 55 contacts.
Skipping the pass was never a small approximation.

CONDIM 4 IS ALSO SUPPORTED as of 2026-08-13 (task #55). This model has geoms at
`condim="4"` — the hand's fingertips — and 3 of its 55 contacts at qpos0 are
condim 4. The elliptic solver used to cache exactly three Jacobian rows per
contact (normal + two tangents) and one isotropic `mu`, so those three lost
their torsional row; it now carries `dim-1` tangential rows with per-direction
friction and `R`. Gated by
`tests/physics3d/test_elliptic_condim46_vs_mujoco.mojo`.

⚠ THIS MODEL STILL DOES NOT STEP IN PARITY, AND THE REASON IS NO LONGER THE
SOLVER. Measured from MuJoCo's `qpos0` with zero `qvel` and zero `ctrl`, five
steps of the model alone:

    step 0   ncon  ours 57 / MuJoCo 55   max|d(qvel)| 2.97e+1
    step 1         ours 54 / MuJoCo 59   max|d(qvel)| 3.35e+1

The CONTACT SET differs at step 0, before any solve can. So what remains is
narrow-phase generation on Jaco's meshes, not the friction cone — a different
subsystem from the two (noslip, condim) that were closed to get here. Tracked
separately; do NOT read the residual above as a cone-solver number.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_reach_xml import reach_site_features_xml

comptime pm = parse_xml(reach_site_features_xml)

comptime ReachSiteFeaturesModel = ModelDefFromXML[
    xml=reach_site_features_xml,
    nbody=pm.NBODY,
    njoint=pm.NJOINT,
    nq=pm.NQ,
    nv=pm.NV,
    ngeom=pm.NGEOM,
    nact=pm.NACT,
    ntex=pm.NTEX,
    nmat=pm.NMAT,
    nlight=pm.NLIGHT,
    ncam=pm.NCAM,
    nsite=pm.NSITE,
    neq=pm.NEQ,
    # ⚠ EVERY ONE OF THESE IS TAKEN FROM `pm`, NOT DEFAULTED. Each has a
    # default that silently disables a feature rather than failing:
    #   nexclude  0 -> the `<contact><exclude>` fill writes past a 1-element
    #                  tensor. That is how this was found: an index assert in
    #                  `fields_build`, not a diagnostic.
    #   npair     0 -> `<contact><pair>` rows dropped (that one does raise).
    #   max_tendon 0 -> tendons clamped away.
    #   max_condim 3 -> condim 4/6 geoms lose their torsional/rolling rows
    #                  (this model HAS condim-4 geoms; measured at 1.6e+3 of
    #                   qacc on the gate's ball).
    #   noslip_iter 0 -> the noslip pass is simply not run.
    # `max_equality` is the one that SIZES storage — `neq` alone does not, so
    # passing only `neq` makes equality constraints vanish silently.
    nexclude=pm.NEXCLUDE,
    npair=pm.NPAIR,
    max_tendon=pm.NTENDON,
    max_condim=pm.MAX_CONDIM,
    max_equality=pm.NEQ * 6,
    max_contacts=128,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    # The elliptic branch of `mj_solNoSlip` runs for this now — see the module
    # docstring. `noslip_tolerance="0"` reaches the solver through model META,
    # not through this parameter list.
    noslip_iter=pm.NOSLIP_ITER,
]
