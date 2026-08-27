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

⚠⚠ THIS MODEL DOES STEP IN PARITY, AND THE NOTE THAT SAID OTHERWISE WAS
MEASURED OFF-DISTRIBUTION. Every reach parity number here came from a 40-pose
sweep that draws `qpos` uniformly from dm_control's sampling bounds and steps
whatever comes out. dm_control's OWN reset draws from those same bounds and
then REJECTS anything in contact — 10 of 10 episodes reset with zero contacts.
Dropping the rejection does not perturb the distribution, it replaces it:

                            the sweep          the task
    deepest penetration     -317 mm            -0.55 mm      (575x)
    simultaneous contacts   up to 45           up to 2

Measured where the task actually operates
(`tests/dm_control/test_reach_parity_in_distribution.mojo`):

    contact-free, 400 random-control steps      |d(qvel)| 2.8e-17
    shallow contacts, 12 poses in [-1.4 mm, 0)  |d(qvel)| worst 4.2e-4,
                                                typical ~1e-7,
                                                contact COUNTS 12/12 exact

against the sweep's 62.3. Four to sixteen orders, depending which pose you ask.

⚠ THE OLD TABLE IS KEPT BELOW because the conclusions drawn from it are still
in the tree and in the task list, and because "our numbers were fine, our
sample was not" is the failure this file should teach:

    worst |d(pos)| on a PENETRATING contact     |d(qvel)|
    ~1e-9                                       1e-9 .. 4e-6
    ~2e-2                                       2e-1 .. 1.9

Those 2e-1..1.9 rows are poses penetrating 100-300 mm. They are real
divergences of a real regime; they are not this task's regime.

The solve itself is exact where the geometry is. On the one-contact condim-4
pose our rows match MuJoCo's `efc` table to 7+ digits — every `D` (4836.150
normal and slide, 0.1209 torsional), every `aref`, every constraint force
(17028.905 normal, 6268.873 / 3221.375 slide, 77.509 torsional) and all nine
dry-friction dof rows.

What differs THERE is the up-to-two EXTRA plane-mesh contacts: our DEEPEST
contact agrees with MuJoCo's to 1e-10 or better on every plane-mesh pose, so
the support point is right and only the neighbourhood extras — which
`mjc_PlaneConvex` picks in qhull's facet order — land elsewhere. Tracked as
task #56; `_plane_mesh_contacts` and `build_hull_edge_graph` carry the
per-mesh numbers. Closing it means running qhull, not tightening a tolerance.

⚠ AND IT IS WORTH LESS THAN IT LOOKS. Injecting MuJoCo's OWN `mesh_graph`
collapses five sweep poses from 0.198-1.908 to ~1e-7 — so the graph really is
the cause there — but all five penetrate 100-300 mm. In the shallow band the
task reaches, contact counts already match 12/12 and the worst divergence is
4.2e-4. Do not read #56 as blocking reach.

⚠⚠ THE CYLINDER-MESH NORMAL WAS FILED HERE AS A SECOND DEFECT (task #57) AND
IT IS NOT ONE. At the one clean 1-vs-1 pose our normal sits 9.4e-3 from
MuJoCo's, which read as ours being wrong. Arbitrated against the DEFINITION of
penetration depth — `min over unit n of h(n)`, needing no second
implementation — ours is 0.03 deg from the minimising direction and MuJoCo's
is 0.54 deg. MuJoCo is not converged here at its default `ccd_tolerance`, and
tightening it to 1e-12 moves it to 4.01 deg, i.e. FURTHER away. Gated by
`tests/physics3d/test_epa_optimality_cylinder_mesh.mojo`, which runs on
MuJoCo's own float32 `mesh_vert` so the shape cannot be the excuse.

⚠ AN EARLIER VERSION OF THIS NOTE SAID "the CONTACT SET differs, 57 vs 55 at
step 0". Counts actually match on 38 of 40 poses, and the `qacc` numbers that
claim rested on were an artefact: they compared our post-step `qvel/dt`
against MuJoCo's PRE-step `qacc` from `mj_forward`, which ignores `mj_Euler`'s
IMPLICIT treatment of `dof_damping` — 0.75 on the fingers against an inertia
of ~0.1, i.e. exactly the 1.5% that read as a residual. Kept here rather than
deleted, because it is the shape of mistake this file is most likely to
attract again.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.manipulation_reach_dims import (
    REACH_SITE_FEATURES_DIMS,
)

comptime pm = REACH_SITE_FEATURES_DIMS

comptime ReachSiteFeaturesModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulation/reach_site_features.xml",
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
    # ⚠ THE OBSERVATION IS NOT qpos+qvel. `Reach`'s `_features` observation is
    # 45 numbers assembled by `manipulation_reach_config`, and the default
    # formula here would advertise nq - skip + nv = 17. `Phyics3dEnv` sizes the
    # observation buffer from THIS, not from the config's hook, so leaving it
    # at the default truncates the observation to its first 17 entries with
    # nothing raised. `obs_qpos_skip=0` because the hook indexes qpos itself.
    obs_dim_override=45,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    # The elliptic branch of `mj_solNoSlip` runs for this now — see the module
    # docstring. `noslip_tolerance="0"` reaches the solver through model META,
    # not through this parameter list.
    noslip_iter=pm.NOSLIP_ITER,
]
