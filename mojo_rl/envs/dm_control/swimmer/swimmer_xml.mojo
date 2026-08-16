"""`dm_control` `swimmer` model — port of `dm_control/suite/swimmer.xml` PLUS the
procedural body chain that `swimmer._make_model(n_bodies)` builds on top of it.

This is the first domain in the port whose model is GENERATED, not written: the
reference parses `swimmer.xml` (head only), then appends `n_bodies - 1` nested
segment bodies, one `<motor>` per segment joint, and one `velocimeter`/`gyro`
sensor pair per segment site. `_swimmer_body_xml` below is that same loop, run
at comptime so `parse_xml` still sees a plain string literal's worth of XML.

    swimmer6  = 6 links  -> 5 segments, NQ = NV = 8,  obs 25
    swimmer15 = 15 links -> 14 segments, NQ = NV = 17, obs 61

TWO DELIBERATE SUBSTITUTIONS, both already-charted gaps:

1. THE TARGET BECOMES A MOCAP BODY (gap G4), exactly as reacher and finger
   needed. The reference declares it as a static worldbody geom

       <geom name="target" type="sphere" pos="1 1 .05" size=".1" material="target"/>

   and `Swimmer.initialize_episode` then rewrites `model.geom_pos['target',
   'x'/'y']` every episode. `fields.Model` is one SHARED, UNBATCHED tensor set,
   so a model write is a write for every env in the batch. A mocap body carries
   the pose in `d.mocap_pos`, which IS per-env state. The geom rides its body at
   the body origin, so `geom_xpos['target']` is exactly `d.xpos[TARGET_BODY]`.

       <body name="target" mocap="true" pos="1 1 .05">
         <geom name="target" type="sphere" size=".1" material="target" mass="0"/>
       </body>

   `mass="0"` is ours: the reference geom is static-in-world so its (default
   density) mass never reaches the dynamics, but on a mocap body it would reach
   `mj_fluid`'s per-body loop. It does not actually make `body_mass[target]`
   zero — `compute_inertia_from_geoms` skips a body with no MASS-CONTRIBUTING
   geom entirely, leaving the staging default (mass 1, diag inertia .01), the
   same value finger's mocap target carries. Inert either way: a mocap body has
   no DOF, so the fluid wrench projects to nothing and FK pins its velocity to
   zero, which is what makes the drag zero in the first place.

2. `light_pos['target_light']` is NOT tracked. `initialize_episode` moves the
   light with the target; light position is pure rendering and no observation
   or reward reads it.

WHAT IS NOT SUBSTITUTED, and matters: `<option density="3000">` with
`<flag contact="disable"/>`. Swimmer is the FIRST model in this repo to turn
the fluid path on, so `dynamics/fluid_forces.mojo` (MuJoCo's inertia-box model,
`mj_inertiaBoxFluidModel`) goes from dead code to the dominant force — with
contacts disabled and gravity irrelevant to a planar swimmer, drag is the ONLY
thing that converts joint torque into forward motion. The parity test gates it
directly.

GEOM ORDER, as always: ours is XML text order, MuJoCo's is sorted by body id.
Text order here is ground(0), then the head body's head(1) nose(2) eyes(3)
inertial(4) visual(5), then visual_i/inertial_i per segment, then target last —
so `NOSE_GEOM_IDX` is 2 for every `n_bodies`. The parity test pins it by
checking the record's body id and local pos rather than trusting the count.

⚠ `geom head` is an `ellipsoid`, and `_geom_type_from_str` has no ellipsoid
case — it falls through to sphere SILENTLY (the same trap finger's touch sites
hit). Harmless here twice over: the geom carries `mass="0"` so it contributes
no inertia, and `<flag contact="disable"/>` means no narrow phase ever reads its
shape. Pinned by the parity test so it cannot rot into something load-bearing.

Reference: references/dm_control-main/dm_control/suite/swimmer.py + .xml
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.swimmer.swimmer_dims import (
    DM_SWIMMER6_DIMS,
    DM_SWIMMER15_DIMS,
)


# ⚠ THE PROCEDURAL MJCF GENERATOR LIVED HERE, and it is gone with the
# rest of the embedded MJCF (phase 1b.5). `_swimmer_body_xml(n)` built
# swimmer's body by loop, mirroring dm_control's own generator, and
# `merge_mjcf` composed the 6- and 15-segment models from it AT COMPILE
# TIME. Its OUTPUT is now `assets/swimmer6.xml` and `assets/swimmer15.xml`,
# extracted verbatim, so the models read files like every other domain.
#
# ⚠ That means the two sizes are now FIXED ASSETS rather than a function of
# `n`. dm_control's `swimmer(n_links)` can build any n; this tree ships the
# two the suite defines. Adding a third is: run the generator once, write
# the .xml, regenerate its dims. Do NOT reintroduce a comptime builder —
# it is a comptime reader of the model by another name.


comptime ps6 = DM_SWIMMER6_DIMS

comptime ps15 = DM_SWIMMER15_DIMS

# obs = joints (n-1) + to_target (2) + body_velocities (3n)
comptime DMSwimmer6Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/swimmer6.xml",
    nbody = ps6.NBODY, njoint = ps6.NJOINT, nq = ps6.NQ, nv = ps6.NV,
    ngeom = ps6.NGEOM, nact = ps6.NACT, ntex = ps6.NTEX, nmat = ps6.NMAT,
    nlight = ps6.NLIGHT, ncam = ps6.NCAM, nsite = ps6.NSITE,
    max_contacts=1,
    obs_dim_override=25,
    timestep = ps6.TIMESTEP,
]

comptime DMSwimmer15Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/swimmer15.xml",
    nbody = ps15.NBODY, njoint = ps15.NJOINT, nq = ps15.NQ, nv = ps15.NV,
    ngeom = ps15.NGEOM, nact = ps15.NACT, ntex = ps15.NTEX, nmat = ps15.NMAT,
    nlight = ps15.NLIGHT, ncam = ps15.NCAM, nsite = ps15.NSITE,
    max_contacts=1,
    obs_dim_override=61,
    timestep = ps15.TIMESTEP,
]


# ── Indices. Everything here is independent of `n_bodies` except the target,
# which is always last, so the config stays generic in NBODY.
#
#   body 0            world
#   body 1            head
#   body 2 .. NBODY-2 segment_0 .. segment_{n-2}
#   body NBODY-1      target (mocap)
comptime HEAD_BODY_IDX: Int = 1
comptime FIRST_SEGMENT_BODY_IDX: Int = 2

# Geoms in XML text order (see the module docstring).
comptime GROUND_GEOM_IDX: Int = 0
comptime HEAD_GEOM_IDX: Int = 1
comptime NOSE_GEOM_IDX: Int = 2

# qpos/qvel: three root DOFs, then one hinge per segment. `Physics.joints()` is
# `qpos[3:]`, i.e. everything after these.
comptime N_ROOT_DOF: Int = 3

# `geom_size['target', 0]` — the reward's `bounds`/`margin` scale. Constant in
# the reference too (nothing writes geom_size for swimmer).
comptime TARGET_SIZE: Float64 = 0.1

# The target's z never moves: `initialize_episode` writes only x and y.
comptime TARGET_Z: Float64 = 0.05
