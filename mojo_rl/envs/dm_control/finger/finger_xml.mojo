"""`dm_control` `finger` model — port of `dm_control/suite/finger.xml`.

Verbatim apart from the `<include>` lines and ONE deliberate substitution, the
same one reacher needed (gap G4).

THE SUBSTITUTION — the target becomes a MOCAP BODY. The reference declares it
as a plain worldbody site:

    <site name="target" type="sphere" size=".03" pos="0 0 .4" material="target"/>

and `Turn.initialize_episode` then rewrites `model.site_pos['target', ['x','z']]`
every episode from a fresh random angle. `fields.Model` is a single SHARED,
UNBATCHED tensor set, so a model write is a write for every env in the batch.
A mocap body is the sanctioned workaround: FK skips mocap bodies and the facade
presets their world pose from `d.mocap_pos`, which is per-env `[BATCH, NBODY*3]`
state — so the target moves per episode without the model moving. The site rides
its body, so the `framepos` sensor reads back correctly.

    <body name="target" mocap="true" pos="0 0 .4">
      <site name="target" type="sphere" size=".03" material="target"/>
    </body>

This is physically inert in both versions: the site carries no geom, and a
jointless body contributes no DOF. It adds one body (index 4, appended after
the arm chain and spinner so those keep the reference's own 1..3).

WHAT RESET WRITES THAT WE CARRY AS CONFIG COMPTIMES INSTEAD (all constant per
task, so none of them needs a per-episode model write):
  * `site_size['target', 0]` — .07 (turn_easy) / .03 (turn_hard). Feeds only
    the reward radius via `dist_to_target`, never a contact.
  * `dof_damping['hinge']` — `Spin.initialize_episode` drops it from the XML's
    .5 to .03. The XML below keeps the reference's .5; the spin config applies
    .03. This one is NOT cosmetic — it changes the spinner's dynamics.
  * `site_rgba['target'/'tip', 3] = 0` in Spin — pure visuals, dropped.

`<option cone="elliptic" iterations="200">` — the cone is passed through as
`cone_type=ConeType.ELLIPTIC` (a `ModelDefFromXML` parameter; the parser does
not read the attribute). MuJoCo's 200 solver iterations already match our
Newton default, so `iterations` needs no plumbing. `<flag gravity="disable"/>`
IS parsed and zeroes the gravity vector.

⚠ `joint proximal` carries `ref="-90"`, so its qpos0 is NOT zero — the one
place in this model where the reference configuration differs from all-zeros.

GEOM ORDER, as always: ours is XML text order, MuJoCo's is sorted by body id.
The parity test pins both.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.finger.finger_dims import (
    DM_FINGER_DIMS,
    DM_FINGER_SPIN_DIMS,
)


# `Spin.initialize_episode` writes `dof_damping['hinge'] = .03` (the XML says
# .5). That is a real dynamics change and `fields.Model` is shared+unbatched,
# so it cannot be a per-episode write — spin loads its OWN asset,
# `finger_spin.xml`, with the value already substituted (`finger.xml` keeps the
# reference's .5 for turn). The substitution is asserted in the parity test,
# because a silent no-op here would leave spin running the turn dynamics.


comptime pmf = DM_FINGER_DIMS

comptime pmfs = DM_FINGER_SPIN_DIMS

# obs (spin) = position (4) + velocity (3) + touch (2) = 9
comptime DMFingerSpinModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/finger_spin.xml",
    nbody=pmf.NBODY, njoint=pmf.NJOINT, nq=pmf.NQ, nv=pmf.NV,
    ngeom=pmf.NGEOM, nact=pmf.NACT, ntex=pmf.NTEX, nmat=pmf.NMAT,
    nlight=pmf.NLIGHT, ncam=pmf.NCAM, nsite=pmf.NSITE,
    max_contacts=8,
    obs_dim_override=9,
    timestep=pmf.TIMESTEP,
    cone_type = ConeType.ELLIPTIC,
]

# obs (turn) = position (4) + velocity (3) + touch (2) + target_position (2)
#            + dist_to_target (1) = 12
comptime DMFingerTurnModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/finger.xml",
    nbody=pmf.NBODY, njoint=pmf.NJOINT, nq=pmf.NQ, nv=pmf.NV,
    ngeom=pmf.NGEOM, nact=pmf.NACT, ntex=pmf.NTEX, nmat=pmf.NMAT,
    nlight=pmf.NLIGHT, ncam=pmf.NCAM, nsite=pmf.NSITE,
    max_contacts=8,
    obs_dim_override=12,
    timestep=pmf.TIMESTEP,
    cone_type = ConeType.ELLIPTIC,
]

# Body indices in worldbody DFS order (0 = world); `target` appended last.
comptime PROXIMAL_BODY_IDX: Int = 1
comptime DISTAL_BODY_IDX: Int = 2
comptime SPINNER_BODY_IDX: Int = 3
comptime TARGET_BODY_IDX: Int = 4

# Site indices in XML text order — pinned by the parity test.
comptime TOUCHTOP_SITE_IDX: Int = 0
comptime TOUCHBOTTOM_SITE_IDX: Int = 1
comptime TIP_SITE_IDX: Int = 2
comptime TARGET_SITE_IDX: Int = 3

# qpos / qvel addresses (three hinges, in XML order).
comptime PROXIMAL_ADR: Int = 0
comptime DISTAL_ADR: Int = 1
comptime HINGE_ADR: Int = 2

# `radius = model.geom_size['cap1'].sum()` in Turn.initialize_episode — the
# arm length at which the target is placed around the hinge (.04 + .09).
comptime SPINNER_RADIUS: Float64 = 0.13

# Target site z when Turn writes only x/z: the body sits at the hinge height.
comptime TARGET_Z: Float64 = 0.4
