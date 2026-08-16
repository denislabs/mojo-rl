"""`dm_control` `humanoid` model — port of `dm_control/suite/humanoid.xml`.

Verbatim apart from the `<include>` lines and the `<sensor>` block.

THE SENSOR BLOCK IS DROPPED, and that is not a shortcut. `humanoid.xml`
declares 18 `touch`, 6 `force`, 6 `torque`, an accelerometer, a velocimeter
and a gyro — and NONE of the four tasks reads any of them. `grep sensordata
humanoid.py` returns exactly one line, `torso_subtreelinvel`, which we compute
directly from `Data.xvel` via `sensors.subtree_linvel`. (`merge_mjcf` does not
carry a `<sensor>` section anyway, so keeping the block would have been
decorative.) Every SITE stays, so `nsite` still matches MuJoCo's 25 and a
future `touch` port has its zones.

WHAT THIS MODEL EXERCISES that no earlier ported domain does:

  * `<freejoint name="root"/>`. MJCF sugar for `<joint type="free">`; our
    scanners look for the literal `"<joint"` in ~20 places, so `merge_mjcf`
    now normalizes the alias textually before anything scans. Without it the
    torso welds to the world and nq/nv come out 7/6 short — silently, since an
    unrecognized element is not an error.
  * JOINT SPRINGS. Every `<joint>` inherits `stiffness="1"` from
    `<default class="body">`, with 5/10/20 on the big joints and 3/6 on the
    ankles. The integrators have always assembled `fnet = qfrc - bias -
    damping - stiffness - frictionloss`, but our Gym humanoid sets
    `stiffness="0"` everywhere, so this is the first model that actually
    loads that term.
  * THREE-DEEP nested default classes (`body` > `big_joint` >
    `big_stiff_joint`) plus `childclass="body"` on the torso.

Body order is the tree DFS in both engines, so our indices match MuJoCo's
here — unlike the GEOM order, which still differs (see `point_mass_xml`).
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.humanoid.humanoid_dims import DM_HUMANOID_DIMS





comptime pmh = DM_HUMANOID_DIMS

# Shared model parameters — the two obs layouts differ only in `OBS_DIM`.
#
#   feature obs = joint_angles (21) + head_height (1) + extremities (12)
#               + torso_vertical (3) + com_velocity (3) + velocity (27) = 67
#   pure state  = position (28) + velocity (27)                         = 55
comptime HUMANOID_OBS_DIM: Int = 67
comptime HUMANOID_PURE_OBS_DIM: Int = 55

comptime DMHumanoidModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/humanoid.xml",
    nbody=pmh.NBODY, njoint=pmh.NJOINT, nq=pmh.NQ, nv=pmh.NV,
    ngeom=pmh.NGEOM, nact=pmh.NACT, ntex=pmh.NTEX, nmat=pmh.NMAT,
    nlight=pmh.NLIGHT, ncam=pmh.NCAM, nsite=pmh.NSITE,
    max_contacts=32,
    obs_dim_override=HUMANOID_OBS_DIM,
    timestep=pmh.TIMESTEP,
]

comptime DMHumanoidPureModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/humanoid.xml",
    nbody=pmh.NBODY, njoint=pmh.NJOINT, nq=pmh.NQ, nv=pmh.NV,
    ngeom=pmh.NGEOM, nact=pmh.NACT, ntex=pmh.NTEX, nmat=pmh.NMAT,
    nlight=pmh.NLIGHT, ncam=pmh.NCAM, nsite=pmh.NSITE,
    max_contacts=32,
    obs_dim_override=HUMANOID_PURE_OBS_DIM,
    timestep=pmh.TIMESTEP,
]

# Body indices — tree DFS, identical to MuJoCo's (asserted in the parity test).
comptime TORSO_BODY_IDX: Int = 1
comptime HEAD_BODY_IDX: Int = 2
comptime RIGHT_FOOT_BODY_IDX: Int = 7
comptime LEFT_FOOT_BODY_IDX: Int = 10
comptime RIGHT_HAND_BODY_IDX: Int = 13
comptime LEFT_HAND_BODY_IDX: Int = 16

comptime N_EXTREMITIES: Int = 4


def extremity_body_indices() -> List[Int]:
    """Bodies whose egocentric offsets form `Physics.extremities()`, IN ORDER.

    The reference iterates `for side in ('left_', 'right_')` then
    `for limb in ('hand', 'foot')`, so the observation order is left_hand,
    left_foot, right_hand, right_foot. Getting it wrong permutes 12
    observation slots without changing the shape — nothing but a value check
    would catch it, which is why the order lives here with the reason attached
    rather than being open-coded at the two call sites.

    A function rather than a `comptime` list: a comptime `List` is not
    `ImplicitlyCopyable`, so it cannot be materialized into a runtime loop.
    """
    return [
        LEFT_HAND_BODY_IDX,
        LEFT_FOOT_BODY_IDX,
        RIGHT_HAND_BODY_IDX,
        RIGHT_FOOT_BODY_IDX,
    ]

# The free root joint occupies qpos[0:7]; `joint_angles()` is qpos[7:].
comptime ROOT_QPOS_SIZE: Int = 7
