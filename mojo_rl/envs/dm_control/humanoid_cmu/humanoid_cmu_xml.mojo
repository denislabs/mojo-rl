"""`dm_control` `humanoid_CMU` model — port of `dm_control/suite/humanoid_CMU.xml`.

Verbatim apart from the `<include>` lines and the `<sensor>` block. The body of
this file was EXTRACTED FROM THE REFERENCE PROGRAMMATICALLY rather than
transcribed, so the 17 kB of joint ranges and unnormalized quaternions below
cannot carry a typo the eye would miss.

THE SENSOR BLOCK IS DROPPED, exactly as in `humanoid_xml`. The XML declares
eight sensors and the three tasks read ONE of them — `grep sensordata
humanoid_CMU.py` returns a single line, `thorax_subtreelinvel`, which we
compute from `Data.xvel` via `sensors.subtree_linvel`. (`merge_mjcf` does not
carry a `<sensor>` section anyway.) All five SITES stay, so `nsite` still
matches MuJoCo's 5 and a future `touch` port has its zones.

WHAT THIS MODEL EXERCISES THAT NO EARLIER PORTED DOMAIN DOES:

  * A NAMED TOP-LEVEL DEFAULT CLASS, `<default class="main">`. It is the only
    one in all nineteen suite domains — every other model opens with a bare
    `<default>`. `_extract_section_inner` is depth-counted so the block is
    read whole, and `_strip_nested_defaults` removes the nested `humanoid`
    class from the root lookup; but this is the first model that proves it,
    and a regression here would silently hand every geom and joint the WRONG
    DEFAULTS rather than fail.
  * FIFTY-SIX ACTUATORS AND FIFTY-SEVEN JOINTS, against a comptime parser that
    recorded 32 of each until 2026-08-03. Both scans were `while count < CAP`
    while `ParsedModel` counted the tags independently, so before the widening
    this model would have built, exposed all 56 controls, and silently applied
    zero force through 24 of them. See `MAX_COMPTIME_ACTUATORS`.

WARNING: COUNT MODEL ELEMENTS WITH MuJoCo, NOT WITH grep. `grep -c '<joint '`
on the reference says 60 and `mjModel.njnt` says 57; `grep -c '<motor '` says
57 and `nu` says 56; `<geom` says 52 and `ngeom` says 50; `<site` says 6 and
`nsite` says 5. Every difference is an element sitting inside a `<default>`
block. The first draft of this port sized three comptime caps off those greps.

Measured against MuJoCo 3.10.0:
    nq 63   nv 62   nu 56   na 0    nbody 32   nsite 5   nexclude 5
    njnt 57 (1 free + 56 hinge)
    ngeom 50 (1 plane, 8 sphere, 39 capsule, 2 ellipsoid)

Body order is the tree DFS in both engines, so our indices match MuJoCo's here
— asserted in the parity test rather than assumed.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_dims import (
    DM_HUMANOID_CMU_DIMS,
)





comptime pmhc = DM_HUMANOID_CMU_DIMS

# observation = joint_angles (nq-7 = 56) + head_height (1) + extremities (12)
#             + torso_vertical (3) + com_velocity (3) + velocity (nv = 62)
comptime HUMANOID_CMU_OBS_DIM: Int = 137

comptime DMHumanoidCMUModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/humanoid_cmu.xml",
    nbody=pmhc.NBODY, njoint=pmhc.NJOINT, nq=pmhc.NQ, nv=pmhc.NV,
    ngeom=pmhc.NGEOM, nact=pmhc.NACT, ntex=pmhc.NTEX, nmat=pmhc.NMAT,
    nlight=pmhc.NLIGHT, ncam=pmhc.NCAM, nsite=pmhc.NSITE,
    # `<contact><exclude>` x5. ⚠ THIS PARAMETER DEFAULTS TO 0 AND NOTHING
    # CHECKS IT. Omitting it builds a model with no exclusions at all, which
    # simulates fine and quietly collides the five body pairs MuJoCo never
    # collides (the two clavicles against each other and against both neck
    # segments). The only symptom is a dynamics divergence you would go looking
    # for in the solver. It was omitted in the first draft of this file, and
    # `merge_mjcf` was ALSO dropping the whole `<contact>` section — two
    # independent zeros multiplying to the same silent answer.
    nexclude=pmhc.NEXCLUDE,
    # humanoid uses 32. Raised here because the CMU skeleton has fingers,
    # thumbs and toes — many more small geoms in close proximity — and an
    # undersized bound DROPS contacts silently. The parity test reports
    # MuJoCo's max `ncon` over its rollouts so this number stays evidence-based.
    max_contacts=64,
    obs_dim_override=HUMANOID_CMU_OBS_DIM,
    timestep=pmhc.TIMESTEP,
]

# Body indices — tree DFS, identical to MuJoCo's (asserted in the parity test).
#
# WARNING: THE REFERENCE BODY IS `thorax`, NOT `torso`. humanoid_CMU has no
# body named torso at all, and `humanoid_CMU.py`'s `torso_vertical_orientation`
# reads the THORAX. Reusing `humanoid`'s TORSO_BODY_IDX = 1 would silently read
# the free-jointed ROOT body instead.
comptime THORAX_BODY_IDX: Int = 14
comptime HEAD_BODY_IDX: Int = 17
comptime LEFT_HAND_BODY_IDX: Int = 22
comptime LEFT_FOOT_BODY_IDX: Int = 5
comptime RIGHT_HAND_BODY_IDX: Int = 29
comptime RIGHT_FOOT_BODY_IDX: Int = 10

comptime N_EXTREMITIES: Int = 4


def extremity_body_indices() -> List[Int]:
    """Bodies whose egocentric offsets form `Physics.extremities()`, IN ORDER.

    The reference iterates `for side in ('l', 'r')` then
    `for limb in ('hand', 'foot')`, so the observation order is lhand, lfoot,
    rhand, rfoot. Getting it wrong permutes 12 observation slots without
    changing the shape — nothing but a value check would catch it.

    WARNING: the SIDE PREFIXES differ from `humanoid`'s ('left_'/'right_') and
    the body names differ; the ORDER happens to coincide. Do not infer one from
    the other.

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
