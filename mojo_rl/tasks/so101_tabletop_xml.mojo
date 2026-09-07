"""`so101_tabletop` — the family as a COMPILE UNIT.

    families/so101_tabletop.family   the slot table (authored)
    scenes/so101_tabletop.xml        composed from it (GENERATED)
    so101_tabletop_dims.mojo         read off MuJoCo (GENERATED, CI-checked)
    THIS FILE                        the comptime model def

⚠ THIS IS THE WHOLE POINT OF THE FIXED SCENE BUDGET. Every task in this family
instantiates every slot, so these dimensions are CONSTANT across the family and
the GPU leg is ONE monomorphisation with different tasks in different lanes
(`TASK_LAYER_PLAN.md` §3.1). A task that needs another object is a different
family and a rebuild — and that is correct, because it is a different model.

⚠ TO CHANGE THE SLOT TABLE: edit the `.family`, then

    pixi run gen-family-scenes && pixi run gen-dims

`gen-dims-check` and `gen-family-scenes-check` fail the build if either is
stale, so the budget cannot drift from what MuJoCo says the scene is.

Measured (MuJoCo 3.10.0): nbody 13, njnt 9, nq 27, nv 24, ngeom 37, and
**four kinematic trees with dof blocks [6, 6, 6, 6]** — the robot plus one per
free slot. That block structure is not incidental: it is exactly what the
block-diagonal solver work exploits
(`docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md`), and it is why a family
is affordable at all. Sparse `nC` is 39 against a dense `nv*nv` of 576.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.tasks.so101_tabletop_dims import SO101_TABLETOP_DIMS


comptime _pm = SO101_TABLETOP_DIMS

# ⚠ A BUDGET, AND IT IS NOT YET MEASURED. `max_contacts` bounds the constraint
# solve and, through `ME = 4*MC + 2*NJOINT + NV`, sizes the solver's shared
# memory. SO-ARM101 alone ships 16; this scene adds three free props, a table
# and a floor, so a pick-and-place can plausibly carry more contacts at once.
#
# ⚠⚠ TOO SMALL IS NOT AN ERROR, IT IS A SILENTLY EASIER PROBLEM — an
# overflowed budget DROPS contacts. Too large costs solver time and shared
# memory. 32 is a starting point chosen to be comfortably above what a
# three-prop tabletop should reach; P2's three real tasks are what should
# measure it (print the contacts actually solved, not the budget).
comptime SO101_TABLETOP_MAX_CONTACTS: Int = 32


# ── THE OBSERVATION'S WIDTH, AND WHY IT IS NOT THE DEFAULT ────────────────
#
# `ModelDefFromXML`'s default is `nq - obs_qpos_skip + nv` with
# `obs_qpos_skip = 1`. That default is written for a FLOATING-BASE Gym model,
# where `qpos[0]` is the root x the reward already prices and the observation
# is deliberately translation-invariant.
#
# ⚠⚠ THIS FAMILY HAS NO FLOATING BASE. SO-101 is bolted to the world and
# `qpos[0]` is `shoulder_pan` — the joint that decides which way the arm is
# pointing. The default silently dropped it, and a policy that cannot see its
# own base rotation is not a policy with a bad observation, it is a policy
# with an unobservable state. Nothing had trained on this family yet, which is
# the only reason this is a correction and not a regression.
#
# The three extra words are the ACTIVE MASK, one per free slot —
# `TASK_LAYER_PLAN.md` §3.4, written by `So101TabletopConfig.
# custom_extract_obs_gpu`. Layout:
#
#   [0            .. NQ)               qpos, IN FULL
#   [NQ           .. NQ+NV)            qvel
#   [NQ+NV        .. NQ+NV+N_FREE)     1.0 if that free slot is active
#   [.. +3)                            the gripper site, in world coordinates
#   [.. +3)                            goal SUBJECT minus the gripper
#   [.. +3)                            goal TARGET minus the subject
#
# ⚠ AN INACTIVE SLOT'S POSE AND VELOCITY WORDS ARE ZEROED, not left at the
# park pose. See `tasks/obs.mojo` for why both halves are needed.
#
# ⚠⚠ THE LAST NINE WORDS ARE THE REWARD'S OWN GEOMETRY, AND WITHOUT THEM THE
# POLICY COULD NOT SEE HALF ITS REWARD. `So101TabletopConfig`'s shaped term
# pays `SHAPE_W_REACH` on the gripper-to-subject distance — and the gripper's
# Cartesian position is FORWARD KINEMATICS over six joint angles, which the
# observation above does not contain. Measured on a 190k-step `gather` run:
# the critic converged (mean_q 33.8, critic_loss 0.30) and the return did not
# move at all (-6.7 .. -9.5, no trend), with `mean_reward` pinned at -0.026
# from the first sample to the last. Roughly half the shaped reward was
# computed from a quantity the network would have had to learn FK to recover.
#
# ⚠ `SoArm101ReachConfig` — the config that DOES train on this robot — has had
# this all along: its 21 words are qpos(6) + qvel(6) + ee(3) + target(3) +
# ee_to_target(3). The relative vectors are in the observation on purpose.
#
# ⚠ APPENDED, NOT INSERTED, so `OBS_MASK_BASE` and every index before it are
# unchanged and the mask gates keep testing what they tested.
comptime SO101_TABLETOP_N_FREE_SLOTS: Int = 3
comptime SO101_TABLETOP_N_GOAL_WORDS: Int = 9
comptime SO101_TABLETOP_OBS_DIM: Int = (
    _pm.NQ + _pm.NV + SO101_TABLETOP_N_FREE_SLOTS
    + SO101_TABLETOP_N_GOAL_WORDS
)

comptime So101TabletopModel = ModelDefFromXML[
    xml_path="mojo_rl/tasks/scenes/so101_tabletop.xml",
    nbody=_pm.NBODY,
    njoint=_pm.NJOINT,
    nq=_pm.NQ,
    nv=_pm.NV,
    ngeom=_pm.NGEOM,
    nact=_pm.NACT,
    ntex=_pm.NTEX,
    nmat=_pm.NMAT,
    nlight=_pm.NLIGHT,
    ncam=_pm.NCAM,
    nsite=_pm.NSITE,
    neq=_pm.NEQ,
    nexclude=_pm.NEXCLUDE,
    npair=_pm.NPAIR,
    timestep=_pm.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=SO101_TABLETOP_MAX_CONTACTS,
    obs_dim_override=SO101_TABLETOP_OBS_DIM,
    action_dim_override=6,
]
