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
    action_dim_override=6,
]
