"""SO-ARM101 + k parked free-jointed slots — the P0 scene-budget probe models.

These are NOT tasks and never will be. They exist to answer ONE question,
which `docs/TASK_LAYER_PLAN.md` §3.3 asks and marks UNPRICED:

    a task family declares a FIXED scene budget, so every task carries every
    slot the family declares, active or not. What does an inactive slot cost?

A parked free body still owns its rows in CRBA and in the mass-matrix
factorisation, and still occupies a broadphase entry. The factorisation is
superlinear in `nv`, so this is not obviously free, and the budget SIZE is the
knob it decides. `docs/TASK_LAYER_PLAN.md` §3.3 says outright: do not commit to
a budget of 6 before P0 reports.

    k=0    nq  6   nv  6   ngeom 32   nbody  9    the control — the arm as it ships
    k=3    nq 27   nv 24   ngeom 35   nbody 12
    k=6    nq 48   nv 42   ngeom 38   nbody 15
    k=9    nq 69   nv 60   ngeom 41   nbody 18   <- the CEILING, see below

## ⚠⚠ NINE IS A HARDWARE CEILING, NOT A SWEEP CHOICE

The GPU Newton solver keeps three NV*NV matrices plus `Je` in threadgroup
memory. On an RTX 5090 (101,376 B/block) k=9 needs 86,676 B and fits; k=10
needs 101,940 B and does not; k=12 needs 136,212 B and ptxas said so by name.
`tools/tasks/gen_park_scenes.py` carries the table and the reason the existing
spill gate does not catch it. **A family on this hardware cannot declare more
than 9 free-jointed slots**, which is a budget answer P0 did not have to
measure — and it is DEVICE-DEPENDENT (an H100's 227 KB would allow more), which
is an uncomfortable property for a budget the design calls FIXED.

(measured with MuJoCo 3.10.0, 2026-09-02; the generator asserts them.)

## ⚠ THE SCENES ARE GENERATED — `tools/tasks/gen_park_scenes.py`

Do not hand-edit `assets/so101_park_k*.xml`. Regenerate with

    pixi run python tools/tasks/gen_park_scenes.py
    pixi run gen-dims

and read that generator's docstring before changing the park pose — the
obvious pose is measurably the worst one, and the reason is written down there
rather than here so it sits next to the code that emits it.

## ⚠ EVERY SCENE HAS ZERO CONTACTS AT REST, AND THAT IS LOAD-BEARING

The generator ASSERTS it. A parked slot that touches something would turn P0's
`nv` sweep into a contact sweep — a smooth, plausible, meaningless curve. The
same reason `max_contacts` stays PINNED across the sweep below.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.envs.robots.so101_park_dims import (
    SO101_PARK_K0_DIMS,
    SO101_PARK_K3_DIMS,
    SO101_PARK_K6_DIMS,
    SO101_PARK_K9_DIMS,
)
from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS


# ⚠⚠ PINNED ACROSS THE SWEEP, AND THAT IS LEG 1's WHOLE VALIDITY.
#
# `max_contacts` sizes the contact arrays AND bounds the PGS/Newton solve,
# which is superlinear in ACTIVE contacts — a quantity that has nothing to do
# with `nv`. Letting it track k would mean leg 1 varies TWO axes while its name
# claims one, which is the shape recorded as
# `feedback_the_gates_name_named_the_wrong_axis`: the fix is to add the
# fixed-axis leg, not to reinterpret the mixed one.
#
# SO-ARM101 ships 16. This is the k=12 value for EVERY k, so the k=0 control
# pays the same solver budget as the widest scene and the only thing moving
# across leg 1 is the slot count. Leg 2 varies THIS instead, at k=0.
comptime PARK_MAX_CONTACTS: Int = 16

# ⚠ THE ARM'S OWN HULL BUDGET, UNCHANGED. A parked slot is a BOX — a primitive,
# not a mesh — so it adds no hull vertices. Reusing the arm's constant rather
# than restating 33280 keeps the two from drifting the day the arm's meshes
# change; `fields_build` RAISES if it is too small, so a drift is loud.
comptime PARK_NMESH_VERTS: Int = SO_ARM101_NMESH_VERTS


comptime _k0 = SO101_PARK_K0_DIMS
comptime _k3 = SO101_PARK_K3_DIMS
comptime _k6 = SO101_PARK_K6_DIMS
comptime _k9 = SO101_PARK_K9_DIMS


comptime SoArm101ParkK0Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/robots/assets/so101_park_k0.xml",
    nbody=_k0.NBODY, njoint=_k0.NJOINT, nq=_k0.NQ, nv=_k0.NV,
    ngeom=_k0.NGEOM, nact=_k0.NACT, ntex=_k0.NTEX, nmat=_k0.NMAT,
    nlight=_k0.NLIGHT, ncam=_k0.NCAM, nsite=_k0.NSITE, neq=_k0.NEQ,
    nexclude=_k0.NEXCLUDE, npair=_k0.NPAIR, timestep=_k0.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=PARK_MAX_CONTACTS,
    action_dim_override=6,
]

comptime SoArm101ParkK3Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/robots/assets/so101_park_k3.xml",
    nbody=_k3.NBODY, njoint=_k3.NJOINT, nq=_k3.NQ, nv=_k3.NV,
    ngeom=_k3.NGEOM, nact=_k3.NACT, ntex=_k3.NTEX, nmat=_k3.NMAT,
    nlight=_k3.NLIGHT, ncam=_k3.NCAM, nsite=_k3.NSITE, neq=_k3.NEQ,
    nexclude=_k3.NEXCLUDE, npair=_k3.NPAIR, timestep=_k3.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=PARK_MAX_CONTACTS,
    action_dim_override=6,
]

comptime SoArm101ParkK6Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/robots/assets/so101_park_k6.xml",
    nbody=_k6.NBODY, njoint=_k6.NJOINT, nq=_k6.NQ, nv=_k6.NV,
    ngeom=_k6.NGEOM, nact=_k6.NACT, ntex=_k6.NTEX, nmat=_k6.NMAT,
    nlight=_k6.NLIGHT, ncam=_k6.NCAM, nsite=_k6.NSITE, neq=_k6.NEQ,
    nexclude=_k6.NEXCLUDE, npair=_k6.NPAIR, timestep=_k6.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=PARK_MAX_CONTACTS,
    action_dim_override=6,
]

comptime SoArm101ParkK9Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/robots/assets/so101_park_k9.xml",
    nbody=_k9.NBODY, njoint=_k9.NJOINT, nq=_k9.NQ, nv=_k9.NV,
    ngeom=_k9.NGEOM, nact=_k9.NACT, ntex=_k9.NTEX, nmat=_k9.NMAT,
    nlight=_k9.NLIGHT, ncam=_k9.NCAM, nsite=_k9.NSITE, neq=_k9.NEQ,
    nexclude=_k9.NEXCLUDE, npair=_k9.NPAIR, timestep=_k9.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=PARK_MAX_CONTACTS,
    action_dim_override=6,
]


# ── where the parked slots' state lives ────────────────────────────────────
#
# The arm is 6 hinges, laid out FIRST in both qpos and qvel because it is
# written first in the MJCF. Every parked slot follows, 7 qpos (3 pos + 4 quat)
# and 6 qvel each, in slot order.
#
# ⚠ THE TWO STRIDES DIFFER, and a quaternion is why. Indexing qvel with the
# qpos stride walks off the end of a k=12 model by 12 slots' worth — the exact
# arithmetic slip `physics3d` names `joint_qpos_adr` vs `joint_dof_adr` to
# avoid, and the reason those are two functions and not one.
comptime ARM_NQ: Int = 6
comptime ARM_NV: Int = 6
comptime SLOT_NQ: Int = 7
comptime SLOT_NV: Int = 6


@always_inline
fn slot_qpos_adr(slot: Int) -> Int:
    """First `qpos` index of parked slot `slot`."""
    return ARM_NQ + slot * SLOT_NQ


@always_inline
fn slot_qvel_adr(slot: Int) -> Int:
    """First `qvel` index of parked slot `slot`."""
    return ARM_NV + slot * SLOT_NV
