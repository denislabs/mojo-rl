"""`dm_control` `walker` model — port of `dm_control/suite/walker.xml`.

Body XML is the reference file verbatim apart from the three `<include>`
lines, which `merge_mjcf` splices in instead (same order as the reference).

Unlike cartpole this model is NOT built procedurally — walker.py loads the
file as-is for all three tasks, so there is a single model here.

Two MJCF features this file needs that older ports did not:

  * `zaxis="1 0 0"` on the foot capsules — "point local +Z along this vector",
    MuJoCo's `mjuu_z2quat`. Previously unparsed, so the feet came out oriented
    along +Z (upright pegs instead of forward-pointing soles), which changes
    both their inertia tensors and their contact geometry.
  * a top-level unnamed `<default>` **plus** a named `<default class="walker">`,
    with `childclass="walker"` on the torso. Bare `<joint name="right_hip"
    range="-20 100"/>` elements take their type (hinge, MuJoCo's default),
    axis, damping, armature and solimplimit from those two blocks combined.

`<option timestep="0.0025"/>` states no integrator, so MuJoCo's Euler default
applies — as with pendulum, and unlike cartpole's explicit RK4.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.walker.walker_dims import DM_WALKER_DIMS





comptime pmw = DM_WALKER_DIMS

# obs = orientations (nbody-1 bodies x [xx, xz]) + height (1) + velocity (nv)
#     = 7*2 + 1 + 9 = 24
comptime DMWalkerModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/walker.xml",
    nbody=pmw.NBODY, njoint=pmw.NJOINT, nq=pmw.NQ, nv=pmw.NV,
    ngeom=pmw.NGEOM, nact=pmw.NACT, ntex=pmw.NTEX, nmat=pmw.NMAT,
    nlight=pmw.NLIGHT, ncam=pmw.NCAM, nsite=pmw.NSITE,
    max_contacts=16,
    obs_dim_override=24,
    timestep=pmw.TIMESTEP,
]

# Body indices in worldbody DFS order.
comptime TORSO_BODY_IDX: Int = 1
