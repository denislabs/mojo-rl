"""`dm_control` `cheetah` model — port of `dm_control/suite/cheetah.xml`.

Body XML is the reference file verbatim apart from the three `<include>`
lines, which `merge_mjcf` splices in instead (same order as the reference).

The features this model needs that earlier domains did not:

  * `euler="0 <deg> 0"` on seven geoms — head and every leg segment. MuJoCo
    resolves it per `ResolveOrientation` with the compiler's `eulerseq`
    (default "xyz", lowercase => moving axes => post-multiply). Added to the
    parser 2026-07-29.
  * `<compiler settotalmass="14"/>`, which rescales every body mass after the
    geom-derived inertia pass.
  * NO `angle` attribute on `<compiler>`, so MuJoCo's default of DEGREE
    applies. That default was wrong in our parser until 2026-07-29 — the joint
    ranges here ("-30 60", "-230 50", ...) would otherwise be read as radians
    and the legs would swing unconstrained.
  * joint `stiffness` (a passive spring), 8 on the class and 60-240 per joint.

`<option timestep="0.01"/>` states no integrator => MuJoCo's Euler default,
and cheetah.py passes no `control_timestep`, so one env step is one physics
step.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.cheetah.cheetah_dims import DM_CHEETAH_DIMS





comptime pmc = DM_CHEETAH_DIMS

# obs = position (qpos[1:], nq-1 = 8) + velocity (nv = 9) = 17
comptime DMCheetahModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/cheetah.xml",
    nbody=pmc.NBODY, njoint=pmc.NJOINT, nq=pmc.NQ, nv=pmc.NV,
    ngeom=pmc.NGEOM, nact=pmc.NACT, ntex=pmc.NTEX, nmat=pmc.NMAT,
    nlight=pmc.NLIGHT, ncam=pmc.NCAM, nsite=pmc.NSITE,
    max_contacts=16,
    obs_dim_override=17,
    timestep=pmc.TIMESTEP,
]

# Body indices in worldbody DFS order.
comptime TORSO_BODY_IDX: Int = 1
