"""`manipulation/reassemble_5_bricks_random_order_features` as a COMPTIME model
def.

    nq 37   nv 33   njnt 13   ngeom 431   nbody 27   nsite 181

⚠⚠ FIVE BRICKS PLUS FIVE HINT TWINS. The real bricks are bodies 17, 19, 21, 23
and 25 and the translucent contactless copies sit between them — the same
stride-2 interleave as every other task in this family, just twice as long.

⚠⚠ THE BAKE WELDED BRICK 2, AND THAT IS NOT A TASK CONSTANT. This task draws
`initial_order` every episode and strips the freejoint from `initial_order[0]`,
so the reference's model MOVES. `FIXED_BRICK = 2` is simply what
`initialize_episode_mjcf` happened to draw when this XML was baked, and
`manipulation_reassemble`'s relabeling is what makes one bake serve every
episode. The gate asserts it against a freshly constructed, once-reset
reference env — which is exactly what the generator saw.

⚠ `obs_dim_override=112` — 5 `desired_order` + 42 robot + 5 x 13. The
`desired_order` observable EXISTS here (`randomize_desired_order=True`) and
sorts FIRST, so this is not `reassemble_3`'s layout at a different length.

⚠⚠ `max_contacts` IS 512. This is the first task where the reference raises its
OWN limits — `_Common.__init__` sets `nconmax = 400` and `njmax = 1200` for
`num_bricks > 3`, "since each stacked pair generates a large number of
contacts". Measured at a real reset: 97 contacts and 300 constraint rows for a
five-brick tower. 512 is headroom over that, and over the within-margin records
our narrow phase emits and MuJoCo does not count.

⚠ THIS IS THE LARGEST MODEL IN THE SUITE AND IT IS CLOSE TO THE PARSER'S
COMPTIME CAPS: 431 geoms against `MAX_COMPTIME_RENDER_GEOMS` 448, and 181 sites
against `MAX_COMPTIME_RENDER_SITES` 192. Ten Duplos at 41 geoms and 17 sites
each is most of that, so a SIX-brick task (`_Common` allows up to six) would
need both raised. They fail loudly and name themselves, so this is a note for
whoever adds one, not a latent bug.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_reassemble5_xml import reassemble5_xml

comptime pm = parse_xml(reassemble5_xml)

comptime Reassemble5Model = ModelDefFromXML[
    xml=reassemble5_xml,
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
    # ⚠ Every one of these is taken from `pm`, not defaulted — see
    # `manipulation_reach_def` for the individual consequences.
    nexclude=pm.NEXCLUDE,
    npair=pm.NPAIR,
    max_tendon=pm.NTENDON,
    max_condim=pm.MAX_CONDIM,
    max_equality=pm.NEQ * 6,
    max_contacts=512,
    obs_dim_override=112,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]
