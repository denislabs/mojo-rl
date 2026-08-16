"""`dm_control` `dog fetch` — the model Phase 5 needs (port of `dog.py`).

`fetch` is the one dog task that calls `get_model_and_assets(remove_ball=False)`
and so keeps what stand/walk/trot/run delete: the `ball` body and its
`ball_root` free joint, the `target` geom, four `wall_*` geoms and two cameras.
It also takes `floor_size`'s DEFAULT of 10 rather than
`move_speed * _DEFAULT_TIME_LIMIT`, because `fetch()` overrides only
`remove_ball` — which is why this model is not one of `dog_xml.mojo`'s three.

    stand / walk   floor 15      trot  floor 45      run  floor 135
    fetch          floor 10

⚠ GENERATED, like `dog_xml.mojo`, by `tests/dm_control/dog_ref.py::port_fragment`
— here with `(floor_size=10, remove_ball=False)`. Regenerate rather than
hand-edit; that function is where the port's text-level deviations live.

⚠ SEPARATE FILE, NOT A FOURTH CONSTANT IN `dog_xml.mojo`. The fetch body is its
own ~70 kB of MJCF and dog_xml.mojo is already ~80 kB; §15 of the port plan
measured comptime XML as the dominant build cost, so folding this in would tax
stand/walk/trot/run — which never use it — on every build.

THE `tennis_ball` DEVIATION, and why it is here rather than in the generator
----------------------------------------------------------------------------
dog.xml dresses the ball with

    <texture name="tennis_ball" file="tennis_ball.png" gridsize="3 4" .../>
    <material name="tennis_ball" texture="tennis_ball"/>

`port_fragment` strips both, because they name a PNG ON DISK and a ported XML
carries no asset bundle — MuJoCo cannot compile the string with them present.
That was free for stand/walk/trot/run, which delete the ball; fetch keeps it,
and its geom still says `material="tennis_ball"`. So the material is
re-supplied below as a FLAT COLOUR.

⚠ THIS CHANGES `mat_rgba` AND THE BALL'S `geom_rgba`, and nothing else. It is
rendering-only — no MuJoCo table that feeds the dynamics reads a material — but
it IS a real difference from the reference model, so the parity gate must
exempt those two columns EXPLICITLY rather than pass by luck.

⚠ VERIFIED TO COMPILE AND TO MATCH THE REFERENCE DIMENSIONS before this file
was written (nbody 63, njnt 75, nq 87, nv 85, ngeom 134, nsite 12, nu 38,
ntendon 8), by building the same text with these assets in MuJoCo directly.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .dog_xml import DOG_FRAME_SKIP, DOG_MAX_STEPS
from mojo_rl.envs.dm_control.dog.dog_fetch_dims import DM_DOG_FETCH_DIMS


# The flat-colour stand-in for dog.xml's PNG-textured tennis ball. Merged as a
# separate asset fragment so the body text below stays byte-for-byte what the
# generator emits.





comptime dfp = DM_DOG_FETCH_DIMS

# --- observation layout ------------------------------------------------------
#
# `Fetch.get_observation_components` is Stand's plus two entries:
#
#   (Stand)             223   see dog_xml.mojo's layout note
#   ball_state            6   ball_in_head_frame: position THEN velocity,
#                             both rotated into the head site frame
#   target_position       3   target_in_head_frame
#                       -----
#                         232
comptime DOG_FETCH_OBS_DIM: Int = 232


comptime DMDogFetchModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/dog_fetch.xml",
    nbody=dfp.NBODY, njoint=dfp.NJOINT, nq=dfp.NQ, nv=dfp.NV,
    ngeom=dfp.NGEOM, nact=dfp.NACT, ntex=dfp.NTEX, nmat=dfp.NMAT,
    nlight=dfp.NLIGHT, ncam=dfp.NCAM, nsite=dfp.NSITE,
    max_tendon=dfp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    # dog's own four feet plus the ball against the floor, the walls, and
    # whatever limb is nudging it.
    max_contacts=28,
    obs_dim_override=DOG_FETCH_OBS_DIM,
    obs_qpos_skip=0,
    neq=dfp.NEQ,
    nexclude=dfp.NEXCLUDE,
    timestep=dfp.TIMESTEP,
    # ⚠ DERIVED, NEVER HAND-WRITTEN. The ball is `class="bouncy"` and the
    # target `class="velcro"`, both condim=6; without this they are silently
    # downgraded to four pyramid edges and the ball spins and rolls unopposed
    # (defect 004fe439, and defect 8 in a second dress). dog's own geoms
    # already carry condim 6 on 42 of 128, so this is not new to fetch — it is
    # simply not optional.
    max_condim=dfp.MAX_CONDIM,
    # MuJoCo `m->na` — see dog_xml.
    na = 38,
]


# --- indices, read out of a COMPILED mjModel, never counted by hand ----------
#
# The ball is appended last, so dog's own qpos/qvel layout is untouched and
# only the tail is new — the same property quadruped fetch relies on.
comptime FETCH_BALL_BODY_IDX: Int = 62
comptime FETCH_BALL_QPOS_0: Int = 80
comptime FETCH_BALL_DOF_0: Int = 79

# `target` is declared in the worldbody BEFORE the dog, so it takes geom id 1
# (the floor is 0) and the ball, added last, is 133.
comptime FETCH_GEOM_BALL: Int = 133
comptime FETCH_GEOM_TARGET: Int = 1
comptime FETCH_GEOM_FLOOR: Int = 0

# Sites are UNCHANGED from stand — fetch adds none, so `head`, `upper_bite`
# and `lower_bite` keep their ids and dog_config's constants stay valid.
comptime FETCH_SITE_HEAD: Int = 5
comptime FETCH_SITE_UPPER_BITE: Int = 6
comptime FETCH_SITE_LOWER_BITE: Int = 7

# --- reward geometry, read from the compiled model --------------------------
# `bite_radius`  = site_size['upper_bite', 0]
# `target_radius`= geom_size['target', 0]
# `bring_margin` = geom_size['floor', 0]  (the floor HALF-extent, i.e. 10)
comptime FETCH_BITE_RADIUS: Float64 = 0.005
comptime FETCH_TARGET_RADIUS: Float64 = 0.1
comptime FETCH_BRING_MARGIN: Float64 = 10.0

# `Fetch.initialize_episode` throws the ball from 0.75 * floor half-extent.
comptime FETCH_THROW_RADIUS: Float64 = 0.75 * FETCH_BRING_MARGIN
comptime FETCH_THROW_HEIGHT_MAX: Float64 = 3.0
comptime FETCH_THROW_SPEED_MAX: Float64 = 5.0
comptime FETCH_BALL_SPAWN_Z: Float64 = 0.05
