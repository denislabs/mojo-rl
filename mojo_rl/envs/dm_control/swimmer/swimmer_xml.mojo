"""`dm_control` `swimmer` model — port of `dm_control/suite/swimmer.xml` PLUS the
procedural body chain that `swimmer._make_model(n_bodies)` builds on top of it.

This is the first domain in the port whose model is GENERATED, not written: the
reference parses `swimmer.xml` (head only), then appends `n_bodies - 1` nested
segment bodies, one `<motor>` per segment joint, and one `velocimeter`/`gyro`
sensor pair per segment site. `_swimmer_body_xml` below is that same loop, run
at comptime so `parse_xml` still sees a plain string literal's worth of XML.

    swimmer6  = 6 links  -> 5 segments, NQ = NV = 8,  obs 25
    swimmer15 = 15 links -> 14 segments, NQ = NV = 17, obs 61

TWO DELIBERATE SUBSTITUTIONS, both already-charted gaps:

1. THE TARGET BECOMES A MOCAP BODY (gap G4), exactly as reacher and finger
   needed. The reference declares it as a static worldbody geom

       <geom name="target" type="sphere" pos="1 1 .05" size=".1" material="target"/>

   and `Swimmer.initialize_episode` then rewrites `model.geom_pos['target',
   'x'/'y']` every episode. `fields.Model` is one SHARED, UNBATCHED tensor set,
   so a model write is a write for every env in the batch. A mocap body carries
   the pose in `d.mocap_pos`, which IS per-env state. The geom rides its body at
   the body origin, so `geom_xpos['target']` is exactly `d.xpos[TARGET_BODY]`.

       <body name="target" mocap="true" pos="1 1 .05">
         <geom name="target" type="sphere" size=".1" material="target" mass="0"/>
       </body>

   `mass="0"` is ours: the reference geom is static-in-world so its (default
   density) mass never reaches the dynamics, but on a mocap body it would reach
   `mj_fluid`'s per-body loop. It does not actually make `body_mass[target]`
   zero — `compute_inertia_from_geoms` skips a body with no MASS-CONTRIBUTING
   geom entirely, leaving the staging default (mass 1, diag inertia .01), the
   same value finger's mocap target carries. Inert either way: a mocap body has
   no DOF, so the fluid wrench projects to nothing and FK pins its velocity to
   zero, which is what makes the drag zero in the first place.

2. `light_pos['target_light']` is NOT tracked. `initialize_episode` moves the
   light with the target; light position is pure rendering and no observation
   or reward reads it.

WHAT IS NOT SUBSTITUTED, and matters: `<option density="3000">` with
`<flag contact="disable"/>`. Swimmer is the FIRST model in this repo to turn
the fluid path on, so `dynamics/fluid_forces.mojo` (MuJoCo's inertia-box model,
`mj_inertiaBoxFluidModel`) goes from dead code to the dominant force — with
contacts disabled and gravity irrelevant to a planar swimmer, drag is the ONLY
thing that converts joint torque into forward motion. The parity test gates it
directly.

GEOM ORDER, as always: ours is XML text order, MuJoCo's is sorted by body id.
Text order here is ground(0), then the head body's head(1) nose(2) eyes(3)
inertial(4) visual(5), then visual_i/inertial_i per segment, then target last —
so `NOSE_GEOM_IDX` is 2 for every `n_bodies`. The parity test pins it by
checking the record's body id and local pos rather than trusting the count.

⚠ `geom head` is an `ellipsoid`, and `_geom_type_from_str` has no ellipsoid
case — it falls through to sphere SILENTLY (the same trap finger's touch sites
hit). Harmless here twice over: the geom carries `mass="0"` so it contributes
no inertia, and `<flag contact="disable"/>` means no narrow phase ever reads its
shape. Pinned by the parity test so it cannot rot into something load-bearing.

Reference: references/dm_control-main/dm_control/suite/swimmer.py + .xml
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from mojo_rl.envs.dm_control.swimmer.swimmer_dims import (
    DM_SWIMMER6_DIMS,
    DM_SWIMMER15_DIMS,
)


def _f(x: Float64) -> String:
    """MJCF float literal. `String(Float64)` renders `-60.0` / `24.0`, which is
    what `'{} {}'.format(...)` produces in `_make_model` too."""
    return String(x)


def _swimmer_body_xml(n_bodies: Int) -> String:
    """`swimmer.xml` + `_make_model(n_bodies)`, as one string.

    The reference's `if n_bodies < 3: raise ValueError` is NOT reproduced: a
    comptime function cannot raise, and the only two call sites are the
    literals 6 and 15 below. A smaller n would still generate valid MJCF (the
    reference's floor is about the swimmer being able to swim, not about the
    model compiling), so there is nothing here that fails silently.
    """
    # `scale = n_bodies / 6.0` applied to every `mode="trackcom"` camera that
    # is a direct child of a direct-child body of worldbody (the head's two
    # tracking cameras). Visual only, kept so the model text stays diffable
    # against `etree.tostring(_make_model(n))`.
    var scale = Float64(n_bodies) / 6.0
    var t1 = _f(0.0 * scale) + " " + _f(-0.2 * scale) + " " + _f(0.5 * scale)
    var t2 = _f(-0.9 * scale) + " " + _f(0.5 * scale) + " " + _f(0.15 * scale)

    # `joint_limit = 360.0 / n_bodies`, in degrees, same for every segment.
    var jl = 360.0 / Float64(n_bodies)
    var jrange = _f(-jl) + " " + _f(jl)

    var s = String(
        """
<mujoco model="swimmer">
  <option timestep="0.002" density="3000">
    <flag contact="disable"/>
  </option>

  <default>
    <default class="swimmer">
      <joint type="hinge" pos="0 -.05 0" axis="0 0 1" limited="true" solreflimit=".05 1" solimplimit="0 .8 .1" armature="1e-6"/>
      <default class="inertial">
        <geom type="box" size=".001 .05 .01" rgba="0 0 0 0" mass=".01"/>
      </default>
      <default class="visual">
        <geom type="capsule" size=".01" fromto="0 -.05 0 0 .05 0" material="self" mass="0"/>
      </default>
      <site size=".01" rgba="0 0 0 0"/>
    </default>
    <default class="free">
      <joint limited="false" stiffness="0" armature="0"/>
    </default>
    <motor gear="5e-4" ctrllimited="true" ctrlrange="-1 1"/>
  </default>

  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1" material="grid"/>
    <body name="head" pos="0 0 .05" childclass="swimmer">
      <light name="light_1" diffuse=".8 .8 .8" pos="0 0 1.5"/>
      <geom name="head" type="ellipsoid" size=".02 .04 .017" pos="0 -.022 0"  material="self" mass="0"/>
      <geom name="nose" type="sphere" pos="0 -.06 0" size=".004" material="effector" mass="0"/>
      <geom name="eyes" type="capsule" fromto="-.006 -.054 .005 .006 -.054 .005" size=".004" material="eye" mass="0"/>
      <camera name="tracking1" pos="""
    )
    s += '"' + t1 + '"'
    s += (
        ' xyaxes="1 0 0 0 1 1" mode="trackcom" fovy="60"/>\n'
        '      <camera name="tracking2" pos='
    )
    s += '"' + t2 + '"'
    s += (
        ' xyaxes="0 -1 0 .3 0 1" mode="trackcom" fovy="60"/>\n'
        '      <camera name="eyes" pos="0 -.058 .005" xyaxes="-1 0 0 0 0 1"/>\n'
        '      <joint name="rootx" class="free" type="slide" axis="1 0 0"'
        ' pos="0 -.05 0"/>\n'
        '      <joint name="rooty" class="free" type="slide" axis="0 1 0"'
        ' pos="0 -.05 0"/>\n'
        '      <joint name="rootz" class="free" type="hinge" axis="0 0 1"'
        ' pos="0 -.05 0"/>\n'
        '      <geom name="inertial" class="inertial"/>\n'
        '      <geom name="visual" class="visual"/>\n'
        '      <site name="head"/>\n'
    )

    # ── The `_make_body` loop. Each child is appended to the PREVIOUS body, so
    # the chain nests; children within a body keep `_make_body`'s own order
    # (geom visual, geom inertial, site, joint), which fixes our geom indices.
    for i in range(n_bodies - 1):
        var ix = String(i)
        var pad = String("      ") + String("  ") * i
        s += pad + '<body name="segment_' + ix + '" pos="0 .1 0">\n'
        s += pad + '  <geom class="visual" name="visual_' + ix + '"/>\n'
        s += pad + '  <geom class="inertial" name="inertial_' + ix + '"/>\n'
        s += pad + '  <site name="site_' + ix + '"/>\n'
        s += (
            pad + '  <joint name="joint_' + ix + '" range="' + jrange + '"/>\n'
        )
    for i in range(n_bodies - 1):
        var pad = String("      ") + String("  ") * (n_bodies - 2 - i)
        s += pad + "</body>\n"

    s += (
        "    </body>\n"
        '    <body name="target" mocap="true" pos="1 1 .05">\n'
        '      <geom name="target" type="sphere" size=".1" material="target"'
        ' mass="0"/>\n'
        "    </body>\n"
        '    <light name="target_light" diffuse="1 1 1" pos="1 1 1.5"/>\n'
        "  </worldbody>\n\n"
        "  <actuator>\n"
    )
    for i in range(n_bodies - 1):
        var ix = String(i)
        s += (
            '    <motor name="motor_'
            + ix
            + '" joint="joint_'
            + ix
            + '"/>\n'
        )
    s += "  </actuator>\n\n  <sensor>\n"
    s += (
        '    <framepos name="nose_pos" objtype="geom" objname="nose"/>\n'
        '    <framepos name="target_pos" objtype="geom" objname="target"/>\n'
        '    <framexaxis name="head_xaxis" objtype="xbody" objname="head"/>\n'
        '    <frameyaxis name="head_yaxis" objtype="xbody" objname="head"/>\n'
        '    <velocimeter name="head_vel" site="head"/>\n'
        '    <gyro name="head_gyro" site="head"/>\n'
    )
    for i in range(n_bodies - 1):
        var ix = String(i)
        s += (
            '    <velocimeter name="velocimeter_'
            + ix
            + '" site="site_'
            + ix
            + '"/>\n'
            '    <gyro name="gyro_'
            + ix
            + '" site="site_'
            + ix
            + '"/>\n'
        )
    s += "  </sensor>\n</mujoco>\n"
    return s


# `<sensor>` is not an accumulator in `merge_mjcf` (nor parsed by the full
# parser), so every sensor above is DROPPED on the way in. Kept verbatim
# because it is the reference's declaration of what the observation reads, and
# the config below reproduces each one from the fields it would have been built
# from: `framepos` of a geom -> body xpos + the geom's local pos, `framexaxis`
# -> a column of the body's rotation, `velocimeter`/`gyro` -> the site-frame
# body velocity (`sensors/frame_vel.mojo`).
comptime _swimmer6_body = _swimmer_body_xml(6)
comptime _swimmer15_body = _swimmer_body_xml(15)

comptime dm_swimmer6_xml = merge_mjcf(
    dm_visual_xml, dm_skybox_xml, dm_materials_xml, _swimmer6_body
)
comptime dm_swimmer15_xml = merge_mjcf(
    dm_visual_xml, dm_skybox_xml, dm_materials_xml, _swimmer15_body
)

comptime ps6 = DM_SWIMMER6_DIMS

comptime ps15 = DM_SWIMMER15_DIMS

# obs = joints (n-1) + to_target (2) + body_velocities (3n)
comptime DMSwimmer6Model = ModelDefFromXML[
    xml=dm_swimmer6_xml,
    nbody = ps6.NBODY, njoint = ps6.NJOINT, nq = ps6.NQ, nv = ps6.NV,
    ngeom = ps6.NGEOM, nact = ps6.NACT, ntex = ps6.NTEX, nmat = ps6.NMAT,
    nlight = ps6.NLIGHT, ncam = ps6.NCAM, nsite = ps6.NSITE,
    max_contacts=1,
    obs_dim_override=25,
    timestep = ps6.TIMESTEP,
]

comptime DMSwimmer15Model = ModelDefFromXML[
    xml=dm_swimmer15_xml,
    nbody = ps15.NBODY, njoint = ps15.NJOINT, nq = ps15.NQ, nv = ps15.NV,
    ngeom = ps15.NGEOM, nact = ps15.NACT, ntex = ps15.NTEX, nmat = ps15.NMAT,
    nlight = ps15.NLIGHT, ncam = ps15.NCAM, nsite = ps15.NSITE,
    max_contacts=1,
    obs_dim_override=61,
    timestep = ps15.TIMESTEP,
]


# ── Indices. Everything here is independent of `n_bodies` except the target,
# which is always last, so the config stays generic in NBODY.
#
#   body 0            world
#   body 1            head
#   body 2 .. NBODY-2 segment_0 .. segment_{n-2}
#   body NBODY-1      target (mocap)
comptime HEAD_BODY_IDX: Int = 1
comptime FIRST_SEGMENT_BODY_IDX: Int = 2

# Geoms in XML text order (see the module docstring).
comptime GROUND_GEOM_IDX: Int = 0
comptime HEAD_GEOM_IDX: Int = 1
comptime NOSE_GEOM_IDX: Int = 2

# qpos/qvel: three root DOFs, then one hinge per segment. `Physics.joints()` is
# `qpos[3:]`, i.e. everything after these.
comptime N_ROOT_DOF: Int = 3

# `geom_size['target', 0]` — the reward's `bounds`/`margin` scale. Constant in
# the reference too (nothing writes geom_size for swimmer).
comptime TARGET_SIZE: Float64 = 0.1

# The target's z never moves: `initialize_episode` writes only x and y.
comptime TARGET_Z: Float64 = 0.05
