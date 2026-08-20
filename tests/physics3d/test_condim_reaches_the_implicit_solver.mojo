"""The condim a model declares must reach the solver that steps it.

    pixi run mojo run -I . tests/physics3d/test_condim_reaches_the_implicit_solver.mojo

THREE DEFECTS, ONE CHAIN — a model declares condim 6, and by the time the
contact is solved it is condim 3. `_contact_solve_env` CLAMPS each contact's
own condim down to `MAX_CONDIM` silently, in both cone branches, so every
break in the chain is an absence rather than an error.

  1. `ImplicitIntegrator` HAD NO `MAX_CONDIM` PARAMETER AT ALL. `solve_newton`
     defaults it to 3, so every model stepped by an implicit integrator solved
     its contacts at condim 3 whatever the file said. `EulerIntegrator` has
     carried the parameter since the elliptic cone was generalised; its twin
     was never given it. ⚠ THE TELL IS THE ONE THIS TREE ALREADY KNOWS: a
     comptime parameter that reaches a function and is read by only one of its
     callers. Both integrators still forward it to `solve_newton` ALONE —
     `solve_cg`, `solve_island_pgs` and `solve_contacts` have no such
     parameter, and `solve_contacts` calls `_contact_solve_env` (which does)
     without one. Those three remain condim-3-only; the studio builds `newton`
     only, which is why this was the call that had to change first.

  2. `fmd.max_condim` IGNORED `<contact><pair condim=...>`. It was computed
     from the GEOMS, and a pair's condim does not come from its geoms and is
     not bounded by them — it REPLACES what the mask path would compute.
     apptronik_apollo is exactly that model: every geom it owns is
     `condim="1"` (its root `<default>` says so) and the soles reach condim 6
     only through `<pair condim="6" .../>`. The recorder returned its floor
     of 3 while MuJoCo reports `contact.dim == 6` on all four foot contacts.
     ⚠ AND THE BLOCK HAD TO MOVE: it sat beside the geom walk, where
     `result.pairs` is still EMPTY, so adding the scan in place would have
     compiled, run, and changed nothing.

  3. THE STUDIO BUILT 3 AND NOTHING CHECKED. `fmd.max_condim` has existed
     since spot's feet were found being solved at 3, precisely so a caller
     could compare it against the bound it built — and no caller ever did.
     `STUDIO_MAX_CONDIM` is now 6 and `studio_condim_warning` is the check the
     recorder was always for.

⚠ MEASURED, apptronik_apollo from its `stand` keyframe, worst |d(qpos)| against
MuJoCo 3.10.0 (`<pair condim="6">` soles, pyramidal, implicitfast):

              before      after
      1 step  1.856e-03   2.220e-16
      5       8.900e-03   7.772e-16
     20       1.353e-02   4.441e-15

⚠⚠ THE CONTROL THAT NAMED THE CAUSE, and the reason this is not a solver
tolerance story: editing apollo's two sole pairs from `condim="6"` to
`condim="3"` — in the file BOTH engines read — drops the unfixed engine's
1-step divergence from 1.856e-03 to 5.551e-17. The whole error was the rows
that were never built. The same control exonerated `<option iterations="4"
ls_iterations="10">`: removing them moves MuJoCo's answer not at all, so it
had converged at 4.

⚠ SIX IS NOT FREE — the first version of the fix claimed it was. Interleaved
in one process, min of five rounds, 300 steps: barkour (needs 3) 118.3 ->
132.5 us/step, spot (needs 6) 69.9 -> 86.6. The extra rows are not BUILT for a
condim-3 contact, but `MAX_CONDIM` sizes the region they are built into. One
bound for every model is still the right trade for the studio (12% on a tool
that steps at 8 kHz and renders at 60 Hz, against taking four integrator
instantiations to eight); a batched trainer should pass `fmd.max_condim`.

⚠ 46 OF THE TREE'S 142 MODELS need more than condim 3 — 29 Menagerie scenes
and 17 in-repo assets, including spot, apollo, all four unitree quadrupeds,
both shadow hands and every dm_control dog.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.integrator.implicit import ImplicitIntegrator
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.studio.stepping import (
    STUDIO_MAX_CONDIM, StudioImpFastPyr, studio_condim_warning,
)
from mojo_rl.physics3d.gpu.constants import (
    KEY_META_SIZE, KEY_IDX_NQPOS, KEY_IDX_NCTRL,
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_CONDIM,
)
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields

comptime DT = DType.float64

comptime APOLLO = String(
    "references/mujoco_menagerie-main/apptronik_apollo/scene.xml"
)

# MuJoCo 3.10.0, `mj_step` from keyframe 0, `qpos` after N steps. Read off the
# runtime — NOT produced by this engine.
comptime MJ_QPOS_2_X = 1.1608841282878448e-03
comptime MJ_QPOS_2_Y = 2.3835120651553819e-04
comptime MJ_QPOS_2_Z = 1.0190478384891335e+00

# ⚠ AN EXPLICIT `<pair>`, WITH GEOMS THAT DECLARE condim 1. A fixture whose
# GEOMS said 6 would pass against the old geom-only recorder and prove
# nothing: the pair is the only source of the 6.
comptime XML_PAIR = String(
    """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name='floor' type='plane' size='0 0 .05' condim='1'/>
    <body pos='0 0 .2'>
      <freejoint/>
      <geom name='b' type='box' size='.1 .1 .1' mass='1' condim='1'/>
    </body>
  </worldbody>
  <contact>
    <pair geom1='b' geom2='floor' condim='6' friction='1 1 .01 .001 .001'/>
  </contact>
</mujoco>"""
)
# ⚠ THE NEGATIVE CONTROL FIXTURE. Identical but for the pair's condim, so a
# recorder that simply returned 6 whenever a `<pair>` exists cannot pass both.
comptime XML_PAIR3 = String(
    """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name='floor' type='plane' size='0 0 .05' condim='1'/>
    <body pos='0 0 .2'>
      <freejoint/>
      <geom name='b' type='box' size='.1 .1 .1' mass='1' condim='1'/>
    </body>
  </worldbody>
  <contact>
    <pair geom1='b' geom2='floor' condim='3' friction='1 1 .01 .001 .001'/>
  </contact>
</mujoco>"""
)


def _max_condim_of(xml: String, base: String) raises -> Int:
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    return fmd.max_condim


def test_pair_condim_reaches_max_condim() raises:
    """`<contact><pair condim>` is a source of the model's condim requirement.

    ⚠ EXPECTED VALUES ARE MUJOCO'S: for the fixture, `m.pair_dim == [6]` and
    `contact.dim == 6` despite both geoms declaring `condim="1"`. For apollo,
    `m.pair_dim == [6 6 1 1 1 1]` and all four foot contacts report dim 6.
    """
    print("=== <pair condim> reaches fmd.max_condim ===")
    var got6 = _max_condim_of(XML_PAIR, String(""))
    var got3 = _max_condim_of(XML_PAIR3, String(""))
    print("  pair condim=6 -> ", got6, "   pair condim=3 -> ", got3)
    assert_true(
        got6 == 6,
        "a `<pair condim='6'>` between two condim-1 geoms must make the"
        " model's requirement 6 (MuJoCo reports contact.dim 6); got "
        + String(got6)
        + ". Computing this from the GEOMS alone misses it entirely — a"
        " pair's condim REPLACES what the mask path would compute.",
    )
    # ⚠ THE NEGATIVE CONTROL. Without it a recorder that returned 6 for any
    # model with a `<pair>` at all would pass the row above.
    assert_true(
        got3 == 3,
        "a `<pair condim='3'>` must leave the requirement at the floor of 3,"
        " got " + String(got3),
    )

    var src = read_model_source(APOLLO)
    var apollo_mc = _max_condim_of(src[0], src[1])
    print("  apptronik_apollo -> ", apollo_mc, " (MuJoCo pair_dim max: 6)")
    assert_true(
        apollo_mc == 6,
        "apollo's soles are `<pair condim='6'>` over condim-1 geoms; the"
        " recorded requirement must be 6, got " + String(apollo_mc),
    )
    print("  PASS")


def test_studio_bound_covers_the_domain() raises:
    """The studio's bound, and the check that says so when it stops covering.

    ⚠ THE WARNING SHOULD BE UNREACHABLE, which is the point — it is the
    assertion that `STUDIO_MAX_CONDIM` still covers MuJoCo's condim domain
    {1, 3, 4, 6}, sitting where the user would be told if it ever did not.
    """
    print("=== the studio's MAX_CONDIM covers the models it opens ===")
    print("  STUDIO_MAX_CONDIM =", STUDIO_MAX_CONDIM)
    assert_true(
        STUDIO_MAX_CONDIM >= 6,
        "MuJoCo's condim domain tops out at 6; a studio bound of "
        + String(STUDIO_MAX_CONDIM)
        + " drops rows on 46 of this tree's 142 models, SILENTLY —"
        " `_contact_solve_env` clamps without reporting.",
    )
    var src = read_model_source(APOLLO)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var note = studio_condim_warning(fmd)
    print("  apollo warning: '" + note + "'")
    assert_true(
        note.byte_length() == 0,
        "apollo needs condim " + String(fmd.max_condim) + " and the studio"
        " builds " + String(STUDIO_MAX_CONDIM) + " — it should not warn:"
        " " + note,
    )
    print("  PASS")


def test_apollo_matches_mujoco_with_condim_six() raises:
    """The trajectory, which is the only thing any of the above is for.

    ⚠ THE MODEL IS STEPPED FROM ITS `stand` KEYFRAME, INCLUDING ITS `ctrl`.
    From `qpos0` apollo is a folded humanoid with its actuators commanded to
    zero, which measures a pose the model never asks for; and driving the
    keyframe's `qpos` without its `ctrl` leaves the servos pulling towards
    zero, which is a third trajectory again. Both mistakes have been made in
    this tree.
    """
    print("=== apollo vs MuJoCo, 2 steps from the stand keyframe ===")
    var src = read_model_source(APOLLO)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    assert_true(
        dims.get_nkey() > 0,
        "apollo ships a `stand` keyframe and this gate needs it — nkey is 0",
    )

    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    for i in range(nq):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)

    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var nct = Int(Float64(sf.key_meta.data[KEY_IDX_NCTRL]))
    for a in range(min(nct, nact)):
        actions[a] = Float64(sf.key_ctrl.data[a])
    assert_true(
        nct == nact and nact == 32,
        "the keyframe must drive all 32 actuators — got nctrl "
        + String(nct) + " over nact " + String(nact),
    )

    var integ = StudioImpFastPyr(dims)
    for _ in range(2):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)

    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    var n6 = 0
    for k in range(nc):
        if Int(Float64(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_CONDIM])) == 6:
            n6 += 1
    print("  ncon", nc, " of which condim 6:", n6, " (MuJoCo: 6 of 6)")
    # ⚠ VACUITY. Without live condim-6 contacts this file compares a
    # trajectory in which nothing this fix touches ever happened.
    assert_true(
        nc >= 4 and n6 >= 4,
        "the feet must actually be in contact at condim 6 or the gate is"
        " vacuous — ncon " + String(nc) + ", condim-6 " + String(n6),
    )

    var want: List[Float64] = [
        MJ_QPOS_2_X, MJ_QPOS_2_Y, MJ_QPOS_2_Z
    ]
    var worst = 0.0
    for i in range(3):
        var got = Float64(d.qpos.data[i])
        var e = abs(got - want[i])
        if e > worst:
            worst = e
        print("  qpos[", i, "] ours", got, " mj", want[i], " d", e)
    assert_true(
        worst < 1e-12,
        "apollo's root must track MuJoCo to machine precision after 2 steps;"
        " worst |d| = " + String(worst)
        + ". Before the condim chain was closed this was 1.9e-03 at ONE step,"
        " and the whole of it was contact rows that were never built —"
        " editing the soles down to condim 3 in the shared XML collapses it"
        " to 5.6e-17.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
