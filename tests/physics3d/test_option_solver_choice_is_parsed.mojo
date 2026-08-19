"""`<option cone/solver/integrator>` — which solver the MODEL asks for.

    pixi run mojo run -I . tests/physics3d/test_option_solver_choice_is_parsed.mojo

WHAT WAS MISSING. Neither parser read any of the three. The consequences split
by path, and only one of them was survivable:

  * COMPTIME. `cone_type` is a `ModelDefFromXML` parameter, so every model def
    states its own cone as a hand-written literal. 161 instantiations were
    audited and all agree with their XML — but nothing checked that, and a
    typo would have been a silently wrong friction cone.
  * RUNTIME. There was no parsed value at all, so a tool could not agree with
    the file even in principle. The studio opens ANY MJCF and builds
    `EulerIntegrator[..., CONE_TYPE=ELLIPTIC]` with `SOLVER` defaulting to
    `"pgs"` — for every model, including the menagerie's pyramidal ones and
    spot, which asks for `implicitfast`.

⚠⚠ THE DEFAULTS ARE NOT THE ONES THIS TREE HABITUALLY BUILDS. MuJoCo's default
cone is PYRAMIDAL and its default solver is NEWTON (`m.opt.solver == 2` on a
model whose `<option>` says nothing). Ours were ELLIPTIC and `"pgs"`. The 21
def files that say ELLIPTIC say it because their XML does, not because it is
the default — reading the habit as the default is how the runtime path ended up
applying it to everything.

⚠ NOTHING DISPATCHES ON THESE YET, and that is deliberate rather than an
oversight: the cone and the solver are COMPTIME parameters of the integrator,
so honouring them means instantiating several variants and choosing at runtime
— a change to the tools, not to the parser. Recording the values is what makes
that possible at all, and what lets a tool compare what it BUILT against what
the file WANTED instead of having no way to know. This gate exists so the
values cannot rot in the meantime.

⚠ THE VALUES ARE ASSERTED AGAINST MUJOCO'S OWN `m.opt.*`, measured on the
3.10.0 runtime, not against our own reading of the spec:

    humanoid.xml : cone=0 solver=0 integrator=1 impratio=1
    spot scene   : cone=1 solver=2 integrator=3 impratio=100

`humanoid.xml` is the useful in-tree fixture precisely because it sets
`integrator="RK4" solver="PGS"` — two NON-default values, and `solver="PGS"`
is the one our runtime happened to default to, so a parser that ignored the
attribute would still produce the right number for the wrong reason. `cone` and
`integrator` are what make that row load-bearing.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.types import ConeType, SolverType, IntegratorType
from mojo_rl.physics3d.gpu.constants import (
    MODEL_META_IDX_CONE,
    MODEL_META_IDX_SOLVER,
    MODEL_META_IDX_INTEGRATOR,
)

comptime DT = DType.float64

comptime XML_BARE = String("<mujoco><worldbody/></mujoco>")
comptime XML_SET = String(
    "<mujoco><option cone='elliptic' solver='PGS' integrator='implicitfast'/>"
    "<worldbody/></mujoco>"
)
comptime XML_CG = String(
    "<mujoco><option cone='pyramidal' solver='CG' integrator='RK4'/>"
    "<worldbody/></mujoco>"
)
# ⚠ A VALUE MuJoCo WOULD REJECT. Its compiler refuses the file; we cannot,
# because this runs inside a comptime dimension counter, so the contract is
# "fall back to the default" and that has to be pinned.
comptime XML_JUNK = String(
    "<mujoco><option cone='banana' solver='banana' integrator='banana'/>"
    "<worldbody/></mujoco>"
)

comptime PM_BARE = parse_xml(XML_BARE)
comptime PM_SET = parse_xml(XML_SET)
comptime PM_CG = parse_xml(XML_CG)
comptime PM_JUNK = parse_xml(XML_JUNK)

comptime HUMANOID = String("mojo_rl/envs/humanoid/assets/humanoid.xml")


def test_comptime_parser_reads_the_three() raises:
    """`parse_xml` — the COMPTIME counter every model def is built from."""
    print("=== parse_xml reads cone / solver / integrator ===")
    print("  bare  ", PM_BARE.CONE, PM_BARE.SOLVER, PM_BARE.INTEGRATOR)
    print("  set   ", PM_SET.CONE, PM_SET.SOLVER, PM_SET.INTEGRATOR)
    print("  cg    ", PM_CG.CONE, PM_CG.SOLVER, PM_CG.INTEGRATOR)
    print("  junk  ", PM_JUNK.CONE, PM_JUNK.SOLVER, PM_JUNK.INTEGRATOR)

    # ⚠ MuJoCo's defaults, NOT ours. PYRAMIDAL and NEWTON.
    assert_true(
        PM_BARE.CONE == ConeType.PYRAMIDAL
        and PM_BARE.SOLVER == SolverType.NEWTON
        and PM_BARE.INTEGRATOR == IntegratorType.EULER,
        "a model setting none of the three must get MuJoCo's defaults"
        " (PYRAMIDAL, NEWTON, EULER), got "
        + String(PM_BARE.CONE) + " / " + String(PM_BARE.SOLVER) + " / "
        + String(PM_BARE.INTEGRATOR),
    )
    assert_true(
        PM_SET.CONE == ConeType.ELLIPTIC
        and PM_SET.SOLVER == SolverType.PGS
        and PM_SET.INTEGRATOR == IntegratorType.IMPLICITFAST,
        "`cone='elliptic' solver='PGS' integrator='implicitfast'` was not read",
    )
    # ⚠ CASE. MuJoCo writes "PGS" / "CG" / "Newton" / "RK4" capitalised and
    # matches them case-insensitively; a byte-exact compare against a lowercase
    # literal would miss every real model.
    assert_true(
        PM_CG.SOLVER == SolverType.CG
        and PM_CG.INTEGRATOR == IntegratorType.RK4,
        "capitalised `solver='CG' integrator='RK4'` was not matched — the"
        " compare must be case-insensitive",
    )
    assert_true(
        PM_JUNK.CONE == ConeType.PYRAMIDAL
        and PM_JUNK.SOLVER == SolverType.NEWTON
        and PM_JUNK.INTEGRATOR == IntegratorType.EULER,
        "an unrecognised value must fall back to MuJoCo's default, not guess",
    )
    print("  PASS")


def test_runtime_parser_reaches_model_meta() raises:
    """The RUNTIME path — the studio's — records them in model META.

    ⚠ A DIFFERENT PARSER FROM THE ONE ABOVE. `parse_xml` only counts;
    `build_model_runtime` goes through `parse_xml_full`, and a fix in one has
    repeatedly not been a fix in the other
    (`feedback_physics3d_two_parser_paths`).

    ⚠ THE EXPECTED VALUES ARE MUJOCO'S OWN, read off `m.opt.*` on the 3.10.0
    runtime for this exact file: cone=0 (pyramidal), solver=0 (PGS),
    integrator=1 (RK4).
    """
    print("=== the runtime parser records them in META ===")
    var src = read_model_source(HUMANOID)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)

    var cone = Int(Float64(m.meta.data[MODEL_META_IDX_CONE]))
    var solver = Int(Float64(m.meta.data[MODEL_META_IDX_SOLVER]))
    var integ = Int(Float64(m.meta.data[MODEL_META_IDX_INTEGRATOR]))
    print("  humanoid.xml -> META cone", cone, " solver", solver,
          " integrator", integ, " (MuJoCo: 0 / 0 / 1)")

    assert_true(
        cone == ConeType.PYRAMIDAL,
        "humanoid.xml sets no `cone`, so it is PYRAMIDAL — META says "
        + String(cone),
    )
    assert_true(
        solver == SolverType.PGS,
        "humanoid.xml sets `solver='PGS'` — META says " + String(solver),
    )
    # ⚠ THE ROW THAT CARRIES THE TEST. `solver='PGS'` is also what our
    # integrator defaulted to, so it alone cannot distinguish "parsed" from
    # "ignored". `integrator='RK4'` can: nothing defaults to RK4.
    assert_true(
        integ == IntegratorType.RK4,
        "humanoid.xml sets `integrator='RK4'` — META says "
        + String(integ)
        + ". Nothing defaults to RK4, so this is the row that proves the"
        " attribute was actually read rather than coincidentally matched.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
