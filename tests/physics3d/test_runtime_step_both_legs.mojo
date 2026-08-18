"""The same model, STEPPED on both legs — does the runtime leg simulate?

WHY THIS EXISTS
===============
`test_runtime_model_load` proves a runtime-loaded `Model` is byte-identical to
the comptime one, and `test_dispatchers_both_legs` proves seven dispatchers
agree. Neither runs the LOOP. Between them sits everything the solve does, and
that is where the runtime leg was broken:

* `ContactScratch` allocated `BATCH * SOLVER_WS` from the COMPTIME member. On
  a dynamic provider that is `81*1 + 12*1*(-1) = 69` scalars for every model,
  while `contact_solve` indexes it at offsets derived from the runtime nv/mc
  (walker2d at mc=64 needs 12096). Heap overflow on the first solve.
* Five `rl2(BATCH, SOLVER_WS)` views had the same comptime value — the 3a
  sweep converted the SPELLING to `rl2`/`lt_dyn` and left the VALUE, which is
  worse than missing them, because it reads as converted.

⚠ IT CRASHED AT `free`, NOT AT THE WRITE, so the stack named tcmalloc and the
failure read as an allocator bug. A model with no tendons (walker2d) ran 2000
steps and printed plausible numbers ON CORRUPTED MEMORY.

⚠⚠ WHAT THIS TEST IS ALSO FOR: the runtime leg still SILENTLY SKIPS whole
constraint families, because every `CAP_*` on `DynDims` is 0 and the gates are
`comptime if D.CAP_NTENDON > 0` / `CAP_NEQUALITY` / `CAP_NMESH_VERTS`, plus
`comptime if D.NSITE > 0` in `forward_kinematics` (NSITE is POISON, so that is
false too). A model with none of those features must agree BIT-EXACTLY; a
model with them will diverge, and the divergence is the measure of what is
missing. Both are asserted below, so the day a family is converted this test
tells you by failing.

Run: pixi run mojo run -I . tests/physics3d/test_runtime_step_both_legs.mojo
"""

from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.parser import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.parser.runtime_load import spec_fields_runtime
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel

comptime DT = DType.float64
comptime STEPS = 200
comptime MC = 20


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def both_legs[
    MODEL: ModelDefLike
](
    mut t: Tally, ctx: DeviceContext, path: String, name: String,
    expect_exact: Bool,
) raises:
    """Step both legs from one seeded state; compare qpos/qvel."""
    comptime MD = ModelDims[MODEL, 0]

    # ── static leg ────────────────────────────────────────────────────────
    var ms = Model[DT, MD]()
    MODEL.init_fields[DT](ctx, ms)
    var ds = Data[DT, MD, 1]()
    var integ_s = EulerIntegrator[DT, MD, BATCH=1, MAX_CONDIM=3]()

    # ── runtime leg ───────────────────────────────────────────────────────
    var fmd = parse_model_runtime(path)
    var dims = dims_from_flat(fmd, max_contacts=MD.MAX_CONTACTS)
    var mr = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, mr)
    var sf = spec_fields_runtime[DT](fmd, dims)
    var dr = Data[DT, DynDims, 1](dims)
    var integ_r = EulerIntegrator[DT, DynDims, BATCH=1, MAX_CONDIM=3](dims)

    # ⚠ ONE seeded state, written into BOTH — not each leg's own reset, which
    # would make a divergence in the reset read as a divergence in the step.
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    t.truth(nq == MD.NQ and nv == MD.NV, String(name, ": dims agree (nq ", nq, " nv ", nv, ")"))
    for i in range(nq):
        var v = Scalar[DT](sf.qpos0.data[i]) + Scalar[DT](0.01 * Float64(i % 5))
        ds.qpos.data[i] = v
        dr.qpos.data[i] = v
    for i in range(nv):
        var v = Scalar[DT](0.03 * Float64(i % 7) - 0.05)
        ds.qvel.data[i] = v
        dr.qvel.data[i] = v

    # ⚠ ONE STEP FIRST, AND IT IS THE SHARP TEST. Stepping is deterministic,
    # so if step 1 agrees to the bit, every later step does too — inductively.
    # A difference after 200 steps therefore PROVES a difference in some single
    # step, and measuring step 1 separately says whether the drift is a real
    # per-step disagreement or amplification of one.
    integ_s.step["cpu"](ds, ms)
    integ_r.step["cpu"](dr, mr)
    var one_q = Float64(0)
    var one_v = Float64(0)
    for i in range(nq):
        var dq = abs(Float64(ds.qpos.data[i]) - Float64(dr.qpos.data[i]))
        if dq > one_q:
            one_q = dq
    for i in range(nv):
        var dv = abs(Float64(ds.qvel.data[i]) - Float64(dr.qvel.data[i]))
        if dv > one_v:
            one_v = dv
    print("   ", name, ": after 1 step   max|dqpos| =", one_q,
          " max|dqvel| =", one_v)

    for _ in range(STEPS - 1):
        integ_s.step["cpu"](ds, ms)
        integ_r.step["cpu"](dr, mr)

    var maxq = Float64(0)
    var maxv = Float64(0)
    var ndiff = 0
    for i in range(nq):
        var d0 = abs(Float64(ds.qpos.data[i]) - Float64(dr.qpos.data[i]))
        if d0 > maxq:
            maxq = d0
        if d0 != 0:
            ndiff += 1
    for i in range(nv):
        var d0 = abs(Float64(ds.qvel.data[i]) - Float64(dr.qvel.data[i]))
        if d0 > maxv:
            maxv = d0

    # ⚠ NON-VACUITY: two frozen states agree perfectly. Require the sim to
    # have MOVED before believing any agreement between the legs.
    var moved = Float64(0)
    for i in range(nq):
        moved += abs(
            Float64(ds.qpos.data[i])
            - (Float64(sf.qpos0.data[i]) + 0.01 * Float64(i % 5))
        )
    t.truth(moved > 1e-9, String(name, ": the static leg actually moved (", moved, ")"))

    print(
        "   ", name, ": after", STEPS, "steps  max|dqpos| =", maxq,
        " max|dqvel| =", maxv, " differing qpos slots:", ndiff, "/", nq,
    )
    if expect_exact:
        t.truth(
            one_q == 0.0 and one_v == 0.0,
            String(name, ": ONE step is BIT-EXACT across the legs"),
        )
        # ⚠ THE MULTI-STEP ARM IS A TOLERANCE, NOT AN EQUALITY. Even a
        # bit-exact step can be followed by drift only if something is
        # non-deterministic, so this arm is really a guard on the SIZE of any
        # single-step disagreement the arm above may one day permit.
        t.truth(
            maxq < 1e-12 and maxv < 1e-11,
            String(name, ": ", STEPS, " steps stay within tolerance"),
        )
    else:
        # Documented divergence — the caps disable constraint families on the
        # runtime leg. Asserting it is non-zero keeps this honest: when the
        # families are converted, THIS is what fails and tells you.
        t.truth(
            maxq != 0.0 or maxv != 0.0,
            String(
                name,
                ": diverges as expected (tendon/equality/site/mesh families are"
                " comptime-gated OFF on the runtime leg) — convert them and"
                " flip this arm to exact",
            ),
        )


def main() raises:
    var t = Tally()
    var ctx = DeviceContext()
    print("=== the same model, stepped on BOTH LEGS ===")
    print("--- models with no tendon/equality/site/mesh: must be EXACT ---")
    both_legs[Walker2dModel](
        t, ctx, "mojo_rl/envs/walker2d/assets/walker2d.xml", "walker2d", True
    )
    both_legs[HopperModel](
        t, ctx, "mojo_rl/envs/hopper/assets/hopper.xml", "hopper", True
    )
    print("--- a model WITH tendons + sites: divergence is the measurement ---")
    both_legs[HumanoidModel](
        t, ctx, "mojo_rl/envs/humanoid/assets/humanoid.xml", "humanoid", False
    )

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_runtime_step_both_legs: " + String(t.fails) + " failed")
