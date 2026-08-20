"""The same model, STEPPED on both legs — does the runtime leg simulate?

WHY THIS EXISTS
===============
`test_runtime_model_load` proves a runtime-loaded `Model` is byte-identical to
the comptime one, and `test_dispatchers_both_legs` proves seven dispatchers
agree. Neither runs the LOOP. Between them sits everything the solve does, and
that is where the runtime leg was broken — twice, in two different ways.

## Round 1: it could not step at all

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

## Round 2: it stepped, and SILENTLY SKIPPED WHOLE FAMILIES

Nineteen gates read `D.CAP_NTENDON > 0` / `CAP_NEQUALITY` / `CAP_NMESH_VERTS`
as "does this model have the feature". `CAP_*` answers a different question —
*which container*, exact on a static provider and **0 on a dynamic one** — so
on the runtime leg all nineteen were false and tendon rows, tendon limits,
equality rows and ALL mesh collision were compiled out. They are now
`may_exist[D.NTENDON]()` etc., which is true on a dynamic provider because
`DIM_POISON` is -1; see `fields/dims.mojo`.

⚠ THE MESH ARM EXISTS BECAUSE THE FIRST DRAFT OF THIS FILE DID NOT HAVE ONE.
`ModelDims`' SECOND PARAMETER is the mesh vertex budget, and passing 0 makes
the static leg raise "mesh vertex capacity exceeded" — which reads exactly
like "the comptime leg cannot do mesh collision without an env config". It
can; `Phyics3dEnv` merely happens to read the number off its config. so_arm100
is therefore a full both-legs arm, not a smoke test.

⚠ AND `detect_contacts_auto` PICKED A DIFFERENT ALGORITHM. `D.NGEOM` is
poison, so `-1 >= SAP_THRESHOLD` was false and a runtime model always took
the O(N^2) narrow phase while its comptime twin took SAP — two paths whose
contact ORDER and record conventions differ. That was most of the humanoid
number below, and 3c-b had explicitly filed it as benign.

WHAT THE NUMBERS WERE, so a regression is readable:

| model | qpos, 200 steps, BEFORE | AFTER |
|---|---|---|
| walker2d | 2.2e-15 | 2.2e-15 (never had a gated family) |
| humanoid (2 tendons, 18 geoms) | **1.5e-3** | **2.7e-15** |

A twelve-order-of-magnitude jump is what re-breaking a gate looks like.

## What the residual 1 ulp IS — MEASURED, not assumed

The legs are NOT bit-exact on walker2d and humanoid, and they are on hopper.
The difference first appears in `xpos`/`xquat` — FORWARD KINEMATICS, before
any contact exists (`meta[0]` is 0 on both legs at step 1) — and it is 1 ulp,
amplified to 2.5e-13 by the mass-matrix solve into `qacc`.

**It is the OPTIMIZER, and the control is decisive: built with
`--no-optimization`, every field of every model matches to the BIT.** The
static leg's strides are compile-time constants, so LLVM unrolls and contracts
the quaternion multiply-adds differently from the runtime-stride form. Nothing
in the physics differs. That is why the one-step arm below is a bounded ulp
count and not an equality — and the bound is what would catch a real
divergence, since a missing family costs 1e-3, not 1e-15.

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
from mojo_rl.envs.robots.so_arm100_xml import SoArm100Model, SO_ARM100_NMESH_VERTS
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel

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
    MODEL: ModelDefLike, MESH_VERTS: Int = 0
](
    mut t: Tally, ctx: DeviceContext, path: String, name: String,
    nmesh_verts: Int = 0,
) raises:
    """Step both legs from one seeded state; compare qpos/qvel.

    ⚠ `nmesh_verts` IS NOT IN THE FILE and cannot be derived before the
    meshes load — `dims_from_flat`'s docstring says so, and the builder raises
    with the number it needs. Pass the comptime def's own budget.
    """
    # ⚠ THE SECOND PARAMETER IS THE MESH VERTEX BUDGET, and passing 0 is how
    # this test spent its first draft believing the comptime leg could not do
    # mesh collision at all. `NMESH_VERTS` is not on `ModelDefLike` — whether
    # a model's meshes are COLLIDABLE is a decision, not a property of the
    # MJCF, so `ModelDims` takes it here and `Phyics3dEnv` happens to read it
    # off its config. A caller with no config passes it directly, exactly as
    # the runtime leg passes it to `dims_from_flat`.
    comptime MD = ModelDims[MODEL, MESH_VERTS]

    # ── static leg ────────────────────────────────────────────────────────
    var ms = Model[DT, MD]()
    MODEL.init_fields[DT](ctx, ms)
    var ds = Data[DT, MD, 1]()
    var integ_s = EulerIntegrator[DT, MD, BATCH=1, MAX_CONDIM=3]()

    # ── runtime leg ───────────────────────────────────────────────────────
    var fmd = parse_model_runtime(path)
    var dims = dims_from_flat(
        fmd, max_contacts=MD.MAX_CONTACTS, nmesh_verts=nmesh_verts
    )
    var mr = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, mr)
    var sf = spec_fields_runtime[DT](fmd, dims, mr)
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
    # ⚠⚠ A BOUND, NOT AN EQUALITY, AND THE HEADER SAYS WHY: at `-O0` these
    # ARE bit-exact, so the residual is LLVM contracting the FK quaternion
    # multiply-adds differently under folded strides. The bound is set two
    # orders above the observed 1-ulp noise and TEN orders below the 1.5e-3 a
    # single skipped constraint family costs, so it separates the two cases
    # without pretending the noise is not there.
    t.truth(
        one_q < 1e-14 and one_v < 1e-13,
        String(
            name, ": ONE step agrees to FP noise (", one_q, " / ", one_v, ")"
        ),
    )
    t.truth(
        maxq < 1e-12 and maxv < 1e-11,
        String(name, ": ", STEPS, " steps stay within tolerance"),
    )


def main() raises:
    var t = Tally()
    var ctx = DeviceContext()
    print("=== the same model, stepped on BOTH LEGS ===")
    print("--- no gated family, no SAP: the control ---")
    both_legs[Walker2dModel](
        t, ctx, "mojo_rl/envs/walker2d/assets/walker2d.xml", "walker2d"
    )
    both_legs[HopperModel](
        t, ctx, "mojo_rl/envs/hopper/assets/hopper.xml", "hopper"
    )
    # ⚠ THESE TWO ARE THE POINT OF THE TEST, and neither was covered before
    # the `may_exist` conversion: humanoid exercises the TENDON rows and, at
    # ngeom 18, the SAP dispatch; so_arm100 exercises MESH COLLISION (ten
    # collidable meshes) and, at ngeom 33, SAP again. Without them the file
    # gated only the families that were never broken.
    print("--- 2 tendons + ngeom 18 (SAP): the tendon + broadphase arms ---")
    both_legs[HumanoidModel](
        t, ctx, "mojo_rl/envs/humanoid/assets/humanoid.xml", "humanoid"
    )
    # ⚠⚠ THE EQUALITY ARM, AND IT EXISTS BECAUSE THE RUNTIME LEG CRASHED
    # HERE. `may_exist` opened `build_weld_equality_rows` to dynamic
    # providers and it turned out to hold three `InlineArray[…, V_SIZE]` —
    # length 0 on a dynamic provider — that had never been swept, because the
    # ONLY caller sat behind `comptime if D.CAP_NEQUALITY > 0` and was
    # unreachable. Opening a gate can expose a latent zero-size container
    # behind it, and no audit of REACHABLE code can see one.
    print("--- a mocap WELD + meshes: the equality arm ---")
    both_legs[SawyerReachModel, 2048](
        t, ctx, "mojo_rl/envs/metaworld/assets/sawyer_reach.xml", "sawyer",
        nmesh_verts=2048,
    )
    print("--- 10 collidable meshes + ngeom 33 (SAP): the mesh arm ---")
    both_legs[SoArm100Model, SO_ARM100_NMESH_VERTS](
        t, ctx, "mojo_rl/envs/robots/assets/so_arm100.xml", "so_arm100",
        nmesh_verts=SO_ARM100_NMESH_VERTS,
    )

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_runtime_step_both_legs: " + String(t.fails) + " failed")
