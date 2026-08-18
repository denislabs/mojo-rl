"""What does the RUNTIME leg cost per step, against its comptime twin?

    pixi run mojo build -I . -o /tmp/b benchmarks/bench_runtime_step.mojo && /tmp/b

⚠ BUILD ONCE AND RUN THE BINARY — `mojo run` recompiles.

Companion to `bench_runtime_load.mojo`, which measures the one-off parse and
build. This measures the LOOP, which is what a viewer pays 60 times a second
and what `docs/PHYSICS3D_STUDIO_PLAN.md` §5 asks S0 to report.

The two legs run the SAME `step` body — 3a made it dimension-agnostic — and
differ in exactly two ways:

* strides come from a `RuntimeLayout` instead of a folded constant, so LLVM
  cannot unroll or vectorise the inner loops the same way;
* `Scratch` picks the HEAP on a dynamic provider, because `CAP_*` is 0 there.
  §10.7 measured a fixed-cap `InlineArray` under a runtime bound at
  1.13-1.18x WORSE than the `List`, so this is the faster of the two, not a
  compromise.

⚠ INTERLEAVED, AND THE MINIMUM IS REPORTED. Identical code has spanned
1.4-1.7x across a session on this machine; running all of leg A and then all
of leg B attributes that drift to the leg. Each round runs A then B, and the
answer is the best round each leg managed.

⚠ BOTH LEGS ARE STEPPED FROM ONE SEEDED STATE and their final qpos is
compared, so a leg that got fast by doing less work is visible. A benchmark
whose two arms silently diverge is measuring two different programs.
"""

from std.time import perf_counter_ns

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
comptime STEPS = 400
comptime ROUNDS = 5
comptime WARMUP = 50


def _fmt(v: Float64, places: Int = 2) -> String:
    var mul = 1.0
    for _ in range(places):
        mul *= 10.0
    var scaled = Int(v * mul + (0.5 if v >= 0 else -0.5))
    var whole = scaled // Int(mul)
    var frac = scaled % Int(mul)
    if frac < 0:
        frac = -frac
    var f = String(frac)
    while f.byte_length() < places:
        f = "0" + f
    return String(whole) + "." + f


def bench[MODEL: ModelDefLike](ctx: DeviceContext, path: String, name: String) raises:
    comptime MD = ModelDims[MODEL, 0]

    var ms = Model[DT, MD]()
    MODEL.init_fields[DT](ctx, ms)
    var ds = Data[DT, MD, 1]()
    var integ_s = EulerIntegrator[DT, MD, BATCH=1, MAX_CONDIM=3]()

    var fmd = parse_model_runtime(path)
    var dims = dims_from_flat(fmd, max_contacts=MD.MAX_CONTACTS)
    var mr = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, mr)
    var sf = spec_fields_runtime[DT](fmd, dims)
    var dr = Data[DT, DynDims, 1](dims)
    var integ_r = EulerIntegrator[DT, DynDims, BATCH=1, MAX_CONDIM=3](dims)

    var nq = dims.get_nq()
    var nv = dims.get_nv()

    var best_s = Float64(1e30)
    var best_r = Float64(1e30)
    for _ in range(ROUNDS):
        # ⚠ BOTH ARMS RESEEDED EVERY ROUND, from the same state. Letting them
        # run on from the previous round would put round 5 in a different pose
        # from round 1 — a different contact count, hence a different amount of
        # work — and the "minimum" would then be the round that happened to be
        # airborne.
        for i in range(nq):
            var v = Scalar[DT](sf.qpos0.data[i]) + Scalar[DT](0.01 * Float64(i % 5))
            ds.qpos.data[i] = v
            dr.qpos.data[i] = v
        for i in range(nv):
            var v = Scalar[DT](0.03 * Float64(i % 7) - 0.05)
            ds.qvel.data[i] = v
            dr.qvel.data[i] = v
        for _ in range(WARMUP):
            integ_s.step["cpu"](ds, ms)
            integ_r.step["cpu"](dr, mr)

        var a0 = perf_counter_ns()
        for _ in range(STEPS):
            integ_s.step["cpu"](ds, ms)
        var a1 = perf_counter_ns()
        for _ in range(STEPS):
            integ_r.step["cpu"](dr, mr)
        var a2 = perf_counter_ns()

        var us_s = Float64(a1 - a0) / 1000.0 / Float64(STEPS)
        var us_r = Float64(a2 - a1) / 1000.0 / Float64(STEPS)
        if us_s < best_s:
            best_s = us_s
        if us_r < best_r:
            best_r = us_r

    # ⚠ NON-VACUITY: the two arms must still be simulating the same thing.
    var worst = Float64(0)
    for i in range(nq):
        var e = abs(Float64(ds.qpos.data[i]) - Float64(dr.qpos.data[i]))
        if e > worst:
            worst = e

    print(
        "  ", name,
        " comptime ", _fmt(best_s), " us/step   runtime ", _fmt(best_r),
        " us/step   ratio ", _fmt(best_r / best_s, 3),
        "   (legs agree to ", worst, ")",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=== per-step cost, comptime vs runtime dims (CPU, BATCH=1) ===")
    print("   ", STEPS, "steps x", ROUNDS, "rounds, interleaved, min reported")
    bench[Walker2dModel](ctx, "mojo_rl/envs/walker2d/assets/walker2d.xml", "walker2d")
    bench[HopperModel](ctx, "mojo_rl/envs/hopper/assets/hopper.xml", "hopper  ")
    bench[HumanoidModel](ctx, "mojo_rl/envs/humanoid/assets/humanoid.xml", "humanoid")
