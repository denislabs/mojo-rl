"""TaskEmbedder — embed correctness, grad overfit, CPU↔GPU parity (Phase 3.4).

    pixi run            mojo run -I . tests/nn/test_dreamer4_task_embedder.mojo  # CPU
    pixi run -e apple   mojo run -I . tests/nn/test_dreamer4_task_embedder.mojo  # +GPU

Three checks:
  1. EMBED — out[b,t,a,:] == task_table[id_b] + agent_base, identical across all
     t and all NAGENT agent tokens (the broadcast).
  2. GRAD OVERFIT — fitting a (per-sequence-constant) target agent input by
     plain SGD on table+base drives the loss to ~0, proving the broadcast
     gradient reduction is correct.
  3. CPU↔GPU PARITY — embed output and both param grads match after one
     backward (identical params via seed() before each make).
"""

from std.memory import alloc
from std.math import sin, abs
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.deep_agents.dreamer4.task_embedder import TaskEmbedder


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _maxdiff(a: UnsafePointer[Scalar[DT], MutAnyOrigin],
             b: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


comptime D = 8
comptime NTASK = 4
comptime NAGENT = 1
comptime B = 3
comptime T = 2
comptime AG = NAGENT * D
comptime BT = B * T
comptime N = BT * AG
comptime TW = NTASK * D


def main() raises:
    print("=" * 70)
    print("TaskEmbedder — embed / grad / parity (Phase 3.4)")
    print("=" * 70)

    var ids = _alloc(B)
    ids[0] = 0.0
    ids[1] = 2.0
    ids[2] = 1.0

    # ── 1. CPU embed correctness ────────────────────────────────────────
    var te = TaskEmbedder[D, NTASK, NAGENT].make[target="cpu", INIT=Xavier]()
    var agent_in = _alloc(N)
    te.embed_into["cpu", B, T](ids, agent_in)

    var tab = te.task_table.value_unsafe_ptr_cpu()
    var base = te.agent_base.value_unsafe_ptr_cpu()
    var max_embed_err: Float64 = 0.0
    var max_bcast_err: Float64 = 0.0
    for b in range(B):
        var idb = Int(Float64(ids[b]) + 0.5)
        for t in range(T):
            var bt = b * T + t
            for a in range(NAGENT):
                for d in range(D):
                    var got = agent_in[bt * AG + a * D + d]
                    var want = tab[idb * D + d] + base[d]
                    var e = abs(Float64(got) - Float64(want))
                    if e > max_embed_err:
                        max_embed_err = e
                    # broadcast: must equal the t=0,a=0 value for this b
                    var ref0 = agent_in[(b * T) * AG + d]
                    var be = abs(Float64(got) - Float64(ref0))
                    if be > max_bcast_err:
                        max_bcast_err = be
    print("   embed max|err| =", max_embed_err, " (must be 0)")
    print("   broadcast max|err| =", max_bcast_err, " (must be 0)")
    assert_true(max_embed_err == 0.0, "embed = table[id]+base")
    assert_true(max_bcast_err == 0.0, "broadcast over (t,a)")

    # ── 2. CPU grad overfit (per-sequence-constant target) ──────────────
    var target = _alloc(N)
    for b in range(B):
        for t in range(T):
            var bt = b * T + t
            for a in range(NAGENT):
                for d in range(D):
                    # constant over (t,a) so the broadcast embedding can fit it
                    target[bt * AG + a * D + d] = Scalar[DT](
                        0.4 * sin(0.3 + 0.6 * Float64(b * D + d))
                    )
    var grad_in = _alloc(N)
    comptime LR = Scalar[DT](0.05)
    var first: Float64 = 0.0
    var last: Float64 = 0.0
    for step in range(300):
        te.zero_grad["cpu"]()
        te.embed_into["cpu", B, T](ids, agent_in)
        var loss: Float64 = 0.0
        for i in range(N):
            var diff = agent_in[i] - target[i]
            grad_in[i] = diff
            loss += 0.5 * Float64(diff) * Float64(diff)
        te.accumulate_grad["cpu", B, T](grad_in)
        # plain SGD step on both params
        var vp = te.task_table.value_unsafe_ptr_cpu()
        var gp = te.task_table.grad_unsafe_ptr_cpu()
        for i in range(TW):
            vp[i] = vp[i] - LR * gp[i]
        var bvp = te.agent_base.value_unsafe_ptr_cpu()
        var bgp = te.agent_base.grad_unsafe_ptr_cpu()
        for i in range(D):
            bvp[i] = bvp[i] - LR * bgp[i]
        if step == 0:
            first = loss
        last = loss
        if step % 60 == 0:
            print("   step", step, " loss =", loss)
    print("   first =", first, "  last =", last)
    assert_true(last < 1e-4 * first + 1e-8, "grad overfit must converge")

    # ── 3. CPU↔GPU parity ───────────────────────────────────────────────
    var ctx = DeviceContext()
    seed(99)
    var tc = TaskEmbedder[D, NTASK, NAGENT].make[target="cpu", INIT=Xavier]()
    seed(99)
    var tg = TaskEmbedder[D, NTASK, NAGENT].make[target="gpu", INIT=Xavier](ctx)

    var gin = _alloc(N)
    for i in range(N):
        gin[i] = Scalar[DT](0.2 * sin(1.1 + 0.5 * Float64(i)))

    # CPU embed + grad
    var out_c = _alloc(N)
    tc.zero_grad["cpu"]()
    tc.embed_into["cpu", B, T](ids, out_c)
    tc.accumulate_grad["cpu", B, T](gin)

    # GPU embed + grad (device buffers)
    var outd = ctx.enqueue_create_buffer[DT](N)
    var gind = ctx.enqueue_create_buffer[DT](N)
    var ginh = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        ginh.unsafe_ptr()[i] = gin[i]
    ctx.enqueue_copy(gind, ginh)
    ctx.synchronize()
    tg.zero_grad["gpu"]()
    tg.embed_into["gpu", B, T](
        ids, rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](outd.unsafe_ptr())
    )
    tg.accumulate_grad["gpu", B, T](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gind.unsafe_ptr())
    )
    ctx.synchronize()

    var outh = ctx.enqueue_create_host_buffer[DT](N)
    var gtabh = ctx.enqueue_create_host_buffer[DT](TW)
    var gbaseh = ctx.enqueue_create_host_buffer[DT](D)
    ctx.enqueue_copy(outh, outd)
    ctx.enqueue_copy(gtabh, tg.task_table.grd.dev.value())
    ctx.enqueue_copy(gbaseh, tg.agent_base.grd.dev.value())
    ctx.synchronize()

    var d_out = _maxdiff(
        out_c, rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](outh.unsafe_ptr()), N)
    var d_tab = _maxdiff(
        tc.task_table.grad_unsafe_ptr_cpu(),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gtabh.unsafe_ptr()), TW)
    var d_base = _maxdiff(
        tc.agent_base.grad_unsafe_ptr_cpu(),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gbaseh.unsafe_ptr()), D)
    print("   embed      CPU↔GPU max|Δ| =", d_out)
    print("   grad table CPU↔GPU max|Δ| =", d_tab)
    print("   grad base  CPU↔GPU max|Δ| =", d_base)
    assert_true(d_out < 2e-6, "embed parity")
    assert_true(d_tab < 2e-6, "task_table grad parity")
    assert_true(d_base < 2e-6, "agent_base grad parity")

    print("=" * 70)
    print("ALL PASSED — TaskEmbedder (Phase 3.4)")
    print("=" * 70)
