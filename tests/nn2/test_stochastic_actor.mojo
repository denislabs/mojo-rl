"""StochasticActor tests — Phase 5.7.

Covers:
  - forward shape: input (BATCH × OBS) → output (BATCH × 2*ACT)
  - for_each_param: trunk Linear params + 2 head Linear params with
    correct prefixes
  - backward gradcheck (FD vs analytical) on a small actor
  - GPU parity vs CPU
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor


struct ParamRecord(Movable & ImplicitlyDestructible):
    var name: String
    var n_elems: Int
    var apply_decay: Bool

    def __init__(out self, name: String, n_elems: Int, apply_decay: Bool):
        self.name = name
        self.n_elems = n_elems
        self.apply_decay = apply_decay


struct RecordVisitor(ParamVisitor):
    var records: List[ParamRecord]

    def __init__(out self):
        self.records = List[ParamRecord]()

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.records.append(ParamRecord(name, n_elems, apply_decay))


def test_forward_shape_cpu() raises:
    """obs (BATCH × OBS) → output (BATCH × 2*ACT). Should run without
    error and produce finite values."""
    comptime OBS = 8
    comptime ACT = 3
    comptime BATCH = 4

    var actor = StochasticActor[
        OBS, ACT,
        Linear[OBS, 16], Tanh[16], Linear[16, 16], Tanh[16],
    ].make["cpu", INIT=Kaiming]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    for k in range(BATCH * OBS):
        in_buf[k] = Scalar[DT](Float32(k) * 0.1 - 0.5)
    for k in range(BATCH * 2 * ACT):
        out_buf[k] = -999.0

    var input  = TileTensor(in_buf,  row_major[BATCH, OBS]())
    var output = TileTensor(out_buf, row_major[BATCH, 2 * ACT]())
    actor.forward["cpu", BATCH](input, output)

    # All outputs should have been written (no -999 sentinels remaining).
    for b in range(BATCH):
        for j in range(2 * ACT):
            assert_true(output[b, j] > -990.0,
                "output[" + String(b) + "," + String(j) + "] not written")

    # Sanity: the two slices (mu, log_std) should generally differ
    # — they come from different heads with independent Kaiming-init
    # weights. (Probabilistic; could spuriously match — but with
    # 4×3=12 entries the odds of all-equal are infinitesimal.)
    var any_diff = False
    for b in range(BATCH):
        for j in range(ACT):
            if fabs(output[b, j] - output[b, ACT + j]) > Scalar[DT](1e-6):
                any_diff = True
                break
    assert_true(any_diff, "mu and log_std heads produced identical output — "
        "Parallel split broken?")

    in_buf.free()
    out_buf.free()
    print("  test_forward_shape_cpu PASSED")


def test_for_each_param_cpu() raises:
    """Trunk = Linear[OBS, 8] + Tanh[8] + Linear[8, 8] gives 4 trunk
    visits; heads add 4 more (a.weight, a.bias, b.weight, b.bias).
    Total = 8."""
    comptime OBS = 4
    comptime ACT = 2
    var actor = StochasticActor[
        OBS, ACT,
        Linear[OBS, 8], Tanh[8], Linear[8, 8],
    ].make["cpu", INIT=Zero]()

    var v = RecordVisitor()
    actor.for_each_param["cpu"](String("actor0"), v)

    # 2 Linear in trunk → 4 visits (weight+bias each). Tanh has none.
    # Plus heads Parallel → 4 visits (2 heads × 2 params each).
    assert_equal(len(v.records), 8)

    # Trunk leaves: actor0.trunk.0 (Linear), actor0.trunk.1 (Tanh, no params),
    # actor0.trunk.2 (Linear).
    assert_equal(v.records[0].name, String("actor0.trunk.0.weight"))
    assert_equal(v.records[0].n_elems, OBS * 8)
    assert_true(v.records[0].apply_decay)
    assert_equal(v.records[1].name, String("actor0.trunk.0.bias"))
    assert_true(not v.records[1].apply_decay)
    assert_equal(v.records[2].name, String("actor0.trunk.2.weight"))
    assert_equal(v.records[2].n_elems, 8 * 8)
    assert_equal(v.records[3].name, String("actor0.trunk.2.bias"))

    # Head leaves: actor0.heads.a.weight / .bias, actor0.heads.b.weight / .bias.
    assert_equal(v.records[4].name, String("actor0.heads.a.weight"))
    assert_equal(v.records[4].n_elems, 8 * ACT)
    assert_equal(v.records[5].name, String("actor0.heads.a.bias"))
    assert_equal(v.records[5].n_elems, ACT)
    assert_equal(v.records[6].name, String("actor0.heads.b.weight"))
    assert_equal(v.records[7].name, String("actor0.heads.b.bias"))

    print("  test_for_each_param_cpu PASSED (8 params, trunk + heads)")


def test_backward_gradcheck_cpu() raises:
    """FD gradcheck on a small actor."""
    comptime OBS = 3
    comptime ACT = 2
    comptime BATCH = 2
    comptime EPS_FD: Scalar[DT] = 5e-3
    comptime TOL_REL: Scalar[DT] = 3e-2     # gradcheck through 2 Linears + Tanh + Parallel

    var actor = StochasticActor[
        OBS, ACT,
        Linear[OBS, 4], Tanh[4],
    ].make["cpu", INIT=Kaiming]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)

    for k in range(BATCH * OBS):
        in_buf[k] = Scalar[DT](0.3 + Float32(k) * 0.1)
    for k in range(BATCH * 2 * ACT):
        go_buf[k] = Scalar[DT](0.1 + Float32(k) * 0.05)
    for k in range(BATCH * OBS):
        gi_buf[k] = -999.0

    var input    = TileTensor(in_buf,  row_major[BATCH, OBS]())
    var output   = TileTensor(out_buf, row_major[BATCH, 2 * ACT]())
    var grad_out = TileTensor(go_buf,  row_major[BATCH, 2 * ACT]())
    var grad_in  = TileTensor(gi_buf,  row_major[BATCH, OBS]())

    actor.forward["cpu", BATCH](input, output)
    actor.backward["cpu", BATCH](grad_out, grad_in)

    var max_rel: Scalar[DT] = 0.0
    for bi in range(BATCH):
        for di in range(OBS):
            in_buf[bi * OBS + di] += EPS_FD
            actor.forward["cpu", BATCH](input, output)
            var L_plus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for j2 in range(2 * ACT):
                    L_plus += go_buf[b2 * 2 * ACT + j2] * output[b2, j2]
            in_buf[bi * OBS + di] -= Scalar[DT](2.0) * EPS_FD
            actor.forward["cpu", BATCH](input, output)
            var L_minus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for j2 in range(2 * ACT):
                    L_minus += go_buf[b2 * 2 * ACT + j2] * output[b2, j2]
            in_buf[bi * OBS + di] += EPS_FD
            var fd = (L_plus - L_minus) / (Scalar[DT](2.0) * EPS_FD)
            var an = grad_in[bi, di]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel: max_rel = rel

    print("StochasticActor gradcheck max-rel-err = " + String(max_rel))
    assert_true(max_rel < TOL_REL,
        "gradcheck failed: " + String(max_rel))

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_backward_gradcheck_cpu PASSED")


def test_gpu_parity() raises:
    """Forward parity vs CPU. Same weights → same outputs."""
    comptime OBS = 6
    comptime ACT = 2
    comptime BATCH = 3
    comptime TOL: Scalar[DT] = 1e-5

    var ctx = DeviceContext()
    # Build CPU + GPU actors with Zero init — bit-identical weights.
    var actor_cpu = StochasticActor[
        OBS, ACT, Linear[OBS, 8], Tanh[8],
    ].make["cpu", INIT=Zero]()
    var actor_gpu = StochasticActor[
        OBS, ACT, Linear[OBS, 8], Tanh[8],
    ].make["gpu", INIT=Zero](ctx)

    # Copy CPU weights into GPU (since Zero init produced identical
    # zeros, but Zero-init produces 0 weight → trivial; let's
    # explicitly set distinguishable weights so the test is meaningful).
    # Set CPU weights to small non-zero values + mirror to GPU.
    # First trunk Linear (Linear[OBS, 8]).
    var trunk_lin_cpu_ref = Pointer(to=actor_cpu.trunk.children[0])
    var head_a_lin_cpu_ref = Pointer(to=actor_cpu.heads.branch_a)
    var head_b_lin_cpu_ref = Pointer(to=actor_cpu.heads.branch_b)

    # Helper inline: set CPU weights from a deterministic pattern.
    for i in range(OBS):
        for j in range(8):
            trunk_lin_cpu_ref[].weight[i * 8 + j] = Scalar[DT](
                Float32(i + j) * 0.05 - 0.1
            )
    for j in range(8):
        trunk_lin_cpu_ref[].bias[j] = Scalar[DT](0.01 * Float32(j))
    for i in range(8):
        for j in range(ACT):
            head_a_lin_cpu_ref[].weight[i * ACT + j] = Scalar[DT](
                Float32(i + j) * 0.07 + 0.02
            )
    for j in range(ACT):
        head_a_lin_cpu_ref[].bias[j] = Scalar[DT](0.05 * Float32(j))
    for i in range(8):
        for j in range(ACT):
            head_b_lin_cpu_ref[].weight[i * ACT + j] = Scalar[DT](
                Float32(i + j) * 0.03 - 0.05
            )
    for j in range(ACT):
        head_b_lin_cpu_ref[].bias[j] = Scalar[DT](-0.02 * Float32(j))

    # Upload to GPU.
    var w_t_host = ctx.enqueue_create_host_buffer[DT](OBS * 8)
    var b_t_host = ctx.enqueue_create_host_buffer[DT](8)
    var w_a_host = ctx.enqueue_create_host_buffer[DT](8 * ACT)
    var b_a_host = ctx.enqueue_create_host_buffer[DT](ACT)
    var w_b_host = ctx.enqueue_create_host_buffer[DT](8 * ACT)
    var b_b_host = ctx.enqueue_create_host_buffer[DT](ACT)
    ctx.synchronize()
    for k in range(OBS * 8):
        w_t_host.unsafe_ptr()[k] = trunk_lin_cpu_ref[].weight[k]
    for k in range(8):
        b_t_host.unsafe_ptr()[k] = trunk_lin_cpu_ref[].bias[k]
    for k in range(8 * ACT):
        w_a_host.unsafe_ptr()[k] = head_a_lin_cpu_ref[].weight[k]
    for k in range(ACT):
        b_a_host.unsafe_ptr()[k] = head_a_lin_cpu_ref[].bias[k]
    for k in range(8 * ACT):
        w_b_host.unsafe_ptr()[k] = head_b_lin_cpu_ref[].weight[k]
    for k in range(ACT):
        b_b_host.unsafe_ptr()[k] = head_b_lin_cpu_ref[].bias[k]
    ctx.enqueue_copy(actor_gpu.trunk.children[0].weight_dev.value(), w_t_host)
    ctx.enqueue_copy(actor_gpu.trunk.children[0].bias_dev.value(),   b_t_host)
    ctx.enqueue_copy(actor_gpu.heads.branch_a.weight_dev.value(),    w_a_host)
    ctx.enqueue_copy(actor_gpu.heads.branch_a.bias_dev.value(),      b_a_host)
    ctx.enqueue_copy(actor_gpu.heads.branch_b.weight_dev.value(),    w_b_host)
    ctx.enqueue_copy(actor_gpu.heads.branch_b.bias_dev.value(),      b_b_host)

    # Input.
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    ctx.synchronize()
    for k in range(BATCH * OBS):
        in_host.unsafe_ptr()[k] = Scalar[DT](Float32(k) * 0.15 - 0.5)

    var in_buf_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    for k in range(BATCH * OBS):
        in_buf_cpu[k] = in_host.unsafe_ptr()[k]
    var input_cpu  = TileTensor(in_buf_cpu, row_major[BATCH, OBS]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, 2 * ACT]())

    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * ACT)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu  = TileTensor(in_dev,  row_major[BATCH, OBS]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, 2 * ACT]())

    actor_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    actor_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * 2 * ACT)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var max_fwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(2 * ACT):
            var diff = fabs(output_cpu[b, j] - out_host.unsafe_ptr()[b * 2 * ACT + j])
            if diff > max_fwd: max_fwd = diff
    print("forward max-diff = " + String(max_fwd))
    assert_true(max_fwd < TOL, "forward parity: " + String(max_fwd))

    in_buf_cpu.free()
    out_buf_cpu.free()
    print("  test_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 StochasticActor tests (CPU + GPU, Phase 5.7)")
    print("=" * 60)
    test_forward_shape_cpu()
    test_for_each_param_cpu()
    test_backward_gradcheck_cpu()
    test_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
