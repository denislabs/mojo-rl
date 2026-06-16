"""Phase C gate — PCN checkpoint round-trip + AdamW (CPU).

PCN has NO checkpointing of its own. Because `PCModule` conforms to nn2
`Module` (its weight slab is a `Saveable` `Param`), it gets the v2
checkpoint envelope for free: `save_state_v2` / `load_state_v2` walk
`for_each_param` and serialize named, size-checked sections.

This test also drives training with **AdamW** (not Adam), exercising the
optimizer-generic `pc_module_train_one_batch[OPT: Optimizer]` — i.e. the
trainer is now "fully on nn2 Adam/AdamW".

Validates:
  - PCN gains checkpointing (new capability): trained weights survive a
    save → fresh-net → load round-trip, byte-faithful.
  - `pc_module_train_one_batch` works with AdamW as well as Adam.
  - The reloaded net is functionally identical (same `forward_eval`).

Run:
    pixi run mojo run -I . tests/pcn/test_pc_module_checkpoint_nn2.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from std.math import abs
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.optimizer import AdamW
from mojo_rl.nn2.core.checkpoint import save_state_v2, load_state_v2

from mojo_rl.experimental.pcn.pc_block import PCBlock
from mojo_rl.experimental.pcn.predictive_model import PCIdentity
from mojo_rl.experimental.pcn.pc_sequential import PCSequential
from mojo_rl.experimental.pcn.pc_module import PCModule
from mojo_rl.experimental.pcn.pc_initializer import PCXavier
from mojo_rl.experimental.pcn.pc_module_trainer import pc_module_train_one_batch


def main() raises:
    comptime IN = 4
    comptime H = 8
    comptime OUT = 2
    comptime BATCH = 16
    comptime T_INFER = 20
    comptime PATH = String("/tmp/test_pc_module_ckpt.ckpt")

    comptime Net = PCModule[
        PCBlock[IN, H, PCIdentity],
        PCBlock[H, OUT, PCIdentity],
    ]
    comptime Seq = PCSequential[
        PCBlock[IN, H, PCIdentity],
        PCBlock[H, OUT, PCIdentity],
    ]
    comptime PSIZE = Seq.PARAM_SIZE

    seed(0)

    # Data: a representable linear target.
    var x_s = List[Scalar[DT]](capacity=BATCH * IN)
    for _ in range(BATCH * IN):
        x_s.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    var w_true = List[Scalar[DT]](capacity=IN * OUT)
    for _ in range(IN * OUT):
        w_true.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    var y_s = List[Scalar[DT]](length=BATCH * OUT, fill=Scalar[DT](0))
    for b in range(BATCH):
        for j in range(OUT):
            var acc = Scalar[DT](0)
            for i in range(IN):
                acc += x_s[b * IN + i] * w_true[i * OUT + j]
            y_s[b * OUT + j] = acc
    var x_in = LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin](
        x_s.unsafe_ptr()
    )
    var y_target = LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        y_s.unsafe_ptr()
    )

    # Train with AdamW (proves optimizer-generic trainer).
    var net = Net.make_pcn[PCXavier]()
    var opt = AdamW.make["cpu", Net](net)
    opt.lr = Scalar[DT](1e-2)
    for _ in range(60):
        _ = pc_module_train_one_batch[BATCH](
            net, opt, x_in, y_target, T_INFER, Scalar[DT](0.1)
        )

    # Save trained weights.
    save_state_v2[Net](net, PATH)

    # Fresh net (different draws) — must differ before load.
    var net2 = Net.make_pcn[PCXavier]()
    var pre_diff = Float64(0.0)
    for k in range(PSIZE):
        pre_diff += abs(
            Float64(net.weights.val.cpu[k]) - Float64(net2.weights.val.cpu[k])
        )
    assert_true(pre_diff > 1e-3, "fresh net unexpectedly equals trained net")

    # Load — weights must now match the trained net.
    load_state_v2[Net](net2, PATH)
    var post_diff = Float64(0.0)
    for k in range(PSIZE):
        post_diff += abs(
            Float64(net.weights.val.cpu[k]) - Float64(net2.weights.val.cpu[k])
        )
    print("weight |Δ| sum: pre-load =", pre_diff, " post-load =", post_diff)
    assert_true(post_diff < 1e-4, "loaded weights do not match saved weights")

    # Functional equivalence: same forward_eval output.
    var o1 = List[Scalar[DT]](length=BATCH * OUT, fill=Scalar[DT](0))
    var o2 = List[Scalar[DT]](length=BATCH * OUT, fill=Scalar[DT](0))
    var p1 = LayoutTensor[DT, Layout.row_major(PSIZE), MutAnyOrigin](
        net.weights.value_unsafe_ptr_cpu()
    )
    var p2 = LayoutTensor[DT, Layout.row_major(PSIZE), MutAnyOrigin](
        net2.weights.value_unsafe_ptr_cpu()
    )
    var out1 = LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        o1.unsafe_ptr()
    )
    var out2 = LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        o2.unsafe_ptr()
    )
    Seq.forward_eval[BATCH](x_in, p1, out1)
    Seq.forward_eval[BATCH](x_in, p2, out2)
    var out_diff = Float64(0.0)
    for k in range(BATCH * OUT):
        out_diff += abs(Float64(o1[k]) - Float64(o2[k]))
    assert_true(out_diff < 1e-4, "reloaded net forward differs")

    print("PASS: PCN checkpoint round-trip + AdamW (Phase C)")
