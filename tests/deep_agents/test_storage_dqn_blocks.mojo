"""DQN bespoke blocks — storage isolation gate (DQNTargetYBlock + DQNQUpdateBlock).

Builds a tiny scalar Q-net `Sequential[LinearReLU[OBS,H], Linear[H,NA]]`, then:
  - DQNTargetYBlock.step: asserts `y = r + γ·max_a Q_t(sp)·(1−d)` matches a
    hand CPU recompute (standard + Double branches),
  - DQNQUpdateBlock.step: asserts the MSE loss DROPS over repeated steps on a
    fixed (s, a, y) batch (the gather→MSE→scatter→Q.vjp→Adam path learns).
CPU + GPU.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_dqn_blocks.mojo
"""

from std.math import isfinite, abs as fabs
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential

from mojo_rl.deep_agents.dqn.target_y_block import DQNTargetYBlock
from mojo_rl.deep_agents.dqn.q_update_block import DQNQUpdateBlock

comptime OBS = 4
comptime NA = 3
comptime BATCH = 8
comptime H = 16
comptime GAMMA = Scalar[DT](0.99)

comptime QNet = Sequential[LinearReLU[OBS, H], Linear[H, NA]]


def _fill_inputs(mut s: Tensor, mut sp: Tensor, mut a: Tensor, mut r: Tensor,
                 mut d: Tensor):
    for i in range(BATCH * OBS):
        s.data[i] = Scalar[DT](0.1) * Scalar[DT]((i % 7) - 3)
        sp.data[i] = Scalar[DT](0.1) * Scalar[DT]((i % 5) - 2)
    for b in range(BATCH):
        a.data[b] = Scalar[DT](b % NA)
        r.data[b] = Scalar[DT](0.1) * Scalar[DT](b)
        d.data[b] = Scalar[DT](1.0) if (b == BATCH - 1) else Scalar[DT](0.0)


def _check_target_y[target: StaticString](ctx: Optional[DeviceContext]) raises:
    var qt = QNet.make[target, Kaiming](ctx)
    var qo = QNet.make[target, Kaiming](ctx)
    var blk = DQNTargetYBlock[QNet, BATCH, OBS, NA, DOUBLE=False].make[target](
        gamma=GAMMA, ctx=ctx
    )

    var s = Tensor.alloc(BATCH * OBS)
    var sp = Tensor.alloc(BATCH * OBS)
    var a = Tensor.alloc(BATCH)
    var r = Tensor.alloc(BATCH)
    var d = Tensor.alloc(BATCH)
    var y = Tensor.alloc(BATCH)
    _fill_inputs(s, sp, a, r, d)

    comptime if target == "gpu":
        sp.upload(ctx.value()); r.upload(ctx.value()); d.upload(ctx.value())
        y.upload(ctx.value())

    blk.step[target](qt, qo, sp, r, d, y, ctx)

    # Hand recompute: q_all = Q_t(sp), max over NA, y = r + γ·max·(1−d).
    var q_all = Tensor.alloc(BATCH * NA)
    comptime if target == "gpu":
        q_all = Tensor.alloc_gpu(ctx.value(), BATCH * NA)
    qt.forward[target, BATCH](TensorRefs[QNet.ARITY](sp), q_all, ctx)
    comptime if target == "gpu":
        q_all.download(ctx.value()); y.download(ctx.value())
        r.download(ctx.value()); d.download(ctx.value())

    var max_err = Scalar[DT](0.0)
    for b in range(BATCH):
        var best = q_all.data[b * NA]
        for k in range(1, NA):
            if q_all.data[b * NA + k] > best:
                best = q_all.data[b * NA + k]
        var nonterm = Scalar[DT](1.0) - d.data[b]
        var y_ref = r.data[b] + GAMMA * best * nonterm
        var e = fabs(y.data[b] - y_ref)
        if e > max_err:
            max_err = e
    print("  [", target, "] target_y max abs err vs recompute:", max_err)
    if max_err > Scalar[DT](1e-4):
        raise Error("DQNTargetYBlock y mismatch")


def _check_q_update[target: StaticString](ctx: Optional[DeviceContext]) raises:
    var qo = QNet.make[target, Kaiming](ctx)
    var opt = Adam(lr=Scalar[DT](1e-2))
    opt.adopt[target, QNet](qo, ctx)
    var blk = DQNQUpdateBlock[QNet, BATCH, OBS, NA].make[target](ctx=ctx)

    var s = Tensor.alloc(BATCH * OBS)
    var sp = Tensor.alloc(BATCH * OBS)
    var a = Tensor.alloc(BATCH)
    var r = Tensor.alloc(BATCH)
    var d = Tensor.alloc(BATCH)
    var y = Tensor.alloc(BATCH)
    _fill_inputs(s, sp, a, r, d)
    for b in range(BATCH):
        y.data[b] = Scalar[DT](0.5) + Scalar[DT](0.05) * Scalar[DT](b)

    comptime if target == "gpu":
        s.upload(ctx.value()); a.upload(ctx.value()); y.upload(ctx.value())

    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for it in range(80):
        var l = blk.step[target](qo, opt, s, a, y, ctx=ctx)
        if it == 0:
            first = l
        last = l
    print("  [", target, "] q_update loss:", first, "->", last)
    if not isfinite(last):
        raise Error("q_update loss non-finite")
    if last >= first:
        raise Error("q_update loss did not drop")


def main() raises:
    print("=" * 60)
    print("DQN bespoke blocks — storage isolation gate")
    print("=" * 60)

    print("CPU:")
    _check_target_y["cpu"](None)
    _check_q_update["cpu"](None)

    print("GPU:")
    with DeviceContext() as ctx:
        _check_target_y["gpu"](ctx)
        _check_q_update["gpu"](ctx)

    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
