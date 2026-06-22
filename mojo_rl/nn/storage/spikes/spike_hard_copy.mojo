"""Gate for `hard_copy` (zero-series gap #2) — verbatim Param+State Module copy.

Net = Sequential[Linear, BatchNorm1D] (Params: Linear W/b + BN γ/β; States: BN
running_mean / running_var). Fill `src` with distinct known values (params AND
states), make `dst` from a different init, hard_copy src→dst, then compare every
param + state element-wise — must be BIT-IDENTICAL, including the BN running
stats (the params-only-copy bug this guards against).

Run: pixi run mojo run -I . mojo_rl/nn/storage/spikes/spike_hard_copy.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.param import ParamVisitor
from mojo_rl.nn.storage.core.initializer import Deterministic, Zero
from mojo_rl.nn.storage.core.hard_copy import hard_copy, _CollectVisitor
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.batch_norm_1d import BatchNorm1D


struct _FillVisitor(ParamVisitor):
    """Writes param.data[i] = seed + i*0.1 into every visited Param/State, so
    the gate exercises non-default values (incl. BN running stats)."""
    var seed: Scalar[DT]

    def __init__(out self, seed: Scalar[DT]):
        self.seed = seed

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            param.data[i] = self.seed + Scalar[DT](i) * Scalar[DT](0.1)


def main() raises:
    comptime IN = 4
    comptime H = 3
    comptime target = "cpu"
    comptime ctx = Optional[DeviceContext](None)

    var src = Sequential[Linear[IN, H], BatchNorm1D[H]].make[target, Deterministic]()
    var dst = Sequential[Linear[IN, H], BatchNorm1D[H]].make[target, Zero]()

    # Fill src params + states with distinct known values.
    var fill = _FillVisitor(Scalar[DT](3.0))
    src.for_each_param[target](fill, ctx)
    src.for_each_state[target](fill, ctx)

    # Confirm src and dst differ BEFORE the copy.
    var cs0 = _CollectVisitor()
    src.for_each_param[target](cs0, ctx)
    src.for_each_state[target](cs0, ctx)
    var cd0 = _CollectVisitor()
    dst.for_each_param[target](cd0, ctx)
    dst.for_each_state[target](cd0, ctx)
    var pre_diff: Scalar[DT] = 0
    for s in range(len(cs0.vals)):
        for i in range(len(cs0.vals[s])):
            var d = cs0.vals[s][i] - cd0.vals[s][i]
            pre_diff = max(pre_diff, d if d >= 0 else -d)

    hard_copy[target](src, dst)

    # Compare every param + state element-wise AFTER the copy.
    var cs = _CollectVisitor()
    src.for_each_param[target](cs, ctx)
    src.for_each_state[target](cs, ctx)
    var cd = _CollectVisitor()
    dst.for_each_param[target](cd, ctx)
    dst.for_each_state[target](cd, ctx)

    var post_diff: Scalar[DT] = 0
    var n_state = 0
    for s in range(len(cs.vals)):
        if cs.names[s].find("running") != -1:
            n_state += 1
        for i in range(len(cs.vals[s])):
            var d = cs.vals[s][i] - cd.vals[s][i]
            post_diff = max(post_diff, d if d >= 0 else -d)

    print("sections:", len(cs.vals), " state(running) sections:", n_state)
    print("pre-copy  max|src-dst|:", pre_diff)
    print("post-copy max|src-dst|:", post_diff)
    if pre_diff > Scalar[DT](0) and post_diff == Scalar[DT](0) and n_state == 2:
        print("HARD_COPY OK — params + BN running stats bit-identical")
    else:
        print("HARD_COPY FAIL")
