"""GPU smoke for `hard_copy` (gap #2) — exercises the device download→upload
copy path. src/dst are made from DIFFERENT inits (device params differ), then
hard_copy src→dst; collecting (which downloads) must show bit-identical params.

Run (Apple Metal): pixi run -e apple mojo run -I . \
    mojo_rl/nn/storage/spikes/spike_hard_copy_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Deterministic, Zero
from mojo_rl.nn.storage.core.hard_copy import hard_copy, _CollectVisitor
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.batch_norm_1d import BatchNorm1D


def main() raises:
    comptime IN = 4
    comptime H = 3
    comptime target = "gpu"
    var ctx = DeviceContext()
    var octx = Optional[DeviceContext](ctx)

    var src = Sequential[Linear[IN, H], BatchNorm1D[H]].make[target, Deterministic](octx)
    var dst = Sequential[Linear[IN, H], BatchNorm1D[H]].make[target, Zero](octx)

    var cs0 = _CollectVisitor()
    src.for_each_param[target](cs0, octx)
    var cd0 = _CollectVisitor()
    dst.for_each_param[target](cd0, octx)
    var pre_diff: Scalar[DT] = 0
    for s in range(len(cs0.vals)):
        for i in range(len(cs0.vals[s])):
            var d = cs0.vals[s][i] - cd0.vals[s][i]
            pre_diff = max(pre_diff, d if d >= 0 else -d)

    hard_copy[target](src, dst, octx)

    var cs = _CollectVisitor()
    src.for_each_param[target](cs, octx)
    src.for_each_state[target](cs, octx)
    var cd = _CollectVisitor()
    dst.for_each_param[target](cd, octx)
    dst.for_each_state[target](cd, octx)
    var post_diff: Scalar[DT] = 0
    for s in range(len(cs.vals)):
        for i in range(len(cs.vals[s])):
            var d = cs.vals[s][i] - cd.vals[s][i]
            post_diff = max(post_diff, d if d >= 0 else -d)

    print("pre-copy  max|src-dst| params:", pre_diff)
    print("post-copy max|src-dst| params+states:", post_diff)
    if pre_diff > Scalar[DT](0) and post_diff == Scalar[DT](0):
        print("HARD_COPY_GPU OK — device download/upload copy bit-identical")
    else:
        print("HARD_COPY_GPU FAIL")
