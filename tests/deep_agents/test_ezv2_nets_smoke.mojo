"""EZv2 SimSiam projector + predictor nets — build + CPU forward + BN seam.

Checks the two new EZv2 nets (the MuZero h/g/f are validated separately by
`test_mz_nets_smoke`):
  * ``EZProjectorNet[HIDDEN, PROJ, PROJ_HID]`` — IN=HIDDEN, OUT=PROJ, finite fwd.
  * ``EZPredictorNet[PROJ, BOTTLENECK]`` — IN=PROJ, OUT=PROJ, finite fwd.
  * The BatchNorm train/eval seam: ``set_attr["training"]`` flips mode through
    the whole Sequential (default training; eval is deterministic).

Run:
    pixi run mojo run -I . tests/deep_agents/test_ezv2_nets_smoke.mojo
"""

from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    EZProjectorNet,
    EZPredictorNet,
)


def _all_finite(p: List[Scalar[DT]], n: Int) -> Bool:
    for i in range(n):
        var v = Float64(p[i])
        if not (v == v) or v > 1e30 or v < -1e30:
            return False
    return True


def main() raises:
    comptime HIDDEN = 16     # = LATENT of the rep net
    comptime PROJ = 32
    comptime PROJ_HID = 32
    comptime BOTTLENECK = 16
    comptime B = 8

    comptime Proj = EZProjectorNet[HIDDEN, PROJ, PROJ_HID]
    comptime Pred = EZPredictorNet[PROJ, BOTTLENECK]

    # ── Contracts ──
    assert_equal(Proj.IN_DIMS[0], HIDDEN, "projector IN")
    assert_equal(Proj.OUT_DIM, PROJ, "projector OUT")
    assert_equal(Pred.IN_DIMS[0], PROJ, "predictor IN")
    assert_equal(Pred.OUT_DIM, PROJ, "predictor OUT")
    print("contracts: OK")

    var proj = Proj.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()

    # ── projector: hidden → projection ──
    var hin = Tensor.alloc(B * HIDDEN)
    for i in range(B * HIDDEN):
        hin.data[i] = Scalar[DT](0.13) * Scalar[DT](i % 9) - Scalar[DT](0.4)
    var pj = Tensor.alloc(B * PROJ)
    proj.forward["cpu", B](TensorRefs[Proj.ARITY](hin), pj, None)
    assert_true(_all_finite(pj.data, B * PROJ), "projector non-finite (train mode)")
    print("projector forward finite (train mode): OK")

    # ── predictor: projection → projection ──
    var pr = Tensor.alloc(B * PROJ)
    pred.forward["cpu", B](TensorRefs[Pred.ARITY](pj), pr, None)
    assert_true(_all_finite(pr.data, B * PROJ), "predictor non-finite (train mode)")
    print("predictor forward finite (train mode): OK")

    # ── BN eval seam: flip to eval, forward must stay finite ──
    proj.set_attr["training"](Scalar[DT](0.0))
    pred.set_attr["training"](Scalar[DT](0.0))
    var pj2 = Tensor.alloc(B * PROJ)
    proj.forward["cpu", B](TensorRefs[Proj.ARITY](hin), pj2, None)
    assert_true(_all_finite(pj2.data, B * PROJ), "projector non-finite (eval mode)")
    print("BN eval seam (set_attr['training']=0) finite: OK")

    print("EZv2 SimSiam nets smoke: OK")
