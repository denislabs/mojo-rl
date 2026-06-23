"""Localize the NVIDIA crash in the dyn reward path: print around every op so one
run shows the last line before the abort. Reduced to 2 iterations.

Run: pixi run -e nvidia mojo run -I . tests/deep_agents/test_mz_dyn_reward_debug.mojo
Report the LAST printed line before the crash."""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets_spatial import MZDynNetC4Spatial
from mojo_rl.deep_agents.muzero.loss_ops import soft_ce_slice_loss_and_grad
from mojo_rl.deep_agents.zero.twohot_targets import mz_two_hot_target_batch


def main() raises:
    comptime CH = 8
    comptime ACT = 7
    comptime BINS = 51
    comptime HH = 6
    comptime WW = 7
    comptime NB = 2
    comptime LATENT = CH * HH * WW
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime B = 16
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW, NB]

    print("A: making dyn net ...")
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    print("B: made. making Adam ...")
    var opt = Adam(lr=Scalar[DT](2e-3))

    var din = Tensor.alloc(B * DYN_IN)
    for b in range(B):
        var base = b * DYN_IN
        for i in range(LATENT):
            din.data[base + i] = Scalar[DT](0.05) * Scalar[DT](((b * 7 + i) % 11) - 5)
        for a in range(ACT):
            din.data[base + LATENT + a] = Scalar[DT](0.0)
        din.data[base + LATENT + (b % ACT)] = Scalar[DT](1.0)
    print("C: built input din")

    var r_scalars = List[Scalar[DT]](length=B, fill=0)
    for b in range(B):
        r_scalars[b] = Scalar[DT]((b % 3) - 1)
    var twr = Tensor.alloc(B * BINS)
    mz_two_hot_target_batch[B, BINS](r_scalars, 0, Scalar[DT](-1.0), Scalar[DT](1.0), twr.data, 0)
    print("D: built two-hot reward target")

    var dout = Tensor.alloc(B * DYN_OUT)
    var gdout = Tensor.alloc(B * DYN_OUT)
    var gdin = Tensor.alloc(B * DYN_IN)
    var gscale = Scalar[DT](1.0) / Scalar[DT](B)

    for step in range(2):
        print("--- step", step, ": E zero_grad ...")
        dyn.zero_grad["cpu"](None)
        print("    F forward ...")
        dyn.forward["cpu", B](TensorRefs[Dyn.ARITY](din), dout, None)
        print("    G forward OK. seeding grad ...")
        for i in range(B * DYN_OUT):
            gdout.data[i] = Scalar[DT](0)
        var rl = soft_ce_slice_loss_and_grad[B, DYN_OUT, LATENT, BINS](
            dout.data, twr.data, gscale, gdout.data
        )
        print("    H loss =", rl / Scalar[DT](B), ". vjp ...")
        dyn.vjp["cpu", B](
            TensorRefs[Dyn.ARITY](din), gdout, TensorRefs[Dyn.ARITY](gdin), None
        )
        print("    I vjp OK. adam ...")
        opt.begin_step()
        dyn.for_each_param["cpu"](opt, None)
        print("    J adam OK. step", step, "done.")

    print("ALL DONE — no crash")
