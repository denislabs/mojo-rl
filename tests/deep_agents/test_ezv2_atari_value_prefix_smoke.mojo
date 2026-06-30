"""EZv2-Atari value-prefix nets smoke (Stage 3) — shapes + finite grads, CPU.

Exercises the value-prefix building blocks added for parity decision B1:
  * `EZDynZNetAtari[ACT]` — the z'-only dynamics graph ([z|act] → z', OUT=LATENT,
    no fused reward slot).
  * `EZRewardLSTMAtari[BINS]` — the stateful LSTM value-prefix reward head; driven
    via its recurrent step API (reward_step_forward / reward_step_backward) with
    caller-owned (h,c) carry buffers.

Forward: rep-less synthetic z → dyn_z(z|act) → z' → reward_step_forward(z',h,c) →
value-prefix logits. Backward: reward_step_backward (with zero carry) → grad wrt
z' (finite + non-zero) and ≥1 reward-head param per sub-module gets grad. Then
`ez_atari_init_zero_reward` must make the value-prefix logits ~0.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_value_prefix_smoke.mojo
"""

from std.math import isnan, isinf, abs
from std.testing import assert_true, assert_equal
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    EZDynZNetAtari, EZRewardLSTMAtari, EZDynVPNetAtari, ez_atari_init_zero_reward,
    EZ_LATENT, EZ_LSTM_HIDDEN, EZ_RHID,
)


comptime ACT = 18
comptime BINS = 601
comptime B = 2
comptime H = EZ_LSTM_HIDDEN

comptime DynZ = EZDynZNetAtari[ACT]
comptime Rew = EZRewardLSTMAtari[BINS]
comptime DYN_IN = EZ_LATENT + ACT


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _finite(p: List[Scalar[DT]], n: Int) -> Bool:
    for i in range(n):
        if isnan(p[i]) or isinf(p[i]):
            return False
    return True


def _any_nonzero(p: List[Scalar[DT]], n: Int) -> Bool:
    for i in range(n):
        if p[i] != Scalar[DT](0.0):
            return True
    return False


def _max_abs(p: List[Scalar[DT]], n: Int) -> Scalar[DT]:
    var m = Scalar[DT](0.0)
    for i in range(n):
        var a = abs(p[i])
        if a > m:
            m = a
    return m


struct GradStats(ParamVisitor):
    var n_params: Int
    var n_nonzero: Int

    def __init__(out self):
        self.n_params = 0
        self.n_nonzero = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.n_params += 1
        for i in range(N):
            if grad.data[i] != Scalar[DT](0.0):
                self.n_nonzero += 1
                break


def main() raises:
    print("=" * 70)
    print("EZv2-Atari value-prefix nets smoke (Stage 3, CPU)")
    print("=" * 70)

    # ── dim contract ──
    print("DynZ.OUT_DIM =", DynZ.OUT_DIM, " (expect", EZ_LATENT, ")")
    print("Rew.OUT_DIM  =", Rew.OUT_DIM, " (expect", BINS + EZ_RHID, ")")
    assert_equal(DynZ.IN_DIMS[0], DYN_IN, "dyn_z input = LATENT+ACT")
    assert_equal(DynZ.OUT_DIM, EZ_LATENT, "dyn_z output = LATENT (no reward)")
    assert_equal(Rew.IN_DIMS[0], EZ_LATENT, "reward in[0] = LATENT")
    assert_equal(Rew.IN_DIMS[1], EZ_RHID, "reward in[1] = RHID (h;c)")
    assert_equal(Rew.OUT_DIM, BINS + EZ_RHID, "reward out = BINS+RHID")

    var dyn = DynZ.make["cpu", Kaiming]()
    dyn.set_attr["training"](Scalar[DT](1.0))
    var rew = Rew.make["cpu", Kaiming]()
    rew.set_attr["training"](Scalar[DT](1.0))

    # ── forward: dyn_z(z|act) → z' → reward_step_forward → value-prefix ──
    var din = Tensor.alloc(B * DYN_IN)
    var zprime = Tensor.alloc(B * EZ_LATENT)
    for b in range(B):
        for i in range(EZ_LATENT):
            din.data[b * DYN_IN + i] = _det(b * EZ_LATENT + i + 1, 0.3)
        din.data[b * DYN_IN + EZ_LATENT + (b % ACT)] = Scalar[DT](1.0)  # onehot
    dyn.forward["cpu", B](TensorRefs[DynZ.ARITY](din), zprime, None)
    assert_true(_finite(zprime.data, B * EZ_LATENT), "z' finite")

    # caller-owned (h,c) carry: slab0=prev (zero root state), slab1=out.
    var h = Tensor.alloc(2 * B * H)
    var c = Tensor.alloc(2 * B * H)
    var cache = Tensor.alloc(B * Rew.CACHE_SIZE)
    var vp = Tensor.alloc(B * BINS)
    rew.reward_step_forward["cpu", B](zprime, h, c, cache, vp, None)
    assert_true(_finite(vp.data, B * BINS), "value-prefix logits finite")
    assert_true(_any_nonzero(vp.data, B * BINS), "value-prefix logits non-zero")
    assert_true(_any_nonzero(h.data, 2 * B * H), "h_t written")

    # ── backward: reward_step_backward with zero recurrent carry ──
    var grad_vp = Tensor.alloc(B * BINS)
    for k in range(B * BINS):
        grad_vp.data[k] = _det(k + 7, 1.0)
    var dh_carry = Tensor.alloc(B * H)   # zero (last step)
    var dc_carry = Tensor.alloc(B * H)   # zero
    var grad_zprime = Tensor.alloc(B * EZ_LATENT)
    var dh_prev = Tensor.alloc(B * H)
    var dc_prev = Tensor.alloc(B * H)
    rew.reward_step_backward["cpu", B](
        zprime, grad_vp, h, c, cache, dh_carry, dc_carry,
        grad_zprime, dh_prev, dc_prev, None,
    )
    assert_true(_finite(grad_zprime.data, B * EZ_LATENT), "grad z' finite")
    assert_true(_any_nonzero(grad_zprime.data, B * EZ_LATENT), "grad z' non-zero")
    assert_true(_any_nonzero(dh_prev.data, B * H), "dh_prev non-zero (BPTT carry)")

    var gs = GradStats()
    rew.for_each_param["cpu"](gs, None)
    print("   reward params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "reward head has params")
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 8,
                "≥80% of reward params receive grad")

    # ── init_zero: value-prefix logits collapse to ~0 ──
    var rew0 = Rew.make["cpu", Kaiming]()
    rew0.set_attr["training"](Scalar[DT](1.0))
    ez_atari_init_zero_reward["cpu", BINS](rew0, None)
    var h0 = Tensor.alloc(2 * B * H)
    var c0 = Tensor.alloc(2 * B * H)
    var cache0 = Tensor.alloc(B * Rew.CACHE_SIZE)
    var vp0 = Tensor.alloc(B * BINS)
    rew0.reward_step_forward["cpu", B](zprime, h0, c0, cache0, vp0, None)
    var mx = _max_abs(vp0.data, B * BINS)
    print("   init_zero |value-prefix logits|max =", mx)
    assert_true(mx < Scalar[DT](1e-5), "init_zero → ~0 value-prefix logits")

    # ── fused EZDynVPNetAtari (search drop-in): contract + forward ──
    comptime VPDyn = EZDynVPNetAtari[ACT, BINS]
    assert_equal(VPDyn.IN_DIMS[0], DYN_IN, "VP dyn input = LATENT+ACT (drop-in)")
    assert_equal(VPDyn.OUT_DIM, EZ_LATENT + BINS, "VP dyn output = LATENT+BINS")
    var vpd = VPDyn.make["cpu", Kaiming]()
    vpd.set_attr["training"](Scalar[DT](1.0))
    var vpd_in = Tensor.alloc(B * DYN_IN)
    var vpd_out = Tensor.alloc(B * (EZ_LATENT + BINS))
    for b in range(B):
        for i in range(EZ_LATENT):
            vpd_in.data[b * DYN_IN + i] = _det(b * EZ_LATENT + i + 1, 0.3)
        vpd_in.data[b * DYN_IN + EZ_LATENT + (b % ACT)] = Scalar[DT](1.0)
    vpd.forward["cpu", B](TensorRefs[VPDyn.ARITY](vpd_in), vpd_out, None)
    assert_true(_finite(vpd_out.data, B * (EZ_LATENT + BINS)), "VP dyn forward finite")
    assert_true(_any_nonzero(vpd_out.data, B * EZ_LATENT), "VP dyn z' non-zero")

    _ = dyn^
    _ = rew^
    _ = rew0^
    _ = vpd^
    print("=" * 70)
    print("PASS — value-prefix nets (dyn_z + LSTM reward head + fused VP dyn) CPU smoke")
    print("=" * 70)
