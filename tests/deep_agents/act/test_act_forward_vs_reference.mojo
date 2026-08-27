# +--------------------------------------------------------------------------+ #
# | M5 gate — the whole ACT model vs the reference DETRVAE
# +--------------------------------------------------------------------------+ #
"""Gates `deep_agents/act/loss_graph.mojo` end to end against
`references/act-main`'s `DETRVAE`, built by its own `build()` and run on its own
weights.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref
    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/t \\
        tests/deep_agents/act/test_act_forward_vs_reference.mojo && /tmp/t

⚠ `-Xlinker -ld_classic` required — the expanded graph type mangles past
Apple's linker symbol-length limit. See the M3 gate's header.

## What makes this a real comparison

* Every one of the graph's parameters is loaded from the dump BY NAME, and full
  coverage is asserted. A missed parameter keeps its random init and shows up as
  a small numerical disagreement — indistinguishable from a genuine one.
* The reference's `reparametrize` is patched to `z = mu` for the dump and the
  Mojo `z` node is set deterministic. Two RNG streams cannot agree; the sampling
  arithmetic is gated separately in the M4 gate against an injected `eps`.
* Dropout is off on both sides; BatchNorm runs on running statistics on both.
* `QPOS = ADIM = 14`, because `detr_vae.py` HARDCODES 14 into three layers.
  Patching the reference down to the SO-101's 6 would mean gating against
  something we edited. Our graph is parameterized; 14 exercises the same code.

## What it checks

`a_hat` (the action chunk), `latent_info` (mu and logvar together — the CVAE
encoder's whole output), and both loss terms. `latent_info` is checked
SEPARATELY from `a_hat` because they sit on different paths: `a_hat` would still
look reasonable if the CVAE encoder were subtly wrong, since the latent enters
the decoder through a single token among 14.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.act.loss_graph import ACTLossGraph
from mojo_rl.deep_agents.act.refload import ListParams, LoadRefParams, RefDump


comptime REF_DIR = "/tmp/act_ref"

# Must match `dump_act_reference.py:section_detrvae`.
comptime B = 2
comptime QPOS = 14
comptime ADIM = 14
comptime N_CAM = 2
comptime IMG_H = 64
comptime IMG_W = 96
comptime K = 6
comptime DIM = 16
comptime HEADS = 2
comptime FF = 32
comptime LATENT = 32  # `detr_vae.py:69  self.latent_dim = 32`
comptime N_ENC = 1
comptime N_DEC = 1

comptime KL_WEIGHT = 10.0

comptime G = ACTLossGraph[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC
]

comptime TOL = 3e-4
"""Loose relative to the layer gates (5e-7): this composes a 20-conv ResNet with
two transformer stacks in fp32, and the reference's own reductions run in a
different order. The per-piece gates hold the tight tolerances."""


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def worst(mut t: Tensor, ref b: List[Scalar[DT]], n: Int) -> Float64:
    var w = Float64(0.0)
    for i in range(n):
        w = max(w, abs(Float64(t.data[i]) - Float64(b[i])))
    return w


def load_into(mut t: Tensor, ref src: List[Scalar[DT]], n: Int):
    t.ensure(n)
    for i in range(n):
        t.data[i] = src[i]


def main() raises:
    var fails = 0
    print("ACT full-model gate (reference: " + String(REF_DIR) + ")")
    print("")

    var d = RefDump(String(REF_DIR))
    var g = G.make["cpu", Kaiming]()
    g.set_attr["training"](Scalar[DT](0.0))  # dropout off, BN on running stats

    # ── weights + BN statistics, by name, with coverage asserted ─────────
    var wl = LoadRefParams["vae."](RefDump(String(REF_DIR)))
    g.for_each_param["cpu"](wl, None, String(""))
    var plist = ListParams()
    g.for_each_param["cpu"](plist, None, String(""))
    check(
        fails,
        "every parameter loaded from the reference",
        len(wl.missing) == 0 and len(wl.loaded) == len(plist.names),
        String(len(wl.loaded)) + "/" + String(len(plist.names))
        + (", first missing: " + wl.missing[0] if len(wl.missing) > 0 else ""),
    )

    var sl = LoadRefParams["vae."](RefDump(String(REF_DIR)))
    g.for_each_state["cpu"](sl, None, String(""))
    var slist = ListParams()
    g.for_each_state["cpu"](slist, None, String(""))
    check(
        fails,
        "every BatchNorm running statistic loaded",
        len(sl.missing) == 0 and len(sl.loaded) == len(slist.names),
        String(len(sl.loaded)) + "/" + String(len(slist.names))
        + (", first missing: " + sl.missing[0] if len(sl.missing) > 0 else ""),
    )

    # ── inputs ───────────────────────────────────────────────────────────
    comptime IMG_N = B * N_CAM * 3 * IMG_H * IMG_W
    var qpos = Tensor()
    var images = Tensor()
    var actions = Tensor()
    var ev = Tensor()
    load_into(qpos, d.get(String("vae_qpos")), B * QPOS)
    load_into(images, d.get(String("vae_images")), IMG_N)
    load_into(actions, d.get(String("vae_actions")), B * K * ADIM)

    # enc_valid = [1, 1, valid...] — the CVAE encoder's mask over
    # `[CLS] | qpos | a_1..a_K`, whose first two entries are never padding
    # (`detr_vae.py:96 cls_joint_is_pad`).
    var valid = d.get(String("vae_valid"))
    ev.ensure(B * (K + 2))
    for b in range(B):
        ev.data[b * (K + 2) + 0] = Scalar[DT](1.0)
        ev.data[b * (K + 2) + 1] = Scalar[DT](1.0)
        for t in range(K):
            ev.data[b * (K + 2) + 2 + t] = valid[b * K + t]

    g.set_input["qpos", B](qpos)
    g.set_input["images", B](images)
    g.set_input["actions", B](actions)
    g.set_input["enc_valid", B](ev)

    # The reference's `reparametrize` was patched to `z = mu` for the dump.
    g.set_node_attr["z", "deterministic"](Scalar[DT](1.0))
    g.set_node_attr["zs", "multiplier"](Scalar[DT](1.0))
    g.set_node_attr["kls", "multiplier"](Scalar[DT](KL_WEIGHT))

    var loss = Tensor()
    g.forward[B, "cpu"](loss)

    # ── the CVAE encoder's output, on its own ────────────────────────────
    ref latinfo = g.node_output["latinfo"]()
    var lref = d.get(String("vae_latinfo"))
    var lw = worst(latinfo, lref, B * 2 * LATENT)
    check(
        fails,
        "latent_info (mu | logvar) vs DETRVAE",
        lw < TOL,
        "max|diff| = " + String(lw),
    )

    # ── the action chunk ─────────────────────────────────────────────────
    ref ahat = g.node_output["ahat"]()
    var aref = d.get(String("vae_ahat"))
    var aw = worst(ahat, aref, B * K * ADIM)
    check(
        fails,
        "a_hat (the predicted action chunk) vs DETRVAE",
        aw < TOL,
        "max|diff| = " + String(aw),
    )
    var amag = Float64(0.0)
    for i in range(B * K * ADIM):
        amag = max(amag, abs(Float64(aref[i])))
    check(
        fails,
        "a_hat is non-trivial",
        amag > 0.05,
        "max|ref a_hat| = " + String(amag),
    )

    # ── both loss terms ──────────────────────────────────────────────────
    ref l1 = g.node_output["l1"]()
    var l1ref = d.get(String("vae_l1"))
    check(
        fails,
        "L1 term vs policy.py",
        worst(l1, l1ref, B) < TOL,
        "max|diff| = " + String(worst(l1, l1ref, B)),
    )

    ref klv = g.node_output["kl"]()
    var klref = d.get(String("vae_kl"))
    check(
        fails,
        "KL term vs policy.py:kl_divergence",
        worst(klv, klref, B) < TOL,
        "max|diff| = " + String(worst(klv, klref, B)),
    )

    # The graph output must be exactly l1 + kl_weight*kl — checks the two
    # Scale/Add nodes are wired the way the config says.
    var combo_ok = Float64(0.0)
    for b in range(B):
        var want = (
            Float64(l1ref[b]) + KL_WEIGHT * Float64(klref[b])
        )
        combo_ok = max(combo_ok, abs(Float64(loss.data[b]) - want))
    check(
        fails,
        "loss == L1 + kl_weight * KL",
        combo_ok < TOL,
        "max|diff| = " + String(combo_ok),
    )

    # ── the inference path is z = 0, exactly ─────────────────────────────
    # `set_node_attr["zs","multiplier"](0.0)` must give the same numbers as the
    # reference's "skip the CVAE encoder and feed zeros", i.e. the latent token
    # becomes latent_out_proj's bias alone. Check it changes a_hat (so the
    # latent is actually load-bearing) and that the L1 term still computes.
    # ⚠ SNAPSHOT first. `node_output` returns a ref INTO the graph's pool, so
    # holding `ahat` across a second forward would compare the slot with itself
    # and report a difference of exactly 0.0 no matter what the model did.
    var ahat_z1 = Tensor.alloc(B * K * ADIM)
    for i in range(B * K * ADIM):
        ahat_z1.data[i] = ahat.data[i]

    g.set_node_attr["zs", "multiplier"](Scalar[DT](0.0))
    var loss0 = Tensor()
    g.forward[B, "cpu"](loss0)
    ref ahat0 = g.node_output["ahat"]()
    var moved = Float64(0.0)
    for i in range(B * K * ADIM):
        moved = max(
            moved, abs(Float64(ahat0.data[i]) - Float64(ahat_z1.data[i]))
        )
    check(
        fails,
        "zeroing the latent changes a_hat (the CVAE is load-bearing)",
        moved > 1e-4,
        "max|Delta a_hat| = " + String(moved),
    )

    # And the latent token must then be exactly latent_out_proj(0) = its bias,
    # identical across the batch.
    ref lattok = g.node_output["lattok"]()
    var spread = Float64(0.0)
    for j in range(DIM):
        spread = max(
            spread,
            abs(Float64(lattok.data[DIM + j]) - Float64(lattok.data[j])),
        )
    check(
        fails,
        "at z=0 the latent token is the bias alone (batch-invariant)",
        spread == 0.0,
        "max spread across the batch = " + String(spread),
    )

    # ── backward runs and produces finite, non-zero gradients ────────────
    g.set_node_attr["zs", "multiplier"](Scalar[DT](1.0))
    g.forward[B, "cpu"](loss)
    g.zero_grad["cpu"](None)
    var seed = Tensor.alloc(B)
    for i in range(B):
        seed.data[i] = Scalar[DT](1.0) / Scalar[DT](B)
    g.vjp[B, "cpu"](seed)

    var gl = ListParams()
    g.for_each_param["cpu"](gl, None, String(""))
    var gv = GradScan()
    g.for_each_param["cpu"](gv, None, String(""))
    check(
        fails,
        "every parameter received a finite gradient",
        not gv.saw_nan,
        "NaN/Inf in " + gv.bad_name if gv.saw_nan else "",
    )

    # ⚠ EXACTLY seven parameters legitimately receive no gradient, and they are
    # named rather than tolerated by a threshold — "most params got a gradient"
    # would pass while a whole subtree sat at zero.
    #
    # The first decoder layer's SELF-ATTENTION is mathematically inert in ACT.
    # `tgt` starts as zeros (`torch.zeros_like(query_embed)`), and torch
    # initializes `nn.MultiheadAttention`'s `in_proj_bias`/`out_proj.bias` to 0,
    # so `v = Linear(tgt) = b_v` is CONSTANT across tokens. Attention whose
    # values are constant returns that constant whatever q and k are — so sq,
    # sk and sv's weight can never receive gradient, and at init the whole
    # sublayer output is exactly 0 (hence n1's gamma sees a zero-variance input
    # while its beta does not). `sv.bias` and `n1.beta` DO get gradient, which
    # is why they are absent below.
    #
    # This is a property of the reference we reproduce, not a defect here — and
    # it compounds with the `hs[0]` quirk: in the published model, where only
    # decoder layer 1 affects the output, the decoder's self-attention does
    # nothing at all. With `ACT_DEC_LAYERS >= 2` and `ACT_USE_LAST_HS`, later
    # layers see a non-zero target and their self-attention is live.
    var expected_zero = List[String]()
    expected_zero.append(String("hs.0.sq.0.weight"))
    expected_zero.append(String("hs.0.sq.0.bias"))
    expected_zero.append(String("hs.0.sk.0.weight"))
    expected_zero.append(String("hs.0.sk.0.bias"))
    expected_zero.append(String("hs.0.sv.0.weight"))
    expected_zero.append(String("hs.0.sao.0.weight"))
    expected_zero.append(String("hs.0.n1.0.gamma"))

    var unexpected = String("")
    var n_unexpected = 0
    for i in range(len(gv.zero_names)):
        var found = False
        for j in range(len(expected_zero)):
            if gv.zero_names[i] == expected_zero[j]:
                found = True
        if not found:
            n_unexpected += 1
            if unexpected == "":
                unexpected = gv.zero_names[i]
    check(
        fails,
        "gradient reaches every parameter except the 7 structurally-inert ones",
        n_unexpected == 0
        and len(gv.zero_names) == len(expected_zero),
        String(len(gv.zero_names)) + " zero of " + String(len(gl.names))
        + (
            ", unexpected: " + unexpected if n_unexpected > 0 else " (as expected)"
        ),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act full-model gate failed")


from max.gpu.host import DeviceContext
from mojo_rl.nn.core.param import ParamVisitor


struct GradScan(ParamVisitor):
    """Counts parameters with a non-zero gradient and catches any NaN/Inf.

    "The backward ran" is not the check worth making — a graph that silently
    dropped a branch would also run. What matters is that gradient reaches
    nearly every parameter: a mis-wired node shows up as a whole subtree at
    exactly zero.
    """

    var nonzero: Int
    var saw_nan: Bool
    var bad_name: String
    var zero_names: List[String]

    def __init__(out self):
        self.nonzero = 0
        self.saw_nan = False
        self.bad_name = String("")
        self.zero_names = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.nonzero = move.nonzero
        self.saw_nan = move.saw_nan
        self.bad_name = move.bad_name^
        self.zero_names = move.zero_names^

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var any = False
        for i in range(N):
            var x = Float64(grad.data[i])
            if x != x or x > 1e30 or x < -1e30:
                if not self.saw_nan:
                    self.bad_name = String(name)
                self.saw_nan = True
            if x != 0.0:
                any = True
        if any:
            self.nonzero += 1
        else:
            self.zero_names.append(String(name))
