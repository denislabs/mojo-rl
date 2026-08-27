# +--------------------------------------------------------------------------+ #
# | M2 gate — DETR layers vs the ACT reference
# +--------------------------------------------------------------------------+ #
"""Gates `deep_agents/act/layers.mojo` against `references/act-main`'s OWN
`TransformerEncoderLayer` / `TransformerDecoderLayer` — imported and run, not
reimplemented.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref
    pixi run mojo build -I . -o /tmp/t \\
        tests/deep_agents/act/test_act_layers_vs_reference.mojo && /tmp/t

Order matters here. The first check is a single bare `nn.Linear`, because every
composite below rides on one decision — torch stores `weight` as `[out, in]`
and this framework's `Linear` computes `x @ W` with `W` as `[in, out]`. Inside
a square `[DIM, DIM]` projection a missed transpose produces a plausible wrong
number and no shape error, so it is gated alone before anything trusts it.

Weights are loaded BY NAME (`refload.LoadRefParams`) and full coverage is
asserted: a param the dump does not name would keep its random init and show up
as "close but not equal", which is the same signature as a real numerical
disagreement.

Dropout is off on both sides (reference `.eval()`, ours
`set_attr["training"](0)`) — the layers are compared as functions.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.deep_agents.act.layers import (
    DETRDecoderLayer,
    DETREncoderLayer,
    DETREncoderLayerMasked,
)
from mojo_rl.deep_agents.act.refload import ListParams, LoadRefParams, RefDump


comptime REF_DIR = "/tmp/act_ref"

# Shapes must match `dump_act_reference.py:section_layers`.
comptime B = 2
comptime DIM = 16
comptime HEADS = 4
comptime FF = 32
comptime SEQ = 6
comptime QL = 5
comptime KL = 7
comptime P = 0.1

comptime TOL = 3e-6


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


def fill(mut t: Tensor, ref src: List[Scalar[DT]], n: Int):
    t.ensure(n)
    for i in range(n):
        t.data[i] = src[i]


def report_load(
    mut fails: Int, tag: String, ref loader: LoadRefParams, n_params: Int
):
    """Every param of the module must have come from the dump."""
    var ok = len(loader.missing) == 0 and len(loader.loaded) == n_params
    var detail = String(len(loader.loaded)) + "/" + String(n_params) + " loaded"
    if len(loader.missing) > 0:
        detail += ", first missing: " + loader.missing[0]
    check(fails, tag + ": every param loaded from the dump", ok, detail)


def main() raises:
    var fails = 0
    print("DETR layer gate (reference: " + String(REF_DIR) + ")")
    print("")

    # ── 0. the transpose convention, on one bare Linear ──────────────────
    # torch `weight [out, in]`, `y = x @ Wᵀ`.  ours `W [in, out]`, `y = x @ W`.
    comptime LIN_IN = 5
    comptime LIN_OUT = 3
    comptime LIN_B = 4
    var lin = Linear[LIN_IN, LIN_OUT].make["cpu", Kaiming]()
    var lin_loader = LoadRefParams["lin."](RefDump(String(REF_DIR)))
    lin.for_each_param["cpu"](lin_loader, None, String(""))
    report_load(fails, "linear", lin_loader, 2)

    var lp = TensorPack[1]()
    var lin_x = RefDump(String(REF_DIR)).get(String("lin_x"))
    fill(lp[0], lin_x, LIN_B * LIN_IN)
    var lin_out = Tensor()
    lin.forward["cpu", LIN_B](
        TensorRefs[1, MutAnyOrigin](lp[0]), lin_out
    )
    var lin_ref = RefDump(String(REF_DIR)).get(String("lin_out"))
    check(
        fails,
        "nn.Linear weight layout ([out,in] -> [in,out])",
        worst(lin_out, lin_ref, LIN_B * LIN_OUT) < TOL,
        "max|diff| = " + String(worst(lin_out, lin_ref, LIN_B * LIN_OUT)),
    )

    # ── 1. encoder layer ─────────────────────────────────────────────────
    comptime SN = B * SEQ * DIM
    var enc = DETREncoderLayer[DIM, HEADS, SEQ, FF, P].make["cpu", Kaiming]()
    enc.set_attr["training"](Scalar[DT](0.0))
    var enc_loader = LoadRefParams["enc."](RefDump(String(REF_DIR)))
    enc.for_each_param["cpu"](enc_loader, None, String(""))
    var enc_list = ListParams()
    enc.for_each_param["cpu"](enc_list, None, String(""))
    report_load(fails, "encoder", enc_loader, len(enc_list.names))

    var d = RefDump(String(REF_DIR))
    var ep = TensorPack[2]()
    fill(ep[0], d.get(String("enc_x")), SN)
    fill(ep[1], d.get(String("enc_pos")), SN)
    var eout = Tensor()
    enc.forward["cpu", B](
        TensorRefs[2, MutAnyOrigin](ep[0], ep[1]), eout
    )
    var eref = d.get(String("enc_out"))
    check(
        fails,
        "DETREncoderLayer (post-LN, pos on q/k only, ReLU FFN)",
        worst(eout, eref, SN) < TOL,
        "max|diff| = " + String(worst(eout, eref, SN)),
    )

    # A pre-LN layer, or one that added pos to v, would still be "close" on a
    # random input. Show the comparison has resolving power: perturb pos and
    # confirm the output moves far more than the tolerance.
    for i in range(SN):
        ep[1].data[i] += Scalar[DT](0.5)
    var eout2 = Tensor()
    enc.forward["cpu", B](
        TensorRefs[2, MutAnyOrigin](ep[0], ep[1]), eout2
    )
    var moved = Float64(0.0)
    for i in range(SN):
        moved = max(moved, abs(Float64(eout2.data[i]) - Float64(eout.data[i])))
    check(
        fails,
        "the layer is actually sensitive to pos",
        moved > 0.05,
        "max|Delta out| for +0.5 on pos = " + String(moved),
    )

    # ── 2. masked encoder layer ──────────────────────────────────────────
    var encm = DETREncoderLayerMasked[DIM, HEADS, SEQ, FF, P].make[
        "cpu", Kaiming
    ]()
    encm.set_attr["training"](Scalar[DT](0.0))
    var encm_loader = LoadRefParams["encm."](RefDump(String(REF_DIR)))
    encm.for_each_param["cpu"](encm_loader, None, String(""))
    var encm_list = ListParams()
    encm.for_each_param["cpu"](encm_list, None, String(""))
    report_load(fails, "masked encoder", encm_loader, len(encm_list.names))

    var mp = TensorPack[2]()
    fill(mp[0], d.get(String("encm_x")), SN)
    # c = [pos | key_valid]
    var mpos = d.get(String("encm_pos"))
    var mval = d.get(String("encm_valid"))
    mp[1].ensure(B * (SEQ * DIM + SEQ))
    for b in range(B):
        for i in range(SEQ * DIM):
            mp[1].data[b * (SEQ * DIM + SEQ) + i] = mpos[b * SEQ * DIM + i]
        for i in range(SEQ):
            mp[1].data[b * (SEQ * DIM + SEQ) + SEQ * DIM + i] = mval[
                b * SEQ + i
            ]
    var mout = Tensor()
    encm.forward["cpu", B](
        TensorRefs[2, MutAnyOrigin](mp[0], mp[1]), mout
    )
    var mref = d.get(String("encm_out"))
    check(
        fails,
        "DETREncoderLayerMasked (valid 4/6 and 2/6 per sample)",
        worst(mout, mref, SN) < TOL,
        "max|diff| = " + String(worst(mout, mref, SN)),
    )

    # The mask must matter — rerun with everything valid and require a
    # difference, or the check above is passing on an unmasked computation.
    for b in range(B):
        for i in range(SEQ):
            mp[1].data[b * (SEQ * DIM + SEQ) + SEQ * DIM + i] = Scalar[DT](1.0)
    var mout_all = Tensor()
    encm.forward["cpu", B](
        TensorRefs[2, MutAnyOrigin](mp[0], mp[1]), mout_all
    )
    var mask_effect = Float64(0.0)
    for i in range(SN):
        mask_effect = max(
            mask_effect,
            abs(Float64(mout_all.data[i]) - Float64(mout.data[i])),
        )
    check(
        fails,
        "the key padding mask changes the layer output",
        mask_effect > 0.05,
        "max|all-valid - masked| = " + String(mask_effect),
    )

    # ── 3. decoder layer ─────────────────────────────────────────────────
    comptime QN = B * QL * DIM
    comptime KN = B * KL * DIM
    var dec = DETRDecoderLayer[DIM, HEADS, QL, KL, FF, P].make["cpu", Kaiming]()
    dec.set_attr["training"](Scalar[DT](0.0))
    var dec_loader = LoadRefParams["dec."](RefDump(String(REF_DIR)))
    dec.for_each_param["cpu"](dec_loader, None, String(""))
    var dec_list = ListParams()
    dec.for_each_param["cpu"](dec_list, None, String(""))
    report_load(fails, "decoder", dec_loader, len(dec_list.names))

    var tgt = d.get(String("dec_tgt"))
    var mem = d.get(String("dec_mem"))
    var mpos2 = d.get(String("dec_mpos"))
    var qpos = d.get(String("dec_qpos"))

    var dp = TensorPack[2]()
    fill(dp[0], tgt, QN)
    # c = [query_pos | k_mem = memory + pos | memory]
    comptime CN = QL * DIM + 2 * KL * DIM
    dp[1].ensure(B * CN)
    for b in range(B):
        var base = b * CN
        for i in range(QL * DIM):
            dp[1].data[base + i] = qpos[b * QL * DIM + i]
        for i in range(KL * DIM):
            dp[1].data[base + QL * DIM + i] = (
                mem[b * KL * DIM + i] + mpos2[b * KL * DIM + i]
            )
            dp[1].data[base + QL * DIM + KL * DIM + i] = mem[b * KL * DIM + i]
    var dout = Tensor()
    dec.forward["cpu", B](
        TensorRefs[2, MutAnyOrigin](dp[0], dp[1]), dout
    )
    var dref = d.get(String("dec_out"))
    check(
        fails,
        "DETRDecoderLayer (self-attn + cross-attn, query_pos used 3x)",
        worst(dout, dref, QN) < TOL,
        "max|diff| = " + String(worst(dout, dref, QN)),
    )

    # `query_pos` reaches the cross-attention query as well as the self-
    # attention q/k. Dropping that third use breaks no shape and degrades the
    # model quietly, so check the output depends on it beyond the self-attn.
    for i in range(B * CN):
        pass
    var dp2 = TensorPack[2]()
    fill(dp2[0], tgt, QN)
    dp2[1].ensure(B * CN)
    for i in range(B * CN):
        dp2[1].data[i] = dp[1].data[i]
    for b in range(B):
        for i in range(QL * DIM):
            dp2[1].data[b * CN + i] += Scalar[DT](0.5)  # perturb query_pos
    var dout2 = Tensor()
    dec.forward["cpu", B](
        TensorRefs[2, MutAnyOrigin](dp2[0], dp2[1]), dout2
    )
    var qmoved = Float64(0.0)
    for i in range(QN):
        qmoved = max(
            qmoved, abs(Float64(dout2.data[i]) - Float64(dout.data[i]))
        )
    check(
        fails,
        "the decoder is sensitive to query_pos",
        qmoved > 0.05,
        "max|Delta out| for +0.5 on query_pos = " + String(qmoved),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act layer gate failed")
