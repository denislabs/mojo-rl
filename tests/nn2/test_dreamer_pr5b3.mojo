"""PR-5b part 2 parity: RSSM `_core` (GRU) + `_prior` backward vs jax.vjp.

The hard one — the BlockLinear GRU + group-interleave backward. Params +
inputs from `pr4_fixture.txt`; gradient ground truth from `pr5b2_fixture.txt`
(jax.vjp of the actual reference `_core`/`_prior`). All ≤1e-4.

Run: `pixi run mojo run -I . tests/nn2/test_dreamer_pr5b3.mojo`
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.rssm import RSSM
from mojo_rl.deep_agents2.dreamerv3.onehot_kl import OneHotKL

comptime PR4 = "tests/nn2/dreamerv3/fixtures/pr4_fixture.txt"
comptime PR5B2 = "tests/nn2/dreamerv3/fixtures/pr5b2_fixture.txt"
comptime B = 2
comptime DETER = 16
comptime HIDDEN = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 2
comptime TOKEN = 8
comptime SC = STOCH * CLASSES


def _split_lines(content: String) raises -> List[String]:
    var lines = List[String]()
    var current = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(current)
            current = String("")
        else:
            current += chr(Int(c))
    if current.byte_length() > 0:
        lines.append(current)
    return lines^


def _read_flat(lines: List[String], name: String) raises -> List[Scalar[DT]]:
    var pfx = name + "#size="
    for i in range(len(lines)):
        if lines[i].startswith(pfx):
            var n = atol(String(lines[i][byte=pfx.byte_length():]))
            var out = List[Scalar[DT]]()
            for k in range(n):
                out.append(Scalar[DT](atof(lines[i + 1 + k])))
            return out^
    raise Error("fixture: section not found: " + name)


def _buf(src: List[Scalar[DT]]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](len(src))
    for i in range(len(src)):
        p[i] = src[i]
    return p


def _load(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lines: List[String],
    name: String,
) raises:
    var v = _read_flat(lines, name)
    for i in range(len(v)):
        ptr[i] = v[i]


def _diff(
    got: UnsafePointer[Scalar[DT], MutAnyOrigin], expected: List[Scalar[DT]]
) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(len(expected)):
        var d = got[i] - expected[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


def _load_rssm(
    mut m: RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN],
    pl: List[String],
) raises:
    _load(m.dynin0.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin0/kernel")
    _load(m.dynin0.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin0/bias")
    _load(m.dynin0n.gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin0norm/scale")
    _load(m.dynin1.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin1/kernel")
    _load(m.dynin1.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin1/bias")
    _load(m.dynin1n.gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin1norm/scale")
    _load(m.dynin2.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin2/kernel")
    _load(m.dynin2.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin2/bias")
    _load(m.dynin2n.gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin2norm/scale")
    _load(m.dynhid0.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynhid0/kernel")
    _load(m.dynhid0.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynhid0/bias")
    _load(m.dynhid0n.gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynhid0norm/scale")
    _load(m.dyngru.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dyngru/kernel")
    _load(m.dyngru.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dyngru/bias")
    _load(m.prior0.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/prior0/kernel")
    _load(m.prior0.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/prior0/bias")
    _load(m.prior0n.gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/prior0norm/scale")
    _load(m.prior1.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/prior1/kernel")
    _load(m.prior1.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/prior1/bias")
    _load(m.prior1n.gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/prior1norm/scale")
    _load(m.priorlogit.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/priorlogit/kernel")
    _load(m.priorlogit.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/priorlogit/bias")
    _load(m.obs0.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/obs0/kernel")
    _load(m.obs0.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/obs0/bias")
    _load(m.obs0n.gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/obs0norm/scale")
    _load(m.obslogit.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/obslogit/kernel")
    _load(m.obslogit.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/obslogit/bias")


def _load_rssm_wm(
    mut m: RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN],
    gl: List[String],
) raises:
    """Load RSSM params from the wm-state dump (`pwm.*` in pr5b2) — these
    differ from pr4's per-module inits (single nj context RNG order)."""
    _load(m.dynin0.weight.value_unsafe_ptr_cpu(), gl, "pwm.dynin0/kernel")
    _load(m.dynin0.bias.value_unsafe_ptr_cpu(), gl, "pwm.dynin0/bias")
    _load(m.dynin0n.gamma.value_unsafe_ptr_cpu(), gl, "pwm.dynin0norm/scale")
    _load(m.dynin1.weight.value_unsafe_ptr_cpu(), gl, "pwm.dynin1/kernel")
    _load(m.dynin1.bias.value_unsafe_ptr_cpu(), gl, "pwm.dynin1/bias")
    _load(m.dynin1n.gamma.value_unsafe_ptr_cpu(), gl, "pwm.dynin1norm/scale")
    _load(m.dynin2.weight.value_unsafe_ptr_cpu(), gl, "pwm.dynin2/kernel")
    _load(m.dynin2.bias.value_unsafe_ptr_cpu(), gl, "pwm.dynin2/bias")
    _load(m.dynin2n.gamma.value_unsafe_ptr_cpu(), gl, "pwm.dynin2norm/scale")
    _load(m.dynhid0.weight.value_unsafe_ptr_cpu(), gl, "pwm.dynhid0/kernel")
    _load(m.dynhid0.bias.value_unsafe_ptr_cpu(), gl, "pwm.dynhid0/bias")
    _load(m.dynhid0n.gamma.value_unsafe_ptr_cpu(), gl, "pwm.dynhid0norm/scale")
    _load(m.dyngru.weight.value_unsafe_ptr_cpu(), gl, "pwm.dyngru/kernel")
    _load(m.dyngru.bias.value_unsafe_ptr_cpu(), gl, "pwm.dyngru/bias")
    _load(m.prior0.weight.value_unsafe_ptr_cpu(), gl, "pwm.prior0/kernel")
    _load(m.prior0.bias.value_unsafe_ptr_cpu(), gl, "pwm.prior0/bias")
    _load(m.prior0n.gamma.value_unsafe_ptr_cpu(), gl, "pwm.prior0norm/scale")
    _load(m.prior1.weight.value_unsafe_ptr_cpu(), gl, "pwm.prior1/kernel")
    _load(m.prior1.bias.value_unsafe_ptr_cpu(), gl, "pwm.prior1/bias")
    _load(m.prior1n.gamma.value_unsafe_ptr_cpu(), gl, "pwm.prior1norm/scale")
    _load(m.priorlogit.weight.value_unsafe_ptr_cpu(), gl, "pwm.priorlogit/kernel")
    _load(m.priorlogit.bias.value_unsafe_ptr_cpu(), gl, "pwm.priorlogit/bias")
    _load(m.obs0.weight.value_unsafe_ptr_cpu(), gl, "pwm.obs0/kernel")
    _load(m.obs0.bias.value_unsafe_ptr_cpu(), gl, "pwm.obs0/bias")
    _load(m.obs0n.gamma.value_unsafe_ptr_cpu(), gl, "pwm.obs0norm/scale")
    _load(m.obslogit.weight.value_unsafe_ptr_cpu(), gl, "pwm.obslogit/kernel")
    _load(m.obslogit.bias.value_unsafe_ptr_cpu(), gl, "pwm.obslogit/bias")


def test_core_vjp() raises:
    print("test_core_vjp (BlockLinear GRU + group interleave) ...")
    var p4: String
    with open(PR4, "r") as f:
        p4 = String(f.read())
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var pl = _split_lines(p4)
    var gl = _split_lines(g5)

    var m = RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make["cpu"]()
    _load_rssm(m, pl)

    var deter = _buf(_read_flat(pl, "in.deter0"))
    var stoch = _buf(_read_flat(pl, "in.stoch0"))
    var action = _buf(_read_flat(pl, "in.action"))
    var g_out = _buf(_read_flat(gl, "core.g_out"))
    var g_deter = alloc[Scalar[DT]](B * DETER)
    var g_stoch = alloc[Scalar[DT]](B * SC)
    var g_action = alloc[Scalar[DT]](B * ACT)
    m.core_vjp[B](deter, stoch, action, g_out, g_deter, g_stoch, g_action)

    var dd = _diff(g_deter, _read_flat(gl, "core.g_deter"))
    var ds = _diff(g_stoch, _read_flat(gl, "core.g_stoch"))
    var da = _diff(g_action, _read_flat(gl, "core.g_action"))
    print("  grad inputs: deter", dd, " stoch", ds, " action", da)
    assert_true(dd < Scalar[DT](1e-4), "core grad_deter")
    assert_true(ds < Scalar[DT](1e-4), "core grad_stoch")
    assert_true(da < Scalar[DT](1e-4), "core grad_action")

    var dk_in0 = _diff(m.dynin0.weight.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dynin0/kernel"))
    var dk_in1 = _diff(m.dynin1.weight.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dynin1/kernel"))
    var dk_hid = _diff(m.dynhid0.weight.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dynhid0/kernel"))
    var dk_gru = _diff(m.dyngru.weight.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dyngru/kernel"))
    var db_gru = _diff(m.dyngru.bias.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dyngru/bias"))
    var ds_hid = _diff(m.dynhid0n.gamma.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dynhid0norm/scale"))
    print("  param grads: dynin0.k", dk_in0, " dynin1.k", dk_in1,
          " dynhid0.k", dk_hid, " dyngru.k", dk_gru, " dyngru.b", db_gru,
          " dynhid0n.s", ds_hid)
    assert_true(dk_in0 < Scalar[DT](1e-4), "core grad dynin0.kernel")
    assert_true(dk_in1 < Scalar[DT](1e-4), "core grad dynin1.kernel")
    assert_true(dk_hid < Scalar[DT](1e-4), "core grad dynhid0.kernel")
    assert_true(dk_gru < Scalar[DT](1e-4), "core grad dyngru.kernel")
    assert_true(db_gru < Scalar[DT](1e-4), "core grad dyngru.bias")
    assert_true(ds_hid < Scalar[DT](1e-4), "core grad dynhid0n.scale")
    print("  ok")
    _ = deter; _ = stoch; _ = action


def test_prior_vjp() raises:
    print("test_prior_vjp ...")
    var p4: String
    with open(PR4, "r") as f:
        p4 = String(f.read())
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var pl = _split_lines(p4)
    var gl = _split_lines(g5)

    var m = RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make["cpu"]()
    _load_rssm(m, pl)

    var deter = _buf(_read_flat(pl, "in.deter0"))
    var g_logit = _buf(_read_flat(gl, "prior.g_out"))
    var g_deter = alloc[Scalar[DT]](B * DETER)
    m.prior_vjp[B](deter, g_logit, g_deter)

    var dd = _diff(g_deter, _read_flat(gl, "prior.g_deter"))
    var dk0 = _diff(m.prior0.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gprior.prior0/kernel"))
    var dk1 = _diff(m.prior1.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gprior.prior1/kernel"))
    var dkl = _diff(m.priorlogit.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gprior.priorlogit/kernel"))
    print("  grad_deter", dd, " prior0.k", dk0, " prior1.k", dk1,
          " priorlogit.k", dkl)
    assert_true(dd < Scalar[DT](1e-4), "prior grad_deter")
    assert_true(dk0 < Scalar[DT](1e-4), "prior grad prior0.kernel")
    assert_true(dk1 < Scalar[DT](1e-4), "prior grad prior1.kernel")
    assert_true(dkl < Scalar[DT](1e-4), "prior grad priorlogit.kernel")
    print("  ok")
    _ = deter


def test_kl_backward_in_context() raises:
    print("test_kl_backward_in_context (isolate OneHotKL grad_post/prior) ...")
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var gl = _split_lines(g5)
    var post = _buf(_read_flat(gl, "wm.post"))
    var prior = _buf(_read_flat(gl, "wm.prior"))
    var kl = OneHotKL[STOCH, CLASSES].make(0.01, 1.0)
    var dyn = alloc[Scalar[DT]](B)
    var rep = alloc[Scalar[DT]](B)
    var dyn2: UnsafePointer[Scalar[DT], MutAnyOrigin] = dyn
    var rep2: UnsafePointer[Scalar[DT], MutAnyOrigin] = rep
    kl.forward[B](post, prior, dyn2, rep2)
    var d_dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var d_rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    for i in range(B):
        d_dyn[i] = 1.0
        d_rep[i] = 1.0
    var g_post: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    var g_prior: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    kl.backward[B](d_dyn, d_rep, g_post, g_prior)
    var dp = _diff(g_post, _read_flat(gl, "wm.g_post"))
    var dpr = _diff(g_prior, _read_flat(gl, "wm.g_prior"))
    print("  g_post diff", dp, " g_prior diff", dpr)
    assert_true(dp < Scalar[DT](1e-4), "kl grad_post in context")
    assert_true(dpr < Scalar[DT](1e-4), "kl grad_prior in context")
    print("  ok")


def test_observe_vjp_isolated() raises:
    print("test_observe_vjp_isolated (grad_tokens from grad_post only) ...")
    var p4: String
    with open(PR4, "r") as f:
        p4 = String(f.read())
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var pl = _split_lines(p4)
    var gl = _split_lines(g5)
    var m = RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make["cpu"]()
    _load_rssm(m, pl)
    var deter = _buf(_read_flat(pl, "in.deter0"))
    var stoch = _buf(_read_flat(pl, "in.stoch0"))
    var action = _buf(_read_flat(pl, "in.action"))
    var tokens = _buf(_read_flat(pl, "in.tokens"))
    var g_post = _buf(_read_flat(gl, "wm.g_post"))   # validated cotangent
    var zero_nd: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        B * DETER
    )
    for i in range(B * DETER):
        zero_nd[i] = 0.0
    var g_deter = alloc[Scalar[DT]](B * DETER)
    var g_stoch = alloc[Scalar[DT]](B * SC)
    var g_action = alloc[Scalar[DT]](B * ACT)
    var g_tokens = alloc[Scalar[DT]](B * TOKEN)
    m.observe_vjp[B](deter, stoch, action, tokens, g_post, zero_nd,
                     g_deter, g_stoch, g_action, g_tokens)
    var dt = _diff(g_tokens, _read_flat(gl, "wm.g_tokens"))
    print("  grad_tokens diff (obs path only) =", dt)
    assert_true(dt < Scalar[DT](1e-4), "observe_vjp grad_tokens")
    print("  ok")
    _ = deter; _ = stoch; _ = action; _ = tokens


def test_wm_loss_vjp() raises:
    print("test_wm_loss_vjp (full dyn/rep backward: KL+observe+prior+core) ...")
    var p4: String
    with open(PR4, "r") as f:
        p4 = String(f.read())
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var pl = _split_lines(p4)
    var gl = _split_lines(g5)

    var m = RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make["cpu"](
        unimix=0.01, free_nats=1.0
    )
    _load_rssm_wm(m, gl)   # wm-state params (prior differs from pr4)

    var deter = _buf(_read_flat(pl, "in.deter0"))
    var stoch = _buf(_read_flat(pl, "in.stoch0"))
    var action = _buf(_read_flat(pl, "in.action"))
    var tokens = _buf(_read_flat(pl, "in.tokens"))
    var d_dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var d_rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    for i in range(B):
        d_dyn[i] = 1.0
        d_rep[i] = 1.0
    var g_deter = alloc[Scalar[DT]](B * DETER)
    var g_stoch = alloc[Scalar[DT]](B * SC)
    var g_action = alloc[Scalar[DT]](B * ACT)
    var g_tokens = alloc[Scalar[DT]](B * TOKEN)
    m.loss_vjp[B](deter, stoch, action, tokens, d_dyn, d_rep,
                  g_deter, g_stoch, g_action, g_tokens)

    var dd = _diff(g_deter, _read_flat(gl, "wm.g_deter"))
    var ds = _diff(g_stoch, _read_flat(gl, "wm.g_stoch"))
    var da = _diff(g_action, _read_flat(gl, "wm.g_action"))
    var dt = _diff(g_tokens, _read_flat(gl, "wm.g_tokens"))
    print("  grad inputs: deter", dd, " stoch", ds, " action", da,
          " tokens", dt)
    assert_true(dd < Scalar[DT](1e-4), "wm grad_deter")
    assert_true(ds < Scalar[DT](1e-4), "wm grad_stoch")
    assert_true(da < Scalar[DT](1e-4), "wm grad_action")
    assert_true(dt < Scalar[DT](1e-4), "wm grad_tokens")

    # param grads across all three paths (core / obs / prior)
    var d_in0 = _diff(m.dynin0.weight.grad_unsafe_ptr_cpu(),
                      _read_flat(gl, "gwm.dynin0/kernel"))
    var d_gru = _diff(m.dyngru.weight.grad_unsafe_ptr_cpu(),
                      _read_flat(gl, "gwm.dyngru/kernel"))
    var d_obs = _diff(m.obs0.weight.grad_unsafe_ptr_cpu(),
                      _read_flat(gl, "gwm.obs0/kernel"))
    var d_obl = _diff(m.obslogit.weight.grad_unsafe_ptr_cpu(),
                      _read_flat(gl, "gwm.obslogit/kernel"))
    var d_pr0 = _diff(m.prior0.weight.grad_unsafe_ptr_cpu(),
                      _read_flat(gl, "gwm.prior0/kernel"))
    var d_prl = _diff(m.priorlogit.weight.grad_unsafe_ptr_cpu(),
                      _read_flat(gl, "gwm.priorlogit/kernel"))
    print("  params: dynin0", d_in0, " dyngru", d_gru, " obs0", d_obs,
          " obslogit", d_obl, " prior0", d_pr0, " priorlogit", d_prl)
    assert_true(d_in0 < Scalar[DT](1e-4), "wm grad dynin0.kernel")
    assert_true(d_gru < Scalar[DT](1e-4), "wm grad dyngru.kernel")
    assert_true(d_obs < Scalar[DT](1e-4), "wm grad obs0.kernel")
    assert_true(d_obl < Scalar[DT](1e-4), "wm grad obslogit.kernel")
    assert_true(d_pr0 < Scalar[DT](1e-4), "wm grad prior0.kernel")
    assert_true(d_prl < Scalar[DT](1e-4), "wm grad priorlogit.kernel")
    print("  ok")
    _ = deter; _ = stoch; _ = action; _ = tokens


def main() raises:
    print("=" * 70)
    print("PR-5b2 RSSM _core/_prior backward parity (vs jax.vjp)")
    print("=" * 70)
    test_core_vjp()
    test_prior_vjp()
    test_kl_backward_in_context()
    test_observe_vjp_isolated()
    test_wm_loss_vjp()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
