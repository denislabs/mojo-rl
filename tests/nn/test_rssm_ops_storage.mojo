"""Storage-surface gate for the RSSM custom leaf ops (rssm_ops.mojo port).

For EACH of the 4 ops (ActionSquash / BlockGroupAssemble / GRUGate /
StraightThroughSample):

  - CPU finite-difference of the vjp against L = Σ(w·out) for fixed-ish w
    (exact-gradient ops → tol ~1e-2). EXCEPTION: StraightThroughSample's
    forward (one-hot argmax) is piecewise-constant in z, so FD sees no
    gradient — instead its CPU vjp is checked against the hand-coded
    reference grad_z = (1-u)·sm·(go − Σ go·sm).
  - CPU vs GPU parity for forward AND vjp: max abs diff < 1e-4.

Multi-arity inputs/grad_inputs live in a storage `TensorPack[N]` so that all
elements share the ONE wildcard origin `TensorRefs[N]` requires (independent
`var Tensor`s have distinct origins and cannot be packed — the §B0 rule).

Run: rm -f mojo_rl.mojoc && \
     pixi run -e apple mojo run -I . tests/nn/test_rssm_ops_storage.mojo
"""

from std.math import exp
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.deep_agents.dreamerv3.rssm_ops import (
    ActionSquash,
    BlockGroupAssemble,
    GRUGate,
    StraightThroughSample,
)


# ── small dims ──────────────────────────────────────────────────────────
comptime ACT = 3
comptime DETER = 8
comptime BLOCKS = 2
comptime H = 3
comptime STOCH = 3
comptime CLASSES = 4
comptime B = 5


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


# fixed-ish "random" weight for the scalar loss L = Σ w_i · out_i
def _w(i: Int) -> Scalar[DT]:
    return Scalar[DT](((i * 7 + 3) % 11) - 5) * 0.13


# pseudo-random fill for inputs
def _xfill(i: Int, salt: Int) -> Scalar[DT]:
    return Scalar[DT](((i * 13 + salt * 5 + 1) % 17) - 8) * 0.11


# ════════════════════════════════════════════════════════════════════════
# ActionSquash — arity 1, IN=OUT=ACT.  FD check + CPU/GPU parity.
# ════════════════════════════════════════════════════════════════════════
def _action_squash_cpu_fd() raises -> Bool:
    comptime N = B * ACT
    var op = ActionSquash[ACT]()

    var inp = TensorPack[1]()
    inp[0].ensure(N)
    for i in range(N):
        inp[0].data[i] = _xfill(i, 1)

    var out = Tensor.alloc(N)
    op.forward["cpu", B](TensorRefs[1](inp[0]), out, None)

    var go = Tensor.alloc(N)
    for i in range(N):
        go.data[i] = _w(i)
    var gi = TensorPack[1]()
    op.vjp["cpu", B](TensorRefs[1](inp[0]), go, TensorRefs[1](gi[0]), None)

    var h = Scalar[DT](1e-3)
    for j in range(N):
        var pk = TensorPack[1]()
        var mk = TensorPack[1]()
        pk[0].ensure(N)
        mk[0].ensure(N)
        for i in range(N):
            pk[0].data[i] = inp[0].data[i]
            mk[0].data[i] = inp[0].data[i]
        pk[0].data[j] += h
        mk[0].data[j] -= h
        var op2 = ActionSquash[ACT]()
        var outp = Tensor.alloc(N)
        var outm = Tensor.alloc(N)
        op2.forward["cpu", B](TensorRefs[1](pk[0]), outp, None)
        op2.forward["cpu", B](TensorRefs[1](mk[0]), outm, None)
        var lp = Scalar[DT](0.0)
        var lm = Scalar[DT](0.0)
        for i in range(N):
            lp += _w(i) * outp.data[i]
            lm += _w(i) * outm.data[i]
        var fd = (lp - lm) / (Scalar[DT](2.0) * h)
        if _abs(fd - gi[0].data[j]) > Scalar[DT](1e-2):
            print("    ActionSquash FD mismatch j=", j, " fd=", fd,
                  " an=", gi[0].data[j])
            return False
    return True


def _action_squash_parity(ctx: DeviceContext) raises -> Bool:
    comptime N = B * ACT
    var goc = Tensor.alloc(N)
    var inc = TensorPack[1]()
    inc[0].ensure(N)
    for i in range(N):
        inc[0].data[i] = _xfill(i, 1)
        goc.data[i] = _w(i)

    var op_c = ActionSquash[ACT]()
    var out_c = Tensor.alloc(N)
    var gi_c = TensorPack[1]()
    op_c.forward["cpu", B](TensorRefs[1](inc[0]), out_c, None)
    op_c.vjp["cpu", B](TensorRefs[1](inc[0]), goc, TensorRefs[1](gi_c[0]), None)

    var ing = TensorPack[1]()
    ing[0].ensure(N)
    var gog = Tensor.alloc(N)
    for i in range(N):
        ing[0].data[i] = inc[0].data[i]
        gog.data[i] = goc.data[i]
    ing[0].upload(ctx)
    gog.upload(ctx)
    var op_g = ActionSquash[ACT]()
    var out_g = Tensor.alloc(N)
    var gi_g = TensorPack[1]()
    op_g.forward["gpu", B](TensorRefs[1](ing[0]), out_g, Optional(ctx))
    op_g.vjp["gpu", B](
        TensorRefs[1](ing[0]), gog, TensorRefs[1](gi_g[0]), Optional(ctx)
    )
    out_g.download(ctx)
    gi_g[0].download(ctx)

    for i in range(N):
        if _abs(out_c.data[i] - out_g.data[i]) > Scalar[DT](1e-4):
            print("    ActionSquash fwd parity i=", i)
            return False
        if _abs(gi_c[0].data[i] - gi_g[0].data[i]) > Scalar[DT](1e-4):
            print("    ActionSquash vjp parity i=", i)
            return False
    return True


# ════════════════════════════════════════════════════════════════════════
# BlockGroupAssemble — arity 4 (deter[DETER], x0/x1/x2[H]). FD + parity.
# ════════════════════════════════════════════════════════════════════════
comptime BGA_OUT = BlockGroupAssemble[DETER, H, BLOCKS].OUT_DIM


def _block_group_assemble_cpu_fd() raises -> Bool:
    var op = BlockGroupAssemble[DETER, H, BLOCKS]()

    var inp = TensorPack[4]()
    inp[0].ensure(B * DETER)
    inp[1].ensure(B * H)
    inp[2].ensure(B * H)
    inp[3].ensure(B * H)
    for i in range(B * DETER):
        inp[0].data[i] = _xfill(i, 2)
    for i in range(B * H):
        inp[1].data[i] = _xfill(i, 3)
        inp[2].data[i] = _xfill(i, 4)
        inp[3].data[i] = _xfill(i, 5)

    var out = Tensor.alloc(B * BGA_OUT)
    op.forward["cpu", B](
        TensorRefs[4](inp[0], inp[1], inp[2], inp[3]), out, None
    )

    var go = Tensor.alloc(B * BGA_OUT)
    for i in range(B * BGA_OUT):
        go.data[i] = _w(i)
    var gi = TensorPack[4]()
    op.vjp["cpu", B](
        TensorRefs[4](inp[0], inp[1], inp[2], inp[3]), go,
        TensorRefs[4](gi[0], gi[1], gi[2], gi[3]), None,
    )

    var h = Scalar[DT](1e-3)

    # FD each of the 4 inputs (which==0 deter[B·DETER], else x{which-1}[B·H]).
    for which in range(4):
        var n = (B * DETER) if which == 0 else (B * H)
        for j in range(n):
            var op2 = BlockGroupAssemble[DETER, H, BLOCKS]()
            var pk = TensorPack[4]()
            var mk = TensorPack[4]()
            pk[0].ensure(B * DETER); mk[0].ensure(B * DETER)
            pk[1].ensure(B * H); mk[1].ensure(B * H)
            pk[2].ensure(B * H); mk[2].ensure(B * H)
            pk[3].ensure(B * H); mk[3].ensure(B * H)
            for i in range(B * DETER):
                pk[0].data[i] = inp[0].data[i]; mk[0].data[i] = inp[0].data[i]
            for i in range(B * H):
                pk[1].data[i] = inp[1].data[i]; mk[1].data[i] = inp[1].data[i]
                pk[2].data[i] = inp[2].data[i]; mk[2].data[i] = inp[2].data[i]
                pk[3].data[i] = inp[3].data[i]; mk[3].data[i] = inp[3].data[i]
            pk[which].data[j] += h
            mk[which].data[j] -= h
            var outp = Tensor.alloc(B * BGA_OUT)
            var outm = Tensor.alloc(B * BGA_OUT)
            op2.forward["cpu", B](
                TensorRefs[4](pk[0], pk[1], pk[2], pk[3]), outp, None
            )
            op2.forward["cpu", B](
                TensorRefs[4](mk[0], mk[1], mk[2], mk[3]), outm, None
            )
            var lp = Scalar[DT](0.0)
            var lm = Scalar[DT](0.0)
            for i in range(B * BGA_OUT):
                lp += _w(i) * outp.data[i]
                lm += _w(i) * outm.data[i]
            var fd = (lp - lm) / (Scalar[DT](2.0) * h)
            if _abs(fd - gi[which].data[j]) > Scalar[DT](1e-2):
                print("    BGA FD mismatch which=", which, " j=", j,
                      " fd=", fd, " an=", gi[which].data[j])
                return False
    return True


def _block_group_assemble_parity(ctx: DeviceContext) raises -> Bool:
    var goc = Tensor.alloc(B * BGA_OUT)
    var inc = TensorPack[4]()
    inc[0].ensure(B * DETER)
    inc[1].ensure(B * H)
    inc[2].ensure(B * H)
    inc[3].ensure(B * H)
    for i in range(B * DETER):
        inc[0].data[i] = _xfill(i, 2)
    for i in range(B * H):
        inc[1].data[i] = _xfill(i, 3)
        inc[2].data[i] = _xfill(i, 4)
        inc[3].data[i] = _xfill(i, 5)
    for i in range(B * BGA_OUT):
        goc.data[i] = _w(i)

    var op_c = BlockGroupAssemble[DETER, H, BLOCKS]()
    var out_c = Tensor.alloc(B * BGA_OUT)
    op_c.forward["cpu", B](
        TensorRefs[4](inc[0], inc[1], inc[2], inc[3]), out_c, None
    )
    var gi_c = TensorPack[4]()
    op_c.vjp["cpu", B](
        TensorRefs[4](inc[0], inc[1], inc[2], inc[3]), goc,
        TensorRefs[4](gi_c[0], gi_c[1], gi_c[2], gi_c[3]), None,
    )

    var ing = TensorPack[4]()
    ing[0].ensure(B * DETER)
    ing[1].ensure(B * H)
    ing[2].ensure(B * H)
    ing[3].ensure(B * H)
    var gog = Tensor.alloc(B * BGA_OUT)
    for i in range(B * DETER):
        ing[0].data[i] = inc[0].data[i]
    for i in range(B * H):
        ing[1].data[i] = inc[1].data[i]
        ing[2].data[i] = inc[2].data[i]
        ing[3].data[i] = inc[3].data[i]
    for i in range(B * BGA_OUT):
        gog.data[i] = goc.data[i]
    ing[0].upload(ctx); ing[1].upload(ctx)
    ing[2].upload(ctx); ing[3].upload(ctx)
    gog.upload(ctx)

    var op_g = BlockGroupAssemble[DETER, H, BLOCKS]()
    var out_g = Tensor.alloc(B * BGA_OUT)
    op_g.forward["gpu", B](
        TensorRefs[4](ing[0], ing[1], ing[2], ing[3]), out_g, Optional(ctx)
    )
    var gi_g = TensorPack[4]()
    op_g.vjp["gpu", B](
        TensorRefs[4](ing[0], ing[1], ing[2], ing[3]), gog,
        TensorRefs[4](gi_g[0], gi_g[1], gi_g[2], gi_g[3]), Optional(ctx),
    )
    out_g.download(ctx)
    gi_g[0].download(ctx); gi_g[1].download(ctx)
    gi_g[2].download(ctx); gi_g[3].download(ctx)

    for i in range(B * BGA_OUT):
        if _abs(out_c.data[i] - out_g.data[i]) > Scalar[DT](1e-4):
            print("    BGA fwd parity i=", i)
            return False
    for i in range(B * DETER):
        if _abs(gi_c[0].data[i] - gi_g[0].data[i]) > Scalar[DT](1e-4):
            print("    BGA g_deter parity i=", i)
            return False
    for i in range(B * H):
        if _abs(gi_c[1].data[i] - gi_g[1].data[i]) > Scalar[DT](1e-4):
            return False
        if _abs(gi_c[2].data[i] - gi_g[2].data[i]) > Scalar[DT](1e-4):
            return False
        if _abs(gi_c[3].data[i] - gi_g[3].data[i]) > Scalar[DT](1e-4):
            return False
    return True


# ════════════════════════════════════════════════════════════════════════
# GRUGate — arity 2 (gru[3·DETER], deter[DETER]) → [DETER]. FD + parity.
# ════════════════════════════════════════════════════════════════════════
comptime GRU_DIM = GRUGate[DETER, BLOCKS].GRU_DIM


def _gru_gate_cpu_fd() raises -> Bool:
    var op = GRUGate[DETER, BLOCKS]()

    var inp = TensorPack[2]()
    inp[0].ensure(B * GRU_DIM)
    inp[1].ensure(B * DETER)
    for i in range(B * GRU_DIM):
        inp[0].data[i] = _xfill(i, 6)
    for i in range(B * DETER):
        inp[1].data[i] = _xfill(i, 7)

    var out = Tensor.alloc(B * DETER)
    op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out, None)

    var go = Tensor.alloc(B * DETER)
    for i in range(B * DETER):
        go.data[i] = _w(i)
    var gi = TensorPack[2]()
    op.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gi[0], gi[1]), None
    )

    var h = Scalar[DT](1e-3)

    # FD each of the 2 inputs (which==0 gru[B·GRU_DIM], 1 deter[B·DETER]).
    for which in range(2):
        var n = (B * GRU_DIM) if which == 0 else (B * DETER)
        for j in range(n):
            var op2 = GRUGate[DETER, BLOCKS]()
            var pk = TensorPack[2]()
            var mk = TensorPack[2]()
            pk[0].ensure(B * GRU_DIM); mk[0].ensure(B * GRU_DIM)
            pk[1].ensure(B * DETER); mk[1].ensure(B * DETER)
            for i in range(B * GRU_DIM):
                pk[0].data[i] = inp[0].data[i]; mk[0].data[i] = inp[0].data[i]
            for i in range(B * DETER):
                pk[1].data[i] = inp[1].data[i]; mk[1].data[i] = inp[1].data[i]
            pk[which].data[j] += h
            mk[which].data[j] -= h
            var outp = Tensor.alloc(B * DETER)
            var outm = Tensor.alloc(B * DETER)
            op2.forward["cpu", B](TensorRefs[2](pk[0], pk[1]), outp, None)
            op2.forward["cpu", B](TensorRefs[2](mk[0], mk[1]), outm, None)
            var lp = Scalar[DT](0.0)
            var lm = Scalar[DT](0.0)
            for i in range(B * DETER):
                lp += _w(i) * outp.data[i]
                lm += _w(i) * outm.data[i]
            var fd = (lp - lm) / (Scalar[DT](2.0) * h)
            if _abs(fd - gi[which].data[j]) > Scalar[DT](1e-2):
                print("    GRU FD mismatch which=", which, " j=", j,
                      " fd=", fd, " an=", gi[which].data[j])
                return False
    return True


def _gru_gate_parity(ctx: DeviceContext) raises -> Bool:
    var goc = Tensor.alloc(B * DETER)
    var inc = TensorPack[2]()
    inc[0].ensure(B * GRU_DIM)
    inc[1].ensure(B * DETER)
    for i in range(B * GRU_DIM):
        inc[0].data[i] = _xfill(i, 6)
    for i in range(B * DETER):
        inc[1].data[i] = _xfill(i, 7)
        goc.data[i] = _w(i)

    var op_c = GRUGate[DETER, BLOCKS]()
    var out_c = Tensor.alloc(B * DETER)
    op_c.forward["cpu", B](TensorRefs[2](inc[0], inc[1]), out_c, None)
    var gi_c = TensorPack[2]()
    op_c.vjp["cpu", B](
        TensorRefs[2](inc[0], inc[1]), goc, TensorRefs[2](gi_c[0], gi_c[1]), None
    )

    var ing = TensorPack[2]()
    ing[0].ensure(B * GRU_DIM)
    ing[1].ensure(B * DETER)
    var gog = Tensor.alloc(B * DETER)
    for i in range(B * GRU_DIM):
        ing[0].data[i] = inc[0].data[i]
    for i in range(B * DETER):
        ing[1].data[i] = inc[1].data[i]
        gog.data[i] = goc.data[i]
    ing[0].upload(ctx); ing[1].upload(ctx); gog.upload(ctx)

    var op_g = GRUGate[DETER, BLOCKS]()
    var out_g = Tensor.alloc(B * DETER)
    op_g.forward["gpu", B](TensorRefs[2](ing[0], ing[1]), out_g, Optional(ctx))
    var gi_g = TensorPack[2]()
    op_g.vjp["gpu", B](
        TensorRefs[2](ing[0], ing[1]), gog,
        TensorRefs[2](gi_g[0], gi_g[1]), Optional(ctx),
    )
    out_g.download(ctx)
    gi_g[0].download(ctx); gi_g[1].download(ctx)

    for i in range(B * DETER):
        if _abs(out_c.data[i] - out_g.data[i]) > Scalar[DT](1e-4):
            print("    GRU fwd parity i=", i)
            return False
        if _abs(gi_c[1].data[i] - gi_g[1].data[i]) > Scalar[DT](1e-4):
            print("    GRU g_deter parity i=", i)
            return False
    for i in range(B * GRU_DIM):
        if _abs(gi_c[0].data[i] - gi_g[0].data[i]) > Scalar[DT](1e-4):
            print("    GRU g_gru parity i=", i)
            return False
    return True


# ════════════════════════════════════════════════════════════════════════
# StraightThroughSample — arity 1, [STOCH·CLASSES].  vjp vs hand-coded ref
# grad_z = (1-u)·sm·(go − Σ go·sm) (per (b,s) group) + CPU/GPU parity.
# ════════════════════════════════════════════════════════════════════════
comptime SC = STOCH * CLASSES


def _st_sample_cpu_ref() raises -> Bool:
    var op = StraightThroughSample[STOCH, CLASSES]()
    var one_m_u = Scalar[DT](1.0) - op.unimix

    var inp = TensorPack[1]()
    inp[0].ensure(B * SC)
    for i in range(B * SC):
        inp[0].data[i] = _xfill(i, 8)

    var out = Tensor.alloc(B * SC)
    op.forward["cpu", B](TensorRefs[1](inp[0]), out, None)

    # forward output sanity: each (b,s) group is one-hot.
    for b in range(B):
        for s in range(STOCH):
            var base = (b * STOCH + s) * CLASSES
            var sum = Scalar[DT](0.0)
            for c in range(CLASSES):
                var v = out.data[base + c]
                if v != Scalar[DT](0.0) and v != Scalar[DT](1.0):
                    print("    ST fwd not 0/1")
                    return False
                sum += v
            if _abs(sum - Scalar[DT](1.0)) > Scalar[DT](1e-6):
                print("    ST fwd not one-hot")
                return False

    var go = Tensor.alloc(B * SC)
    for i in range(B * SC):
        go.data[i] = _w(i)
    var gi = TensorPack[1]()
    op.vjp["cpu", B](TensorRefs[1](inp[0]), go, TensorRefs[1](gi[0]), None)

    # hand-coded reference grad_z = (1-u)·sm·(go − Σ go·sm)
    for b in range(B):
        for s in range(STOCH):
            var base = (b * STOCH + s) * CLASSES
            var zmax = inp[0].data[base]
            for c in range(1, CLASSES):
                if inp[0].data[base + c] > zmax:
                    zmax = inp[0].data[base + c]
            var ssum = Scalar[DT](0.0)
            for c in range(CLASSES):
                ssum += exp(inp[0].data[base + c] - zmax)
            var dot = Scalar[DT](0.0)
            for c in range(CLASSES):
                var sm = exp(inp[0].data[base + c] - zmax) / ssum
                dot += go.data[base + c] * sm
            for c in range(CLASSES):
                var sm = exp(inp[0].data[base + c] - zmax) / ssum
                var ref_g = one_m_u * sm * (go.data[base + c] - dot)
                if _abs(ref_g - gi[0].data[base + c]) > Scalar[DT](1e-4):
                    print("    ST vjp ref mismatch b=", b, " s=", s, " c=", c,
                          " ref=", ref_g, " an=", gi[0].data[base + c])
                    return False
    return True


def _st_sample_parity(ctx: DeviceContext) raises -> Bool:
    var goc = Tensor.alloc(B * SC)
    var inc = TensorPack[1]()
    inc[0].ensure(B * SC)
    for i in range(B * SC):
        inc[0].data[i] = _xfill(i, 8)
        goc.data[i] = _w(i)

    var op_c = StraightThroughSample[STOCH, CLASSES]()
    var out_c = Tensor.alloc(B * SC)
    op_c.forward["cpu", B](TensorRefs[1](inc[0]), out_c, None)
    var gi_c = TensorPack[1]()
    op_c.vjp["cpu", B](TensorRefs[1](inc[0]), goc, TensorRefs[1](gi_c[0]), None)

    var ing = TensorPack[1]()
    ing[0].ensure(B * SC)
    var gog = Tensor.alloc(B * SC)
    for i in range(B * SC):
        ing[0].data[i] = inc[0].data[i]
        gog.data[i] = goc.data[i]
    ing[0].upload(ctx); gog.upload(ctx)

    var op_g = StraightThroughSample[STOCH, CLASSES]()
    var out_g = Tensor.alloc(B * SC)
    op_g.forward["gpu", B](TensorRefs[1](ing[0]), out_g, Optional(ctx))
    var gi_g = TensorPack[1]()
    op_g.vjp["gpu", B](
        TensorRefs[1](ing[0]), gog, TensorRefs[1](gi_g[0]), Optional(ctx)
    )
    out_g.download(ctx); gi_g[0].download(ctx)

    for i in range(B * SC):
        if _abs(out_c.data[i] - out_g.data[i]) > Scalar[DT](1e-4):
            print("    ST fwd parity i=", i)
            return False
        if _abs(gi_c[0].data[i] - gi_g[0].data[i]) > Scalar[DT](1e-4):
            print("    ST vjp parity i=", i)
            return False
    return True


def main() raises:
    print("RSSM custom ops — storage-surface gate")
    var ctx = DeviceContext()

    var asq_fd = _action_squash_cpu_fd()
    var asq_par = _action_squash_parity(ctx)
    print("  ActionSquash         FD:", "OK" if asq_fd else "FAIL",
          " parity:", "OK" if asq_par else "FAIL")

    var bga_fd = _block_group_assemble_cpu_fd()
    var bga_par = _block_group_assemble_parity(ctx)
    print("  BlockGroupAssemble   FD:", "OK" if bga_fd else "FAIL",
          " parity:", "OK" if bga_par else "FAIL")

    var gru_fd = _gru_gate_cpu_fd()
    var gru_par = _gru_gate_parity(ctx)
    print("  GRUGate              FD:", "OK" if gru_fd else "FAIL",
          " parity:", "OK" if gru_par else "FAIL")

    var st_ref = _st_sample_cpu_ref()
    var st_par = _st_sample_parity(ctx)
    print("  StraightThroughSample ref:", "OK" if st_ref else "FAIL",
          " parity:", "OK" if st_par else "FAIL")

    assert_true(
        asq_fd and asq_par and bga_fd and bga_par and gru_fd and gru_par
        and st_ref and st_par,
        "RSSM custom ops storage gate",
    )
    print("RSSM OPS STORAGE OK")
