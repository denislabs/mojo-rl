# +--------------------------------------------------------------------------+ #
# | SmolVLADenoise[RECORD=True] — the activation tape V2's backward reads
# +--------------------------------------------------------------------------+ #
"""Recording must change what the forward KEEPS and nothing about what it says.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_denoise_tape.mojo

A backward pass needs the activations its forward produced, and V1's driver
throws all of them away: one scratch pool, reused by sixteen layers, so by the
time the last layer finishes the first fifteen are gone.

The obvious fix is a second driver that re-runs the sequence while saving. It
is also the wrong one. `_a_rule_written_inline_twice_drifts` is this
codebase's single most recurring defect, and a sixteen-layer body written
twice — once to run, once to run-and-save — is that shape exactly, with a
failure mode nobody would catch: the two would agree for months and then one
would get a fix the other did not, and a fine-tune would be computing
gradients of a slightly different network than the one it evaluates.

So there is ONE body. `RECORD` is a comptime struct parameter that changes
where the layer writes, not what it computes:

    RECORD = False    one pool, `XO` IS `X`, the residual overwrites its own
                      input exactly as V1 does. No extra slab, no extra copy.
    RECORD = True     one pool PER LAYER, plus one for the value the final
                      norm consumes. `XO` is its own slot and the running
                      activation is handed to the next layer's pack.

## What this gate has to establish, and why bit-equality is the bar

Leg [1] runs both instantiations over the same weights, the same cache and the
same suffix, and demands the outputs be **BIT-IDENTICAL** — not close. The two
perform the same arithmetic in the same order; only the destination slabs
differ. Any tolerance at all here would accept a recording path that had
quietly reordered something, and reordering is precisely how a saved
activation stops being the one its layer actually consumed.

Leg [2] is the anti-vacuity leg, and it is the one that matters. A tape of
seventeen zeroed slabs would pass leg [1] perfectly — leg [1] does not read
the tape. So: every entry is compared against its neighbour and must DIFFER
(a layer that recorded nothing leaves them equal), the first must equal the
driver's input exactly, and the last must equal what the non-recording driver
left in its own pool — which is the pre-norm activation, arrived at by a
different route.

### MEASURED: an ablation leg [1] cannot see

Removing the `XO`/`X` split — `comptime XO: Int = Self.X` unconditionally, so
the residual overwrites its own input under RECORD too, which is a plausible
simplification of exactly one line:

    [1] RECORD on vs off: compared 4320  differing 0        <- still perfect
    [2] tape: all-zero 0  identical-to-previous 1  seed mismatches 4320

The answer is untouched, because the chain still works: each layer's output
still reaches the next. What breaks is what the tape MEANS — every entry now
holds its layer's output instead of its input, so a backward pass would
differentiate each layer at the wrong point and produce a finite, plausible,
wrong gradient for all sixteen.

Note which check caught it. `identical-to-previous` found only 1 pair, because
consecutive layer outputs are genuinely different values; it was the
tape[0]-equals-the-input check that fired on all 4,320. A neighbour-difference
test alone — the obvious way to write this leg — would have let it through.

⚠ Small dims and a seeded tower. This gate is about plumbing, and the numbers
it compares are its own; parity against `lerobot` is `test_parity_vs_hf.mojo`.
"""

from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.text import (
    SmolVLMTextLayers, SMOLLM_LAYERS, SMOLLM_DIM, SMOLLM_FF, SMOLLM_KV_W,
    SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
)
from mojo_rl.deep_agents.smolvla.expert import (
    SmolVLAExpert, EXPERT_W, EXPERT_FF,
)
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from mojo_rl.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)

comptime P = 24
comptime S = 6
comptime B = 1
comptime L = SMOLLM_LAYERS
comptime W = SMOLLM_DIM
comptime KVW = SMOLLM_KV_W
comptime XN_P = P * W
comptime XN_S = S * EXPERT_W

comptime Tower = SmolVLMTextLayers[L, W, SMOLLM_FF, KVW]
comptime Expert = SmolVLAExpert[L, EXPERT_W, EXPERT_FF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, B]
comptime Pre = SmolVLAPrefill[P, S, B]
comptime Den = SmolVLADenoise[P, S, B]
comptime DenRec = SmolVLADenoise[P, S, B, RECORD=True]
"""⚠ Everything but RECORD must come from the SAME defaults, or leg [1]
compares two different networks and reports the difference as a recording
bug. Naming only what differs is what guarantees that."""


def main() raises:
    print("=" * 70)
    print("SmolVLADenoise recording tape")
    print("=" * 70)
    print("  prefix", P, " suffix", S, " layers", L, " pools", L + 1)

    var d = DeviceContext()
    var ar_pre = smolvla_ar(8, 15, 1, 0)
    var ar_full = smolvla_ar(8, 15, 1, S)
    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + S, 0, P + S)
    var mask_cross = att_2d_mask(ar_full, P, P + S, 0, P)

    var tower = Tower.make["gpu", Deterministic](Optional(d))
    var expert = Expert.make["gpu", Deterministic](Optional(d))
    var cache = Cache.make["gpu"](Optional(d))
    var pre = Pre.make["gpu"](mask_pre, Optional(d))

    var xp = Tensor.alloc(B * XN_P)
    for i in range(B * XN_P):
        xp.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.05
    xp.upload(d)
    var pre_out = Tensor.alloc(B * XN_P)
    pre.run["gpu"](tower, cache, xp, pre_out, Optional(d))
    assert_equal(cache.n_filled(), L, "prefill must fill every layer")

    # the suffix both drivers see
    var xs_h = List[Scalar[DT]](unsafe_uninit_length=B * XN_S)
    for i in range(B * XN_S):
        xs_h[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.03

    # ── [1] recording changes nothing about the answer ───────────────────
    var xs_a = Tensor.alloc(B * XN_S)
    var xs_b = Tensor.alloc(B * XN_S)
    for i in range(B * XN_S):
        xs_a.data[i] = xs_h[i]
        xs_b.data[i] = xs_h[i]
    xs_a.upload(d)
    xs_b.upload(d)

    var den = Den.make["gpu"](mask_self, mask_cross, Optional(d))
    var out_plain = Tensor.alloc(B * XN_S)
    den.step["gpu"](expert, cache, xs_a, out_plain, Optional(d))
    out_plain.download(d)

    var denr = DenRec.make["gpu"](mask_self, mask_cross, Optional(d))
    var out_rec = Tensor.alloc(B * XN_S)
    denr.step["gpu"](expert, cache, xs_b, out_rec, Optional(d))
    out_rec.download(d)

    var diff = 0
    var worst = Scalar[DT](0)
    for i in range(B * XN_S):
        var e = out_rec.data[i] - out_plain.data[i]
        if e < Scalar[DT](0):
            e = -e
        if e > worst:
            worst = e
        if out_rec.data[i] != out_plain.data[i]:
            diff += 1
    print("  [1] RECORD on vs off: compared", B * XN_S, " differing", diff,
          " worst", worst)
    assert_true(
        diff == 0,
        "recording changed the answer — the tape is not a passive observer",
    )

    # ── [2] the tape is real ─────────────────────────────────────────────
    # Every entry downloaded and compared with its neighbour. A tape of
    # zeroed slabs passes leg [1] and fails here.
    var prev = List[Scalar[DT]](unsafe_uninit_length=B * XN_S)
    var same_as_prev = 0
    var all_zero = 0
    var seeded_bad = 0
    var checked = 0
    for l in range(L + 1):
        denr.pools[l][DenRec.X].download(d)
        var nz = 0
        var eq = 0
        for i in range(B * XN_S):
            var y = denr.pools[l][DenRec.X].data[i]
            if y != Scalar[DT](0):
                nz += 1
            if l > 0 and y == prev[i]:
                eq += 1
            if l == 0 and y != xs_h[i]:
                seeded_bad += 1
            prev[i] = y
            checked += 1
        if nz == 0:
            all_zero += 1
        if l > 0 and eq == B * XN_S:
            same_as_prev += 1
    print("  [2] tape:", L + 1, "entries,", checked, "values;  all-zero",
          all_zero, " identical-to-previous", same_as_prev,
          " seed mismatches", seeded_bad)
    assert_equal(
        checked, (L + 1) * B * XN_S, "every tape entry must be inspected"
    )
    assert_true(all_zero == 0, "a tape entry was never written")
    assert_true(
        same_as_prev == 0,
        "two consecutive tape entries are identical — a layer recorded its"
        " predecessor's activation, not its own",
    )
    assert_true(
        seeded_bad == 0,
        "tape[0] is not the driver's input: the seed did not reach pack 0",
    )

    # ── [3] the last entry is the pre-norm activation, by another route ──
    # The non-recording driver leaves it in its single pool; the recording one
    # arrives at pack LAYERS through sixteen hand-offs. Same value or one of
    # them is wrong.
    den.pools[0][Den.X].download(d)
    var tail_bad = 0
    for i in range(B * XN_S):
        if (
            denr.pools[L][DenRec.X].data[i] != den.pools[0][Den.X].data[i]
        ):
            tail_bad += 1
    print("  [3] tape[", L, "] vs the plain driver's pool: compared",
          B * XN_S, " differing", tail_bad)
    assert_true(
        tail_bad == 0,
        "the recorded pre-norm activation is not the one the plain driver"
        " computed",
    )

    print()
    print("PASSED — recording is free of the answer, and the tape is not empty")
