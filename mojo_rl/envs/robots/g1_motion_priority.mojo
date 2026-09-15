"""Motion prioritization: sample the motions we track WORST more often.

BFM-Zero's `train.py:350-394`. After every tracking eval it turns each motion's
EMD into a sampling weight and pushes it into two places — the motion library
the env resets from (`update_sampling_weight_by_id`) and the expert window
sampler the discriminator and the tracking-z draw from
(`expert_slicer.update_priorities`). Without it a run keeps spending the same
effort on the clips it already tracks well.

    priority(m) = 2 ** (SCALE * clamp(EMD_m, MIN, MAX))          # mode "exp"

With the released literals that is `2**(2*clamp(emd, 0.5, 2))`: **2x for a
motion at EMD 0.5, 16x at EMD 2.0 — an 8x spread**, and flat outside the clamp
so neither a solved motion nor a hopeless one can run away with the batch.

⚠ CADENCE IS `eval_every_steps`, NOT `checkpoint_every_steps`. Both are
9 600 000 in `train.py:711-717`, which is why reading the wrong one still gave
the right number — do not rely on that.

## How the weight is applied here

The reference's samplers take float weights directly. Ours draw a uniform index
into a flat table, on device, inside a CAPTURED GRAPH. That last part decides
the design: `_gather_expert_windows` reads the expert start buffer from inside
`train_device_kernels`, so the graph bakes in that buffer's POINTER. Swapping
the table for a bigger one at a refresh would leave the graph reading freed
memory — a silent, catastrophic failure 9.6 M steps into a run.

So the tables are FIXED LENGTH and rewritten IN PLACE. Clip `m` occupies
`round(L * share_m)` consecutive slots of the same `L`-entry table, filled by
cycling through that clip's own entries, and a uniform draw over the table
lands in clip `m` with probability `share_m`. Same allocation, same pointer,
same length, no kernel gains a branch, and the resolution is `1/L` — with L in
the hundreds of thousands that is far finer than anything the weights mean.

⚠ The window tables weight by `prio_m * W_m`, not `prio_m`: `W_m` (the number
of windows the clip has) is the share it ALREADY had under a uniform draw, and
`update_sampling_weight_by_id` MULTIPLIES a weight rather than replacing the
distribution. Weighting by `prio_m` alone would silently drop the length
weighting at the same time.

Cycling rather than truncating matters: a clip whose share buys fewer slots
than it has windows must still rotate through all of them across refreshes,
or the tail of every long clip would never be sampled again.
"""

from std.math import exp2

# ── the reference's literals (`train.py:711-717`) ─────────────────────────
comptime G1_PRIO_MIN: Float64 = 0.5        # prioritization_min_val
comptime G1_PRIO_MAX: Float64 = 2.0        # prioritization_max_val
comptime G1_PRIO_SCALE: Float64 = 2.0      # prioritization_scale
comptime G1_PRIO_REFRESH: Int = 9_600_000  # eval_every_steps, in ENV steps



def g1_motion_priority(emd: Float64) -> Float64:
    """`2 ** (SCALE * clamp(emd, MIN, MAX))` — the reference's "exp" mode."""
    var e = emd
    if e < G1_PRIO_MIN:
        e = G1_PRIO_MIN
    elif e > G1_PRIO_MAX:
        e = G1_PRIO_MAX
    return Float64(exp2(G1_PRIO_SCALE * e))


def g1_priority_shares(ref emd: List[Float64]) -> List[Float64]:
    """Normalised sampling share per motion, `priority_m / sum(priority)`."""
    var n = len(emd)
    var out = List[Float64](capacity=n)
    var tot = 0.0
    for i in range(n):
        tot += g1_motion_priority(emd[i])
    if tot <= 0.0:
        for _ in range(n):
            out.append(0.0 if n == 0 else 1.0 / Float64(n))
        return out^
    for i in range(n):
        out.append(g1_motion_priority(emd[i]) / tot)
    return out^


def g1_fill_motion_table(
    ref emd: List[Float64], length: Int, mut out: List[Int]
):
    """`length` motion ids, clip `m` filling `round(length*share_m)` slots.

    Replaces `motion = floor(u * n_motions)` in `rsi_inject_kernel`: the kernel
    indexes THIS instead, so a uniform draw lands on `m` with probability
    `share_m`. Every motion keeps at least one slot — a motion that drops out
    of the reset mix entirely is one the policy stops practising and then
    un-learns, which the clamp exists to prevent and this must not undo.
    """
    var share = g1_priority_shares(emd)
    out.clear()
    var n = len(share)
    if n == 0 or length <= 0:
        return
    for m in range(n):
        var want = Int(Float64(length) * share[m] + 0.5)
        if want < 1:
            want = 1
        for _ in range(want):
            if len(out) < length:
                out.append(m)
    # rounding can leave the table short; top up round-robin so the length is
    # exactly what the caller allocated
    var m2 = 0
    while len(out) < length:
        out.append(m2 % n)
        m2 += 1


def g1_fill_window_table(
    ref items: List[Int], ref begin: List[Int], ref end: List[Int],
    ref emd: List[Float64], mut cursor: List[Int], length: Int,
    mut out: List[Int],
):
    """A fixed-`length` window-start table weighted by `prio_m * W_m`.

    `cursor[m]` is where clip `m`'s cycling left off last refresh and is
    advanced in place, so successive refreshes walk through a clip's windows
    rather than re-emitting the same prefix forever.
    """
    var prio = List[Float64](capacity=len(emd))
    var tot = 0.0
    for m in range(len(emd)):
        var w = Float64(end[m] - begin[m])
        var p = g1_motion_priority(emd[m]) * w      # multiply the EXISTING weight
        prio.append(p)
        tot += p
    out.clear()
    if tot <= 0.0 or length <= 0:
        for i in range(length):
            out.append(items[i % len(items)] if len(items) > 0 else 0)
        return
    for m in range(len(prio)):
        var w = end[m] - begin[m]
        if w <= 0:
            continue
        var want = Int(Float64(length) * prio[m] / tot + 0.5)
        if want < 1:
            want = 1
        for _ in range(want):
            if len(out) >= length:
                break
            out.append(items[begin[m] + (cursor[m] % w)])
            cursor[m] = cursor[m] + 1
    var m2 = 0
    while len(out) < length:
        var w2 = end[m2 % len(prio)] - begin[m2 % len(prio)]
        if w2 > 0:
            out.append(items[begin[m2 % len(prio)]])
        m2 += 1


def g1_realised_share(ref table: List[Int], m: Int) -> Float64:
    """The share motion `m` actually gets from a filled motion table."""
    if len(table) == 0:
        return 0.0
    var c = 0
    for i in range(len(table)):
        if table[i] == m:
            c += 1
    return Float64(c) / Float64(len(table))
