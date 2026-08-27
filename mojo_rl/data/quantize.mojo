# +--------------------------------------------------------------------------+ #
# | Observation quantisation
# +--------------------------------------------------------------------------+ #
"""Obs storage-dtype conversion, shared by every replay buffer that stores
observations at a narrower dtype than it serves them.

Extracted from `deep_agents/data/gpu_replay.mojo` (2026-08-05, 4d batch 3).
Four sequence buffers — `data/gpu_sequence_replay.mojo` and the three
`zero/*_sequence_replay*.mojo` — imported these helpers from that module,
which made it undeletable for a reason unrelated to replay. They live here
now, and `data/replay_gpu.mojo`'s copy is gone with them.

**Why it exists.** Pixel replay stores obs as `uint8`: Pong frames are
4x84x84 = 28,224 elements, so at CAP=100k obs+next_obs is ~22.6 GB in fp32
against ~5.6 GB in uint8. That is a fits-in-VRAM question, not an
optimisation, and Rainbow's CNN configs default to uint8 accordingly.
"""

from mojo_rl.nn.constants import DT


def obs_quant[SDT: DType](x: Scalar[DT]) -> Scalar[SDT]:
    """`DT` obs element -> storage dtype.

    `SDT == DT` is a pure rebind (bit-identical store). `uint8` stores
    `round(x*255)` clamped to [0, 255] — exact for `k/255` pixel inputs, which
    is what the pixel pipeline produces.
    """
    comptime if SDT == DT:
        return rebind[Scalar[SDT]](x)
    else:
        var v = x * Scalar[DT](255.0) + Scalar[DT](0.5)
        if v < Scalar[DT](0.0):
            v = Scalar[DT](0.0)
        if v > Scalar[DT](255.0):
            v = Scalar[DT](255.0)
        return v.cast[SDT]()


def obs_dequant[SDT: DType](x: Scalar[SDT]) -> Scalar[DT]:
    """Storage dtype -> `DT`.

    `uint8` divides by 255.0 — the same division the pixel pipeline used to
    produce the stored value, so the round trip is bit-identical for `k/255`.
    """
    comptime if SDT == DT:
        return rebind[Scalar[DT]](x)
    else:
        return x.cast[DT]() / Scalar[DT](255.0)


# Underscore aliases: the four sequence buffers already call `_obs_quant` /
# `_obs_dequant`. Kept so the extraction is a one-line import change at each
# site rather than an edit of every call.
comptime _obs_quant = obs_quant
comptime _obs_dequant = obs_dequant
