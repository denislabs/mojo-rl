"""lambda_return — DreamerV3 λ-return reduction.

Verbatim port of `references/dreamerv3-main/dreamerv3/agent.py:482`:

    rets   = [boot[:, -1]]
    live   = (1 - term)[:, 1:] · disc
    cont   = (1 - last)[:, 1:] · lam
    interm = rew[:, 1:] + (1 - cont)·live·boot[:, 1:]
    for t in reversed(range(T-1)):
        rets.append(interm[:, t] + live[:, t]·cont[:, t]·rets[-1])
    ret = stack(reversed(rets)[:-1], 1)        # [B, T-1]

The reference signature has a `val` argument that the body never uses
(`imag_loss` passes `val == boot == tarval`); dropped here.

FORWARD ONLY — `imag_loss` builds the return from the *detached* slow
value (`tarval`) and the policy/value losses consume `sg(ret)`. No
gradient flows through this reduction, so there is no backward.

Per-row recurrence: `out[b, t] = interm[b,t] + live[b,t]·cont[b,t]·out[b,t+1]`
for `t = T-2 … 0`, seeded by `out[b, T-1] := boot[b, T-1]`. `out` is
`[B, T-1]`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from mojo_rl.nn2.constants import DT, TPB


@always_inline
def _row_recurrence[
    T: Int
](
    last: UnsafePointer[Scalar[DT], MutAnyOrigin],
    term: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
    boot: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: Int,
    disc: Scalar[DT],
    lam: Scalar[DT],
):
    """λ-return backward recurrence for one batch row `b`. `*` pointers are
    flat [BATCH·T] (inputs) / [BATCH·(T-1)] (out), row-major."""
    var in_base = b * T
    var out_base = b * (T - 1)
    var ret_next = boot[in_base + (T - 1)]
    var t = T - 2
    while t >= 0:
        # live/cont/interm are defined on the [1:] slice → index t+1 of inputs.
        var live = (Scalar[DT](1.0) - term[in_base + t + 1]) * disc
        var cont = (Scalar[DT](1.0) - last[in_base + t + 1]) * lam
        var interm = (
            rew[in_base + t + 1]
            + (Scalar[DT](1.0) - cont) * live * boot[in_base + t + 1]
        )
        var cur = interm + live * cont * ret_next
        out_buf[out_base + t] = cur
        ret_next = cur
        t -= 1


def lambda_return_cpu[
    BATCH: Int, T: Int
](
    last: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    term: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    rew: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    boot: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    mut out_buf: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    disc: Scalar[DT],
    lam: Scalar[DT],
) raises:
    """CPU λ-return. `last`/`term`/`rew`/`boot` are [BATCH, T]; `out` is
    [BATCH, T-1]."""
    comptime assert T >= 2, "lambda_return needs T >= 2"
    var last_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](last.ptr)
    var term_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](term.ptr)
    var rew_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](rew.ptr)
    var boot_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](boot.ptr)
    var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_buf.ptr)
    for b in range(BATCH):
        _row_recurrence[T](last_p, term_p, rew_p, boot_p, out_p, b, disc, lam)


def _lambda_return_kernel[
    BATCH: Int, T: Int
](
    last: UnsafePointer[Scalar[DT], MutAnyOrigin],
    term: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
    boot: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    disc: Scalar[DT],
    lam: Scalar[DT],
):
    """One thread per batch row; each walks the T-recurrence sequentially."""
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    _row_recurrence[T](last, term, rew, boot, out_buf, b, disc, lam)


def lambda_return_gpu[
    BATCH: Int, T: Int
](
    ctx: DeviceContext,
    last: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    term: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    rew: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    boot: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    mut out_buf: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1,
        origin=MutAnyOrigin, ...,
    ],
    disc: Scalar[DT],
    lam: Scalar[DT],
) raises:
    comptime assert T >= 2, "lambda_return needs T >= 2"
    var last_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](last.ptr)
    var term_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](term.ptr)
    var rew_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](rew.ptr)
    var boot_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](boot.ptr)
    var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_buf.ptr)
    var n_blocks = (BATCH + TPB - 1) // TPB
    ctx.enqueue_function[_lambda_return_kernel[BATCH, T]](
        last_p, term_p, rew_p, boot_p, out_p, disc, lam,
        grid_dim=n_blocks, block_dim=TPB,
    )
