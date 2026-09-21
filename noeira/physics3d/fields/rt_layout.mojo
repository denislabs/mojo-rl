"""`DYN1`/`DYN2` + `rl1`/`rl2` — a tensor view whose EXTENTS are runtime.

This is phase 3a. `Scratch` (2b.2) solved the function-local buffers; this
solves the other half: the `LayoutTensor` views a dispatcher builds over
`Data`/`Model` tensors. Today they read

    comptime L_NV = Layout.row_major(BATCH, D.NV)
    var qvel_v = d.qvel.lt["cpu", L_NV]()

and `D.NV` is `DIM_POISON` on a dynamic provider, so the layout is not
spellable there at all. The runtime form is

    var rl_nv = rl2(BATCH, dm.get_nv())
    var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_nv)

with `dm = d.dims` — the provider VALUE the container now carries.

## Why this is not merely tolerable but FASTER

§15.1 measured it on the Newton solve, static provider both sides, arm B
verified bit-identical to arm A:

| walker2d PYR | hopper PYR | walker2d ELL | hopper ELL |
|---|---|---|---|
| 0.855 | 0.796 | 0.934 | 0.921 |

⇒ **Runtime layouts are ~10-20% FASTER than comptime ones on the shipped
leg.** Unexplained; recorded as measured. It is why 3a is worth doing on its
own, ahead of and independent of any dynamic-leg work — unlike `Scratch`,
where the two legs genuinely want different containers and §15.2 makes
keeping both a RULE, layouts want exactly one kind.

⚠ THAT MEASUREMENT IS THE SOLVE ONLY. FK, collision and the integrators have
different access patterns and are INFERRED, not measured. Each package
carries its own A/B.

## ⚠⚠ THE FAILURE MODE IS SILENT, UNLIKE 2b.1's

A wrong `D.NV` in a comptime layout is a compile error or a poison value. A
wrong extent in a `RuntimeLayout` is a legal layout over the WRONG MEMORY: it
compiles, it runs, and it reads a neighbouring row. That is the cap-as-stride
class from 2b.2, which no gate in the tree could see and which was found by
static audit three times.

So the conversion is a TOKEN-LEVEL TRANSCRIPTION and not a rewrite:

    Layout.row_major(BATCH, D.NBODY * 3)  ->  rl2(BATCH, dm.get_nbody() * 3)

expression tree preserved, one substitution per dimension. Anything that
re-derives an extent by hand is how a wrong one gets in.

## The GPU branches keep their comptime layouts

Decision 3: kernels stay comptime. A kernel signature spells
`LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin]`, and a
runtime provider cannot cross into one anyway (a captured provider reads 0
and the output is silently zeroed). So a dispatcher with both branches keeps
`comptime L_*` for the GPU side and gains `var rl_*` for the CPU side. They
are not redundant; they serve different legs.
"""

from std.utils import IndexList
from layout import Layout, RuntimeLayout

comptime DYN1 = Layout.row_major[1]()
"""A 1-D layout with its extent UNKNOWN at compile time; `rl1` supplies it."""

comptime DYN2 = Layout.row_major[2]()
"""A 2-D layout with both extents UNKNOWN at compile time; `rl2` supplies
them. Nearly every physics3d field is `[BATCH, F]`, so this is the common
one."""


@always_inline
def rl1(n: Int) -> RuntimeLayout[DYN1]:
    """Extents for a `DYN1` view. Pass the LIVE length."""
    return RuntimeLayout[DYN1].row_major(IndexList[1](n))


@always_inline
def rl2(rows: Int, cols: Int) -> RuntimeLayout[DYN2]:
    """Extents for a `DYN2` view — row-major `[rows, cols]`.

    ⚠ `cols` IS THE ROW STRIDE, so it is the argument a cap-as-stride bug
    lands in. It must be the live per-row width (`dm.get_nv()`,
    `dm.get_nbody() * 3`), never a cap and never a container's allocated
    width if that differs.
    """
    return RuntimeLayout[DYN2].row_major(IndexList[2](rows, cols))
