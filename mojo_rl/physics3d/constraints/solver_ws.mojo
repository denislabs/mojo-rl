"""The contact-solver workspace layout, in ONE place, as RUNTIME offsets.

`ContactScratch.solver` is one flat `[BATCH, SOLVER_WS]` tensor holding every
per-contact quantity the solvers share. Today its regions are addressed by
hand-written comptime offsets — and the SAME common block is re-declared in
FIVE places (`contact_solve` x4, `island_pgs_solve`, `newton_solve`,
`cg_solve`). `elliptic_layout.mojo` was written to stop exactly that for the
elliptic region; this module finishes the job for the rest.

## Why this is runtime and not comptime

Every offset here is a multiple of `mc` and `nv`. Phase 2b.2 makes a *cap* 0 on
the dynamic leg, and the previous spelling took its strides from `D.CAP_NV` and
`D.CAP_MAX_CONTACTS` — so on that leg every region base collapses to 0 and the
whole workspace aliases itself. Silently: `solver` is one big float array, so
every offset in range is a "valid" read.

That is not a hypothetical. `feedback_a_cap_and_a_stride_are_the_same_token`
counted **~266 uses of `MC` as an index stride against 18 as a size** across
newton/cg/island. Reclassifying those one by one is the wrong tool; making the
arithmetic exist once, as a function of the live dims, removes the class.

⚠ `mc` AND `nv` ARE THE LIVE DIMS, NOT CAPS. The static leg passes the same
numbers it always did (cap == exact), so every offset is byte-identical to the
comptime chain it replaces — `test_solver_ws_layout.mojo` asserts that against
the literal expressions, which is the only gate a pure-refactor of addressing
can have.

## The map

Common normal block, one entry per contact unless noted:

      0*mc  lambda_n     1*mc  K_n         2*mc  c_dist
      3*mc  c_body       4*mc  c_body_b    5..7*mc  c_p{x,y,z}
      8..10*mc  c_n{x,y,z}          11*mc  pos_bias
     12*mc  inv_K_imp   13*mc  imp_n      14*mc  diag_n
     15*mc  J_n      [mc, nv]
     15*mc + mc*nv   MinvJn  [mc, nv]

`FRIC_BASE = 15*mc + 2*mc*nv` is where the two cone paths diverge; PYRAMIDAL
and ELLIPTIC overlay the same bytes because a model uses one or the other.

PYRAMIDAL (5 edges/contact = 2*(MAX_CONDIM-1) at condim 6):

    +0*mc   lambda_f[5mc]   +5*mc   K_f[5mc]     +10*mc  dir_f[15mc]
    +25*mc  fric_coef[5mc]  +30*mc  condim[mc]   +31*mc  R_f[5mc]
    +36*mc  bias_f[5mc]     +41*mc  J_f[5mc,nv]
    +41*mc + 5*mc*nv        MinvJ_f[5mc,nv]
    +41*mc + 10*mc*nv       lambda_edge_neg[5mc], then C_nt, K_edge_pos,
                            K_edge_neg, R_edge at +5*mc each

ELLIPTIC: see `elliptic_layout.mojo`, whose `ell_jt` IS `FRIC_BASE`. The
functions here mirror it with runtime arguments; the comptime forms stay for
the GPU leg, which decision 3 keeps comptime.

BUDGET `SOLVER_WS = 81*mc + 12*mc*nv` (`fields/contact_scratch.mojo`).
`ws_end_pyramidal` is the high-water mark; `ws_fits` checks it.
"""


# =============================================================================
# Common normal block
# =============================================================================


@always_inline
def ws_lambda_n(mc: Int) -> Int:
    return 0 * mc


@always_inline
def ws_k_n(mc: Int) -> Int:
    return 1 * mc


@always_inline
def ws_c_dist(mc: Int) -> Int:
    return 2 * mc


@always_inline
def ws_c_body(mc: Int) -> Int:
    return 3 * mc


@always_inline
def ws_c_body_b(mc: Int) -> Int:
    return 4 * mc


@always_inline
def ws_c_p(mc: Int, axis: Int) -> Int:
    """`c_px/c_py/c_pz` — contact point, one block per axis at 5/6/7 * mc."""
    return (5 + axis) * mc


@always_inline
def ws_c_n(mc: Int, axis: Int) -> Int:
    """`c_nx/c_ny/c_nz` — contact normal, one block per axis at 8/9/10 * mc."""
    return (8 + axis) * mc


@always_inline
def ws_pos_bias(mc: Int) -> Int:
    return 11 * mc


@always_inline
def ws_inv_k_imp(mc: Int) -> Int:
    return 12 * mc


@always_inline
def ws_imp_n(mc: Int) -> Int:
    return 13 * mc


@always_inline
def ws_diag_n(mc: Int) -> Int:
    return 14 * mc


@always_inline
def ws_j_n(mc: Int) -> Int:
    """`J_n[mc, nv]` — normal-row Jacobian. Contact `c`, dof `i` at `+ c*nv+i`."""
    return 15 * mc


@always_inline
def ws_minv_jn(mc: Int, nv: Int) -> Int:
    """`MinvJn[mc, nv]`."""
    return 15 * mc + mc * nv


# =============================================================================
# Where the cone paths diverge
# =============================================================================


@always_inline
def ws_fric_base(mc: Int, nv: Int) -> Int:
    """First slot after the common block. PYRAMIDAL and ELLIPTIC overlay it —
    a model uses one cone or the other, never both."""
    return 15 * mc + 2 * mc * nv


# =============================================================================
# PYRAMIDAL friction block
# =============================================================================

comptime PYR_EDGES: Int = 5
"""Edge slots per contact: `2*(MAX_CONDIM-1)` at condim 6. The block is sized
for 5 regardless, which is what `friction_solver.mojo` always allocated."""


@always_inline
def ws_lambda_f(mc: Int, nv: Int) -> Int:
    return ws_fric_base(mc, nv) + 0 * mc


@always_inline
def ws_k_f(mc: Int, nv: Int) -> Int:
    return ws_fric_base(mc, nv) + 5 * mc


@always_inline
def ws_dir_f(mc: Int, nv: Int) -> Int:
    """`dir_f[15*mc]` — 3 components x 5 edges per contact."""
    return ws_fric_base(mc, nv) + 10 * mc


@always_inline
def ws_fric_coef(mc: Int, nv: Int) -> Int:
    return ws_fric_base(mc, nv) + 25 * mc


@always_inline
def ws_condim(mc: Int, nv: Int) -> Int:
    return ws_fric_base(mc, nv) + 30 * mc


@always_inline
def ws_r_f(mc: Int, nv: Int) -> Int:
    return ws_fric_base(mc, nv) + 31 * mc


@always_inline
def ws_bias_f(mc: Int, nv: Int) -> Int:
    return ws_fric_base(mc, nv) + 36 * mc


@always_inline
def ws_j_f(mc: Int, nv: Int) -> Int:
    """`J_f[5*mc, nv]`."""
    return ws_fric_base(mc, nv) + 41 * mc


@always_inline
def ws_minv_j_f(mc: Int, nv: Int) -> Int:
    """`MinvJ_f[5*mc, nv]`."""
    return ws_fric_base(mc, nv) + 41 * mc + 5 * mc * nv


@always_inline
def ws_lambda_edge_neg(mc: Int, nv: Int) -> Int:
    return ws_fric_base(mc, nv) + 41 * mc + 10 * mc * nv


@always_inline
def ws_c_nt(mc: Int, nv: Int) -> Int:
    return ws_lambda_edge_neg(mc, nv) + 5 * mc


@always_inline
def ws_k_edge_pos(mc: Int, nv: Int) -> Int:
    return ws_c_nt(mc, nv) + 5 * mc


@always_inline
def ws_k_edge_neg(mc: Int, nv: Int) -> Int:
    return ws_k_edge_pos(mc, nv) + 5 * mc


@always_inline
def ws_r_edge(mc: Int, nv: Int) -> Int:
    return ws_k_edge_neg(mc, nv) + 5 * mc


@always_inline
def ws_end_pyramidal(mc: Int, nv: Int) -> Int:
    """One past the last PYRAMIDAL slot — the high-water mark of the budget."""
    return ws_r_edge(mc, nv) + 5 * mc


# =============================================================================
# ELLIPTIC block — runtime mirror of `elliptic_layout.mojo`
# =============================================================================


@always_inline
def ws_ell_nt(max_condim: Int) -> Int:
    """Tangential rows per contact, `max_condim - 1`, floored at 1 — a zero
    would make every stride in the region alias."""
    var nt = max_condim - 1
    return nt if nt > 0 else 1


@always_inline
def ws_ell_jt(mc: Int, nv: Int) -> Int:
    """Tangent Jacobians. Row `t` of contact `c` at `+ (t*mc + c)*nv`.
    BLOCK-major in `t`, not contact-major — the pyramidal edge list shares
    this base and lays out the same way."""
    return ws_fric_base(mc, nv)


@always_inline
def ws_ell_sc(mc: Int, nv: Int, max_condim: Int) -> Int:
    """Base of the elliptic SCALAR region.

    ⚠ `ntj = max(nt, 4)` KEEPS CONDIM 3 AND 4 BYTE-IDENTICAL to the layout
    that predates the extra tangents — the Jacobian region used to be four
    `mc*nv` blocks, two of which were dead, and the torsion/rolling rows move
    into that dead space. Only condim 6 grows."""
    var nt = ws_ell_nt(max_condim)
    var ntj = nt if nt > 4 else 4
    return ws_ell_jt(mc, nv) + ntj * mc * nv


@always_inline
def ws_ell_mu(mc: Int, nv: Int, max_condim: Int) -> Int:
    """`mu[mc]` — the REGULARIZED cone coefficient, not `friction[t]`."""
    return ws_ell_sc(mc, nv, max_condim) + 0 * mc


@always_inline
def ws_ell_dn(mc: Int, nv: Int, max_condim: Int) -> Int:
    """`D_n[mc]` — `efc_D` of the normal row."""
    return ws_ell_sc(mc, nv, max_condim) + 1 * mc


@always_inline
def ws_ell_dt(mc: Int, nv: Int, max_condim: Int) -> Int:
    """`D_t[nt, mc]` — `efc_D` PER tangential row, entry `t` at `+ t*mc + c`.

    Per row, not one shared `D_f`: MuJoCo sets
    `R[j+1] = R[1]*friction[0]^2/friction[j]^2`, so at
    `friction="0.7 0.7 0.05"` the torsional row is 196x stiffer than the slide
    rows. A shared value makes spin do nothing."""
    return ws_ell_sc(mc, nv, max_condim) + 2 * mc


@always_inline
def ws_ell_fr(mc: Int, nv: Int, max_condim: Int) -> Int:
    """`friction[nt, mc]` — the RAW per-direction coefficient."""
    var nt = ws_ell_nt(max_condim)
    return ws_ell_dt(mc, nv, max_condim) + nt * mc


@always_inline
def ws_ell_bt(mc: Int, nv: Int, max_condim: Int) -> Int:
    """`bias_t[nt, mc]` — velocity-damping bias per tangential row."""
    var nt = ws_ell_nt(max_condim)
    return ws_ell_fr(mc, nv, max_condim) + nt * mc


@always_inline
def ws_ell_ntc(mc: Int, nv: Int, max_condim: Int) -> Int:
    """`ntc[mc]` — how many of the `nt` rows this contact actually has
    (`dim-1`); 0 for a frictionless or non-penetrating contact."""
    var nt = ws_ell_nt(max_condim)
    return ws_ell_bt(mc, nv, max_condim) + nt * mc


@always_inline
def ws_end_elliptic(mc: Int, nv: Int, max_condim: Int) -> Int:
    return ws_ell_ntc(mc, nv, max_condim) + mc


# =============================================================================
# Budget
# =============================================================================


@always_inline
def ws_budget(mc: Int, nv: Int) -> Int:
    """`SOLVER_WS` — must match `fields/contact_scratch.mojo`."""
    return 81 * mc + 12 * mc * nv


@always_inline
def ws_fits(mc: Int, nv: Int, max_condim: Int) -> Bool:
    """Both cone layouts inside the budget.

    ⚠ OVERRUNNING WOULD NOT CRASH — `solver` is `[BATCH, SOLVER_WS]`, so a
    write past the end lands in the NEXT env's row and corrupts a neighbour
    rather than faulting. That is why this is checked rather than assumed."""
    var b = ws_budget(mc, nv)
    return (
        ws_end_pyramidal(mc, nv) <= b
        and ws_end_elliptic(mc, nv, max_condim) <= b
    )
