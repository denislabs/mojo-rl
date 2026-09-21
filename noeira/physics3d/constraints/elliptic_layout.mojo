"""Workspace layout for the ELLIPTIC contact block, in one place.

⚠⚠ ONE SOURCE OF TRUTH, ON PURPOSE — the same argument `je_budget.mojo` makes.
Four call sites index this region and they MUST agree exactly: the PRODUCER
(`constraints/contact_solve._precompute_contact_friction`) and three CONSUMERS
(`solver/newton_solve._newton_solve_env`, the blocked Newton kernel, and
`solver/cg_solve`). They used to each declare their own `comptime ws_*_idx =
SC + k*MC` chain; that worked only while the chain was a fixed seven entries.
It is now `MAX_CONDIM`-dependent, and a consumer computing a stale stride would
read a DIFFERENT contact's friction rather than fail — silently, since every
offset in range is a valid float.

WHAT A CONTACT OWNS. MuJoCo's elliptic cone gives a contact of dimension `dim`
ONE normal row followed by `dim-1` tangential rows
(`engine_core_constraint.c`, `mjCNSTR_CONTACT_ELLIPTIC`). The friction
direction `t` pairs with `con->friction[t]`:

    t = 0, 1   the two SLIDE tangents  (linear Jacobian along t1 / t2)
    t = 2      TORSION about the contact normal   (angular Jacobian)
    t = 3, 4   ROLLING about t1 / t2              (angular Jacobian)

so `NT = MAX_CONDIM - 1` is 2 at condim 3, 3 at condim 4 and 5 at condim 6.
Slots are sized for the model's WORST condim and each contact's own `dim` is a
RUNTIME value (condim is per geom pair), so rows `t >= dim-1` are zeroed per
contact and `ws_ntc` records how many are live.

⚠ `NTJ = max(NT, 4)` KEEPS THE SCALAR REGION WHERE IT WAS AT CONDIM 3. The
Jacobian region used to be four `MC*NV` blocks — `Jt1`, `Jt2`, and two
`MinvJt1`/`MinvJt2` blocks that were zeroed at init and never read again. The
extra tangents move into that dead space, so a condim-3 or condim-4 model gets
byte-identical offsets to the ones it had before this file existed, and only a
condim-6 model grows.

BUDGET. `SOLVER_WS = 81*MC + 12*MC*NV` everywhere (see
`fields/contact_scratch.mojo`). `ell_end` is `24*MC + 6*MC*NV` at condim 3 and
`33*MC + 7*MC*NV` at condim 6, both well inside it — the budget was sized for
the PYRAMIDAL path, which needs `2*(MAX_CONDIM-1) = 10` Jacobian blocks where
elliptic needs 5.
"""


def _max_one[N: Int]() -> Int:
    """`max(N, 1)` — a zero-sized dimension is a crash, not an empty tensor."""
    return N if N > 0 else 1


def ell_nt[MAX_CONDIM: Int]() -> Int:
    """`NT` — tangential rows per contact slot, `MAX_CONDIM - 1`.

    Floored at 1: `MAX_CONDIM` is >= 3 for every model the parser produces
    (`xml_parser._scan_max_condim`), but a hand-built model def that passes 1
    would otherwise size the region at zero and make every stride alias.
    """
    return _max_one[MAX_CONDIM - 1]()


def ell_jt[MC: Int, NV: Int]() -> Int:
    """Base of the tangent Jacobians: `NT` blocks of `MC*NV`.

    Row `t` of contact `c` starts at `ell_jt + (t*MC + c)*NV`. Block-major in
    `t` — NOT contact-major — because the PYRAMIDAL path lays its edge list out
    the same way and the two share `_precompute_contact_friction`'s base.
    """
    return 15 * MC + 2 * MC * NV


def ell_sc[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """Base of the elliptic SCALAR region (the old `SC`)."""
    comptime NT = ell_nt[MAX_CONDIM]()
    comptime NTJ = NT if NT > 4 else 4
    return ell_jt[MC, NV]() + NTJ * MC * NV


def ell_mu[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """`MC` — `con->mu`, the REGULARIZED cone coefficient (see the producer)."""
    return ell_sc[MC, NV, MAX_CONDIM]() + 0 * MC


def ell_dn[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """`MC` — `efc_D` of the normal row, `1/R[0]`."""
    return ell_sc[MC, NV, MAX_CONDIM]() + 1 * MC


def ell_dt[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """`NT*MC` — `efc_D` per tangential row; entry `t` at `+ t*MC + c`.

    ⚠ PER ROW, NOT ONE `D_f` FOR ALL OF THEM. MuJoCo sets
    `R[j+1] = R[1]*friction[0]^2/friction[j]^2`, so the torsional and rolling
    rows are stiffer than the slide rows by exactly the ratio of their friction
    coefficients squared. A ball at `friction="0.7 0.7 0.05"` has a torsional
    row 196x stiffer than its slide rows; the single shared `D_f` this
    replaced made torsion as soft as sliding, which reads as "spin does
    nothing".
    """
    return ell_sc[MC, NV, MAX_CONDIM]() + 2 * MC


def ell_fr[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """`NT*MC` — `con->friction[t]`, the RAW per-direction coefficient.

    Distinct from `ell_mu`, which is the regularized `con->mu`. Both are
    needed: `mu` sets the cone's zone boundaries, `friction[t]` maps row `t`
    into the space where the cone is circular.
    """
    comptime NT = ell_nt[MAX_CONDIM]()
    return ell_dt[MC, NV, MAX_CONDIM]() + NT * MC


def ell_bt[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """`NT*MC` — velocity-damping bias `B * J_t . qvel` per tangential row."""
    comptime NT = ell_nt[MAX_CONDIM]()
    return ell_fr[MC, NV, MAX_CONDIM]() + NT * MC


def ell_ntc[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """`MC` — how many of the `NT` rows this contact actually has (`dim-1`).

    Zero for a contact that is not penetrating, and 0 for a FRICTIONLESS
    (`condim="1"`) contact, which is one normal row and nothing else.
    """
    comptime NT = ell_nt[MAX_CONDIM]()
    return ell_bt[MC, NV, MAX_CONDIM]() + NT * MC


def ell_end[MC: Int, NV: Int, MAX_CONDIM: Int]() -> Int:
    """One past the last slot this layout uses — for the budget assertion."""
    return ell_ntc[MC, NV, MAX_CONDIM]() + MC
