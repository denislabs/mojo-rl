"""`solver_ws` reproduces the comptime workspace offsets EXACTLY.

A refactor that only changes ADDRESSING has no behavioural gate: every offset
in `solver` is a valid float slot, so a wrong one reads a different contact's
data instead of faulting. The only real check is arithmetic identity against
the expressions being replaced — the literal `k * MC` chains that were
duplicated across `contact_solve`, `island_pgs_solve`, `newton_solve` and
`cg_solve`, and the comptime `ell_*` functions.

⚠ THE COMPTIME SIDE IS SPELLED OUT AS LITERALS ON PURPOSE. Comparing
`solver_ws.ws_c_dist(mc)` against a helper that also says `2 * mc` would pass
for any value of both. The literals below are transcribed from the call sites
(`contact_solve.mojo:371-387`, `:1326-1340`) so the two sides have independent
provenance.

Run: pixi run mojo run -I . tests/physics3d/test_solver_ws_layout.mojo
"""

from mojo_rl.physics3d.constraints.solver_ws import (
    ws_lambda_n,
    ws_k_n,
    ws_c_dist,
    ws_c_body,
    ws_c_body_b,
    ws_c_p,
    ws_c_n,
    ws_pos_bias,
    ws_inv_k_imp,
    ws_imp_n,
    ws_diag_n,
    ws_j_n,
    ws_minv_jn,
    ws_fric_base,
    ws_lambda_f,
    ws_k_f,
    ws_dir_f,
    ws_fric_coef,
    ws_condim,
    ws_r_f,
    ws_bias_f,
    ws_j_f,
    ws_minv_j_f,
    ws_lambda_edge_neg,
    ws_c_nt,
    ws_k_edge_pos,
    ws_k_edge_neg,
    ws_r_edge,
    ws_end_pyramidal,
    ws_ell_nt,
    ws_ell_jt,
    ws_ell_sc,
    ws_ell_mu,
    ws_ell_dn,
    ws_ell_dt,
    ws_ell_fr,
    ws_ell_bt,
    ws_ell_ntc,
    ws_end_elliptic,
    ws_budget,
    ws_fits,
)
from mojo_rl.physics3d.constraints.elliptic_layout import (
    ell_nt,
    ell_jt,
    ell_sc,
    ell_mu,
    ell_dn,
    ell_dt,
    ell_fr,
    ell_bt,
    ell_ntc,
    ell_end,
)


struct Tally(Movable):
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def eq(mut self, got: Int, want: Int, what: String):
        self.checks += 1
        if got != want:
            self.fails += 1
            print("  FAIL", what, "got", got, "want", want)

    def truth(mut self, got: Bool, what: String):
        self.checks += 1
        if not got:
            self.fails += 1
            print("  FAIL", what)


def check_common(mut t: Tally, mc: Int, nv: Int):
    """Literals transcribed from contact_solve.mojo:371-387."""
    t.eq(ws_lambda_n(mc), 0 * mc, "lambda_n")
    t.eq(ws_k_n(mc), 1 * mc, "K_n")
    t.eq(ws_c_dist(mc), 2 * mc, "c_dist")
    t.eq(ws_c_body(mc), 3 * mc, "c_body")
    t.eq(ws_c_body_b(mc), 4 * mc, "c_body_b")
    t.eq(ws_c_p(mc, 0), 5 * mc, "c_px")
    t.eq(ws_c_p(mc, 1), 6 * mc, "c_py")
    t.eq(ws_c_p(mc, 2), 7 * mc, "c_pz")
    t.eq(ws_c_n(mc, 0), 8 * mc, "c_nx")
    t.eq(ws_c_n(mc, 1), 9 * mc, "c_ny")
    t.eq(ws_c_n(mc, 2), 10 * mc, "c_nz")
    t.eq(ws_pos_bias(mc), 11 * mc, "pos_bias")
    t.eq(ws_inv_k_imp(mc), 12 * mc, "inv_K_imp")
    t.eq(ws_imp_n(mc), 13 * mc, "imp_n")
    t.eq(ws_diag_n(mc), 14 * mc, "diag_n")
    t.eq(ws_j_n(mc), 15 * mc, "J_n")
    t.eq(ws_minv_jn(mc, nv), 15 * mc + mc * nv, "MinvJn")


def check_pyramidal(mut t: Tally, mc: Int, nv: Int):
    """Literals transcribed from contact_solve.mojo:1325-1340."""
    var fws = 15 * mc + 2 * mc * nv
    t.eq(ws_fric_base(mc, nv), fws, "fric_base")
    t.eq(ws_lambda_f(mc, nv), fws + 0 * mc, "lambda_f")
    t.eq(ws_k_f(mc, nv), fws + 5 * mc, "K_f")
    t.eq(ws_dir_f(mc, nv), fws + 10 * mc, "dir_f")
    t.eq(ws_fric_coef(mc, nv), fws + 25 * mc, "fric_coef")
    t.eq(ws_condim(mc, nv), fws + 30 * mc, "condim")
    t.eq(ws_r_f(mc, nv), fws + 31 * mc, "R_f")
    t.eq(ws_bias_f(mc, nv), fws + 36 * mc, "bias_f")
    t.eq(ws_j_f(mc, nv), fws + 41 * mc, "J_f")
    t.eq(ws_minv_j_f(mc, nv), fws + 41 * mc + 5 * mc * nv, "MinvJ_f")
    var le = fws + 41 * mc + 10 * mc * nv
    t.eq(ws_lambda_edge_neg(mc, nv), le, "lambda_edge_neg")
    t.eq(ws_c_nt(mc, nv), le + 5 * mc, "C_nt")
    t.eq(ws_k_edge_pos(mc, nv), le + 10 * mc, "K_edge_pos")
    t.eq(ws_k_edge_neg(mc, nv), le + 15 * mc, "K_edge_neg")
    t.eq(ws_r_edge(mc, nv), le + 20 * mc, "R_edge")
    t.eq(ws_end_pyramidal(mc, nv), le + 25 * mc, "end_pyramidal")


def check_elliptic[MC: Int, NV: Int, MAX_CONDIM: Int](mut t: Tally):
    """Against the COMPTIME `ell_*` — the functions the GPU leg still uses.

    The two must agree or the CPU and GPU legs address different bytes, which
    is the failure `elliptic_layout.mojo`'s own docstring warns about.
    """
    var mc = MC
    var nv = NV
    var cd = MAX_CONDIM
    t.eq(ws_ell_nt(cd), ell_nt[MAX_CONDIM](), "ell_nt")
    t.eq(ws_ell_jt(mc, nv), ell_jt[MC, NV](), "ell_jt")
    t.eq(ws_ell_sc(mc, nv, cd), ell_sc[MC, NV, MAX_CONDIM](), "ell_sc")
    t.eq(ws_ell_mu(mc, nv, cd), ell_mu[MC, NV, MAX_CONDIM](), "ell_mu")
    t.eq(ws_ell_dn(mc, nv, cd), ell_dn[MC, NV, MAX_CONDIM](), "ell_dn")
    t.eq(ws_ell_dt(mc, nv, cd), ell_dt[MC, NV, MAX_CONDIM](), "ell_dt")
    t.eq(ws_ell_fr(mc, nv, cd), ell_fr[MC, NV, MAX_CONDIM](), "ell_fr")
    t.eq(ws_ell_bt(mc, nv, cd), ell_bt[MC, NV, MAX_CONDIM](), "ell_bt")
    t.eq(ws_ell_ntc(mc, nv, cd), ell_ntc[MC, NV, MAX_CONDIM](), "ell_ntc")
    t.eq(ws_end_elliptic(mc, nv, cd), ell_end[MC, NV, MAX_CONDIM](), "ell_end")


def check_disjoint(mut t: Tally, mc: Int, nv: Int, max_condim: Int):
    """Regions must be strictly ordered and inside the budget.

    Overlap does not crash — `solver` is `[BATCH, SOLVER_WS]`, so running past
    the end writes into the NEXT env's row.
    """
    # common block, ascending
    var common: List[Int] = [
        ws_lambda_n(mc), ws_k_n(mc), ws_c_dist(mc), ws_c_body(mc),
        ws_c_body_b(mc), ws_c_p(mc, 0), ws_c_p(mc, 1), ws_c_p(mc, 2),
        ws_c_n(mc, 0), ws_c_n(mc, 1), ws_c_n(mc, 2), ws_pos_bias(mc),
        ws_inv_k_imp(mc), ws_imp_n(mc), ws_diag_n(mc), ws_j_n(mc),
    ]
    for i in range(1, len(common)):
        t.truth(common[i] > common[i - 1], "common block ascending at " + String(i))
    t.truth(ws_minv_jn(mc, nv) >= ws_j_n(mc) + mc * nv, "J_n does not overlap MinvJn")
    t.truth(
        ws_fric_base(mc, nv) >= ws_minv_jn(mc, nv) + mc * nv,
        "MinvJn does not overlap the friction base",
    )
    t.truth(ws_fits(mc, nv, max_condim), "both cone layouts fit the budget")
    t.truth(
        ws_end_pyramidal(mc, nv) <= ws_budget(mc, nv), "pyramidal inside budget"
    )
    t.truth(
        ws_end_elliptic(mc, nv, max_condim) <= ws_budget(mc, nv),
        "elliptic inside budget",
    )


def main():
    print("=== solver_ws layout identity ===")
    var t = Tally()

    # A spread of shapes: a small arm, walker2d, humanoid, and a wide model.
    var shapes: List[Int] = [4, 9, 12, 23, 27, 48]
    var contacts: List[Int] = [1, 4, 16, 64]
    for si in range(len(shapes)):
        for ci in range(len(contacts)):
            var nv = shapes[si]
            var mc = contacts[ci]
            check_common(t, mc, nv)
            check_pyramidal(t, mc, nv)
            for cd in range(3, 7):
                check_disjoint(t, mc, nv, cd)

    # Elliptic needs comptime arguments for the reference side.
    check_elliptic[4, 9, 3](t)
    check_elliptic[4, 9, 4](t)
    check_elliptic[4, 9, 6](t)
    check_elliptic[16, 27, 3](t)
    check_elliptic[16, 27, 6](t)
    check_elliptic[64, 48, 4](t)
    check_elliptic[1, 4, 3](t)

    # ⚠ NEGATIVE CONTROL. Every assertion above is an equality between two
    # arithmetic expressions; if `eq` were broken the run would report a clean
    # sweep of nothing. Plant a wrong offset and require it to be caught.
    var probe = Tally()
    probe.eq(ws_c_dist(16), 3 * 16, "planted: c_dist as 3*mc")
    probe.eq(ws_ell_nt(6), 4, "planted: ell_nt(6) as 4")
    if probe.fails != 2:
        print("!! THE CHECKER DOES NOT FAIL ON WRONG INPUT — run is VOID")
        t.fails += 1
    else:
        print("  negative control: 2/2 planted errors caught")

    print("checks:", t.checks, " failures:", t.fails)
    if t.fails == 0:
        print("test_solver_ws_layout: ALL PASS")
    else:
        print("test_solver_ws_layout: FAILED")
