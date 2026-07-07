"""Owned PGS contact-solver workspace tensor (migration P4).

`ContactScratch` is the stateful replacement for the "solver workspace"
section of the flat workspace slab (`ws_solver_offset` in
gpu/constants.mojo): one owned `TensorImpl` holding the per-env PGS contact
workspace, allocated once and reused every step. All offsets in the
consumers (constraints/contact_solve_fields.mojo) are relative to the row
start — the legacy `solver_idx` base is gone.
"""

from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl


struct ContactScratch[
    DTYPE: DType,
    NV: Int,
    MAX_CONTACTS: Int,
    BATCH: Int = 1,
](Movable):
    """PGS contact-solver workspace: one owned tensor, `[BATCH, SOLVER_WS]`
    with `SOLVER_WS = 81*MC + 12*MC*NV` (the legacy
    `PGSSolver.solver_workspace_size`).

    Row layout (offsets relative to the row start; `MC = max(MAX_CONTACTS, 1)`):

    Common normal block (15*MC + 2*MC*NV — constraint_builder_gpu.mojo header):
      [0*MC..1*MC)                  lambda_n   Normal impulses
      [1*MC..2*MC)                  K_n        Effective mass
      [2*MC..3*MC)                  c_dist     Contact distance (- includemargin)
      [3*MC..4*MC)                  c_body     Body A index
      [4*MC..5*MC)                  c_body_b   Body B index
      [5*MC..8*MC)                  c_px/py/pz Contact position
      [8*MC..11*MC)                 c_nx/ny/nz Contact normal
      [11*MC..12*MC)                pos_bias   Impedance position correction
      [12*MC..13*MC)                inv_K_imp  imp/K ratio
      [13*MC..14*MC)                imp_n      Normal impedance (direct R_n)
      [14*MC..15*MC)                diag_n     Body invweight0 diagonal
      [15*MC..15*MC+MC*NV)          J_n        Normal Jacobian
      [15*MC+MC*NV..15*MC+2*MC*NV)  MinvJn     M_inv @ J_n^T

    Friction block at FWS = 15*MC + 2*MC*NV (66*MC + 10*MC*NV — the comptime
    ws_* offsets of the legacy PGSSolver.solve_gpu):
      [FWS+0*MC..FWS+5*MC)          lambda_f[5*MC]
      [FWS+5*MC..FWS+10*MC)         K_f[5*MC]
      [FWS+10*MC..FWS+25*MC)        dir_f[15*MC]
      [FWS+25*MC..FWS+30*MC)        fric_coef[5*MC]
      [FWS+30*MC..FWS+31*MC)        condim[MC]
      [FWS+31*MC..FWS+36*MC)        R_f[5*MC]
      [FWS+36*MC..FWS+41*MC)        bias_f[5*MC]
      [FWS+41*MC..+5*MC*NV)         J_f[5*MC*NV]
      [..+5*MC*NV)                  MinvJ_f[5*MC*NV]
      Pyramidal extras (5*MC each): lambda_edge_neg, C_nt, K_edge_pos,
      K_edge_neg, R_edge

    Total = 15*MC + 2*MC*NV + 66*MC + 10*MC*NV = 81*MC + 12*MC*NV.
    """

    comptime MC = Self.MAX_CONTACTS if Self.MAX_CONTACTS > 0 else 1
    comptime SOLVER_WS = 81 * Self.MC + 12 * Self.MC * Self.NV
    comptime L_SOLVER = Layout.row_major(Self.BATCH, Self.SOLVER_WS)

    var solver: TensorImpl[Self.DTYPE]  # [BATCH, SOLVER_WS]

    def __init__(out self) raises:
        self.solver = TensorImpl[Self.DTYPE].alloc(
            Self.BATCH * Self.SOLVER_WS
        )

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Create the device buffer (once, at setup — contents are produced
        on-device thereafter)."""
        self.solver.upload(ctx)
