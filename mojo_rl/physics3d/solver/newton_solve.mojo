"""Newton contact solve over per-field tensors (migration P4, single-source).

Per-field port of `NewtonSolver.solve_gpu` (solver/newton_solver.mojo:1127)
— arithmetic, iteration order and branch structure verbatim.

⚠ THE SOLVER'S BUDGET IS THE MODEL'S, NOT A CONSTANT. `iterations`,
`tolerance`, `ls_iterations` and `ls_tolerance` come off `<option>` through
`MODEL_META_IDX_SOLVER_*`; `NEWTON_ITER_GPU` (1000) and `LINESEARCH_ITER` (50)
survive only as the CEILINGS a `range()` needs, and `NEWTON_TOL_GPU` only as
the float32 floor under the model's tolerance. They were 200 / 1e-8 / 50 and
were the whole budget until 2026-08-26, so a model shipping
`<option iterations="4">` was run to convergence — a DIFFERENT answer, not a
better one. Standalone solver entry only — NOT wired into
the fields integrators (later slice).

Structural transformation (the only deviation, identical to
constraints/contact_solve.mojo): the legacy kernel is 2D-threaded
(thread_y = contact slot) with barriers; this port SERIALIZES it per env.
The init + normal-precompute parallel phases become
`for contact_tid in range(MC)` loops (matching the legacy launch with
block_dim.y = MC and the helpers' internal `contact_tid < nc` guards); the
friction phase becomes `for contact_tid in range(nc)` (its legacy launch
guard). All phases write disjoint per-contact slots, so serialization is
value-identical. The entire Newton core after the legacy
`if not valid_env or contact_tid != 0: return` gate already runs
single-thread and is ported as-is.

Setup phases reuse the already-ported shared constraint-builder helpers
from contact_solve.mojo (`_init_common_normal_ws`,
`_precompute_contact_normal`, `_precompute_contact_friction`
— the latter two are the shared CG/Newton builders, verbatim ports of
`precompute_contact_normal_gpu` / `precompute_contact_friction_gpu`).

Cone-dependent tails at the exact legacy positions with the legacy
iteration count (SOLVER_ITER_GPU=50):
- ELLIPTIC: after the Newton core, `_limits_env` (port of
  `detect_and_solve_limits_gpu`). Nothing else — joint limits, tendon
  equalities and connect/weld are all rows of the system now.
- PYRAMIDAL: joint limits, dry friction, tendon limits, tendon equalities
  and connect/weld are ALL edge rows INSIDE the Newton optimization.
  Nothing runs after the solve.
Row building is call-site gated `comptime if NEQUALITY > 0` /
`NTENDON > 0` — bit-identical to the unconditional form for zero counts.
Excluded: the legacy `dt` metadata read, whose only consumer was the
(unused-arg) limits call.

Workspace: the legacy Newton scratch is 35*MC + 6*MC*NV floats based at
`ws_solver_offset`. This port keeps the exact layout as row-relative
offsets into the fields `ContactScratch.solver` tensor, which is sized for
PGS (81*MC + 12*MC*NV) — strictly larger, so Newton uses a PREFIX of it
(no new scratch struct). ⚠ The ELLIPTIC region's offsets are no longer written
out here: they are `MAX_CONDIM`-dependent and live in
`solver/elliptic_layout.mojo`, which the PRODUCER
(`_precompute_contact_friction`) and all three consumers share. Worst case is
33*MC + 7*MC*NV at condim 6, still inside the PGS budget.

CONDIM. Both cones carry every tangential row a contact declares. PYRAMIDAL
emits `2*(dim-1)` edge rows, ELLIPTIC one normal row plus `dim-1` tangential
ones with per-direction friction and `R` — the elliptic cone math is in
`solver/elliptic_cone.mojo` and is written in MuJoCo's U-space so it does not
assume the tangential rows share a coefficient. Until 2026-08-13 the elliptic
path hard-coded two tangents and one isotropic `mu`, i.e. condim 3 whatever the
geoms declared.

Operands (20): the 19 of `solve_contacts` + `M` (the Newton core
reads the mass matrix for the Gauss term / Hessian; legacy `ws_M_offset`).
The legacy `ws_fnet_offset` comptime was declared but never read — dropped.
"""

from std.math import sqrt, pow, abs
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from .je_budget import je_spills, newton_block_threads
from std.sys.info import size_of
from max.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, LayoutTensor

from ..types import _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_FREE, JNT_BALL
from .cholesky import (
    chol_factor_inline, chol_solve_inline, chol_factor_seg, chol_solve_seg,
    chol_solve_seg_p,
)
from .newton_blocks import build_dof_segments, build_dof_segments_p

# MuJoCo's `mjMINVAL`; see `cholesky.mojo` on why `1e-10` was not the
# reference's number for this guard.
comptime _CHOL_MJMINVAL: Float64 = 1e-15
from ..constraints.solver_ws import (
    ws_budget,
    _max_one_rt,
    ws_c_dist as sw_c_dist,
    ws_pos_bias as sw_pos_bias,
    ws_j_n as sw_j_n,
    ws_ell_jt as sw_ell_jt,
    ws_ell_mu as sw_ell_mu,
    ws_ell_dn as sw_ell_dn,
    ws_ell_dt as sw_ell_dt,
    ws_ell_fr as sw_ell_fr,
    ws_ell_bt as sw_ell_bt,
    ws_ell_ntc as sw_ell_ntc,
    ws_end_elliptic,
)
from .noslip import noslip_pyramidal, noslip_elliptic
from ..constraints.elliptic_layout import (
    ell_nt,
    ell_jt,
    ell_end,
    ell_mu,
    ell_dn,
    ell_dt,
    ell_fr,
    ell_bt,
    ell_ntc,
)
from .elliptic_cone import (
    ell_state_force,
    ell_row_cost,
    ell_hessian_block,
    ell_add_contact_hessian,
    ell_line_deriv,
    ELL_SATISFIED,
    ELL_QUADRATIC,
    ELL_CONE,
)

# `mjModel.opt.noslip_tolerance`, MuJoCo's default — the value used when a
# model's `<option>` does not set the attribute.
#
# ⚠ THIS IS THE FALLBACK, NOT THE VALUE. It was the value until 2026-08-13, on
# the reasoning that no ported model overrode it; dm_control's manipulation
# models all do, with `noslip_tolerance="0"` ("run every iteration"). The real
# number now arrives per-model in `MODEL_META_IDX_NOSLIP_TOLERANCE`; read it
# from there, and note that a 0 read out of META is a SETTING, never "unset"
# to be replaced by this.
#
# ⚠ NO FIXTURE DISTINGUISHES 0 FROM 1e-6 TODAY — measured 8.9e-10 worst, and
# exactly 0.0 on `reach_site_features`. This is a fidelity fix, not a measured
# bug fix; see `_parse_option` in `parser/full_parser.mojo` for the numbers and
# for the confounded experiment that first, wrongly, said otherwise.
comptime NOSLIP_TOLERANCE: Float64 = 1e-6
from .primal import pyramidal_edge_forces, pyramidal_linesearch
from ..constraints.contact_solve import (
    _init_common_normal_ws,
    _precompute_contact_normal,
    _precompute_contact_friction,
)
from ..constraints.limits import _limits_env
from ..constraints.friction_dof import _friction_env
from ..constraints.tendon_limit import (
    build_tendon_limit_rows,
    build_tendon_equality_rows,
)
from ..constraints.scalar_rows import (
    build_scalar_rows,
    max_scalar_rows,
    max_scalar_rows_cap,
    scalar_row_state,
    scalar_row_force,
    scalar_row_cost,
    SROW_QUADRATIC,
    SROW_LIMIT,
    SROW_FRICTION,
    SROW_EQ_BILATERAL,
    DOF_SOLREF_TIMECONST,
    DOF_SOLIMP_DMIN,
    DOF_SOLIMP_DMAX,
)
from ..constraints.equality_tendon import build_weld_equality_rows
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
    DimsLike,
    AsStatic,
    may_exist,
    Scratch,
    cap,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_TREE_SIZE,
    MODEL_META_IDX_NTREE,
    METADATA_SIZE,
    CONTACT_SIZE,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_MEANINERTIA,
    MODEL_META_IDX_NOSLIP_TOLERANCE,
    MODEL_META_IDX_NOSLIP_ITERATIONS,
    MODEL_META_IDX_WARMSTART_DISABLED,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_SOLVER_ITERATIONS,
    MODEL_META_IDX_SOLVER_TOLERANCE,
    MODEL_META_IDX_LS_ITERATIONS,
    MODEL_META_IDX_LS_TOLERANCE,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
)

# One env per BLOCK (not 64 threads/block). The per-env Newton solve stack-
# allocates a large local frame (~ ME*NV + 3*MC*NV + several NV*NV, ~60KB for
# humanoid). With a wide block every thread — including the idle ones past
# BATCH — reserves that frame, and CUDA reserves it for max residency across
# the device, which OOMs at humanoid scale (Metal doesn't pre-reserve). One
# thread per block keeps the reservation to the envs actually running.
from ..constraints.constraint_data import refsafe_timeconst, solref_spring_damper

comptime NS_TPB: Int = 1


# =============================================================================
# Cooperative (one-env-per-block) helpers — verbatim ports of the shared-memory
# helpers in newton_solver.mojo (chol_factor_coop_gpu:174, matvec_mv_jve_coop:484,
# recompute_jfq_coop:546). They operate purely on SHARED-memory LayoutTensors, so
# the port is a straight copy — no slab/field addressing appears in them. @no_inline
# keeps their nested loops OUT of the giant blocked kernel (Mojo inline-explosion
# guard). Used only by the PYRAMIDAL blocked path.
# =============================================================================


# Per-iteration trace of the ELLIPTIC Newton — `scale*|grad|`, how many contacts
# sit in the CONE zone, and the accepted `alpha`. CPU only (a Metal kernel
# cannot `print`), and off in every committed state.
#
# ⚠ THE SHAPE OF THE TRACE IS THE DIAGNOSIS. A stale Hessian does not diverge
# and does not fail a tolerance — it makes `alpha` alternate between two values
# while the gradient creeps down a few percent per PAIR of iterations. No
# aggregate (final `qacc`, a residual, an iteration count) shows that; the
# per-iteration `alpha` column shows it immediately. Board row `unitree_go1`
# was mis-attributed to the elliptic cone's algebra for want of these numbers.
comptime _ELL_TRACE: Bool = False
# Per-iteration trace of the PYRAMIDAL Newton — the accepted `alpha`, the
# improvement, and the gradient AFTER the update. Off in every committed state.
#
# ⚠ THE `alpha` AND `impr` COLUMNS TOGETHER ARE THE DIAGNOSIS. A line search
# whose second-derivative floor binds does not fail and does not diverge: it
# returns a TINY alpha with `improvement` exactly 0.0 while the gradient
# plateaus just above `tolerance`. No aggregate shows that.
comptime _PYR_TRACE: Bool = False

# ⚠⚠ A PRICING KNOB, AND IT IS BIT-IDENTICAL AT EVERY VALUE — that is the only
# reason it is safe to ship. It divides the STRIDE of the blocked kernel's
# cooperative passes without touching `block_dim`, so at `d > 1` several
# threads recompute the same element and store it again. Every cooperative pass
# in this kernel is a PURE STORE of a value derived from reads that are already
# final at that point (audited: the `M_sh` load, the `H` build, both loops of
# `_chol_factor_coop`, both of `_matvec_mv_jve_coop`, both of
# `_recompute_jfq_coop`) — never a read-modify-write — so the duplicates write
# the SAME BITS to the same address and the answer cannot move. `barrier()`
# positions are untouched.
#
# WHY IT EXISTS. `THREADS = MAX_CONTACTS = 16` is fixed by the launch shape,
# so the kernel cannot be run at MORE threads without a guard audit
# (see `:755`). It can be run at FEWER, and that is enough to price the audit
# before paying for it: with `t(p) = S + P/p`, measuring `d=1` (p=16) and
# `d=16` (p=1) gives S and P in closed form. If S dominates, raising the launch
# is capped at `P` no matter how many threads it buys, and the work belongs on
# the tid-0 serial floor instead.
#
# 1 = production. Anything else is a measurement build.
comptime NEWTON_COOP_DIV: Int = 1

# ⚠⚠ THE SERIAL PROBE — the same trick as `NEWTON_COOP_DIV`, pointed at the
# tid-0 floor that knob proved was 90% of newton's excess. It answers "how much
# does THIS term cost" by MEASURING, and it exists because op counts have now
# over-predicted time-saved four times running (PN2c, P2, F2, F3b). Op counts
# get RATIOS BETWEEN CONCURRENT TERMS right — the COOP_DIV split was predicted
# 11% and measured 9.6% — and get TIME SAVED wrong, because serial tid-0 code
# is latency-bound on dependent shared loads, not throughput-bound.
#
# HOW: at `NEWTON_SERIAL_PROBE == N` the selected term runs
# `SERIAL_PROBE_REPEAT - 1` EXTRA times BEFORE its real, untouched instance.
# `t(probe) - t(baseline)` over `REPEAT-1` is the term's marginal cost.
#
# ⚠ THE EXTRA COPIES GO BEFORE, NEVER AFTER, AND THAT IS THE SAFETY ARGUMENT.
# The real instance runs last and overwrites whatever the copies wrote, so the
# state entering the next statement is exactly production's. Each extra block is
# also written as a PURE RECOMPUTE — no in-place mutation, no accumulator, no
# `ctrl_sh` side effect that is not idempotent — so it cannot perturb its own
# inputs. (This is why term 2 repeats only the solve and not the `-search_sh`
# negate, which flips sign every time it runs.)
#
# ⚠ AND AT `== 0` THE `comptime if` ELIDES ENTIRELY, so the production build is
# codegen-identical, not merely bit-identical. A `for _ in range(1)` wrapper
# would NOT have been: it can move register allocation, and a measurement tool
# whose baseline arm is perturbed measures itself.
#
# ⚠ NO `barrier()` MAY APPEAR IN AN EXTRA BLOCK. Every site below sits inside an
# existing `if tid == 0` or block-uniform `if valid_env`, so barrier positions
# are untouched — but a future term that needs one is NOT probeable this way.
#
# ⚠ THESE TERMS ARE ALL SHARED-MEMORY RESIDENT, which is what makes the repeat
# UNBIASED. Repeating a GLOBAL-memory term would measure the warm cost and
# under-report it; shared memory has no cache below it, so pass two costs what
# pass one did.
#
# ⚠⚠ 5-8 ARE THE SETUP, WHICH THE MIN_ITER SWEEP MEASURED AT 94.8% OF NEWTON.
# The loop (terms 1-4) is 5.2%; 1-4 together account for 11% of newton, so the
# setup is where the rest is. These four are the parts of it that can be run
# again without changing the answer. The FIFTH part — the `for j in
# range(NJOINT)` row builder — cannot: it ACCUMULATES `num_edges`, and making it
# idempotent means restructuring 300 lines. It is measured BY SUBTRACTION, the
# same way the 89% was found: whatever the setup total is minus terms 5-8.
#
#   0 = off (production)     3 = the `d_j` reduction inside _chol_factor_coop
#   1 = the gradient loop    4 = the Mv/search read-back loop
#   2 = the per-block solve
#   --- setup (once per solve, not per iteration) ---
#   5 = the workspace init + edge zeroing   (runs for all MC slots even at nc=0)
#   6 = the contact normal/friction precompute
#   7 = the `M_sh` cooperative load          (NV^2 read from global)
#   8 = `Ma = M*qacc` + f_smooth             (tid-0, sum(bn^2) after PN2e)
#   --- inside the Newton LOOP (60.8% of newton, internally unmeasured) ---
#   9 = ONE `_bl_peval`  (the line search's unit; x [lseval] = its total)
#  10 = the H build
comptime NEWTON_SERIAL_PROBE: Int = 0
comptime SERIAL_PROBE_REPEAT: Int = 10

# ⚠⚠ THE ITERATION PROBE — a DIFFERENT INSTRUMENT, and the difference is the
# point. `NEWTON_SERIAL_PROBE` answers "what does this term cost"; it cannot
# answer "is that cost paid once per SOLVE or once per ITERATION", and after the
# serial sweep left 89% of newton unaccounted, that split is what decides where
# to look next. Sweeping a forced MINIMUM iteration count answers it in closed
# form: `t(N)` is FLAT while N is below the count the solver takes anyway and
# LINEAR above it, so
#
#     the KNEE      = the iteration count actually taken (never measured here)
#     the SLOPE     = the per-iteration cost
#     the INTERCEPT extrapolated back to N=0 = the SETUP cost
#
# ⚠ A MINIMUM, NOT A CAP, AND THAT IS A SAFETY DECISION. Capping iterations
# would leave the solve UNCONVERGED, and over 500 steps an unconverged contact
# solver does not merely report a wrong number — the scene diverges, contact
# counts change, and the WORKLOAD stops being the one under test. Forcing EXTRA
# iterations on an already-converged system is the safe direction: the line
# search returns alpha ~ 0, nothing moves, and the answer changes only in its
# last bits. The measurement stays honest because the extra iterations cost full
# price whether or not they achieve anything.
#
# ⚠ NOT bit-exact — the only knob here that is not. It is a timing instrument
# and must never ship at a non-zero value.
#
# ⚠ Both edits are `comptime`-elided at 0, so production codegen is untouched,
# and `NEWTON_MIN_ITER` is comptime-uniform across the threadgroup — it cannot
# desynchronise the loop's barriers, which is the property the two existing
# breaks are documented to preserve.
comptime NEWTON_MIN_ITER: Int = 0

# ⚠⚠ COUNT THE ITERATIONS AND PRINT THEM. This exists because the iteration
# count has now confounded TWO probes, and both times it was INFERRED rather
# than read: the serial sweep divided by an "order 3x" growth taken from two
# terms whose k=0 deltas sat at the resolution limit, and the MIN_ITER sweep
# came back FLAT from 20 to 180 — which no model of this loop produces, since
# `niter_rt` is 100 (measured off the built model, the scenes carry no
# `<option>`) and 180 iterations cannot cost what 20 do while each one still
# factors an 84x84 Cholesky.
#
# The lesson is the cheap one: an indirect instrument for a number you can
# simply COUNT is a false economy. Two sweeps — about 40 minutes of somebody's
# GPU — against one print.
#
# One line per SOLVE for env 0 (`[niter] <count>`), which the attribution
# script already captures into `k*.probe.txt`. At RK4 x FRAME_SKIP that is 8
# lines per step: verbose ON PURPOSE, because the DISTRIBUTION is the answer
# and a mean would hide it. A solver that takes 3 iterations on most steps and
# 100 on a few has a different problem from one that always takes 50.
#
# ⚠ PROBE ONLY, AND NEVER READ A DURATION FROM A RUN WITH IT ON — a `print` per
# solve serialises the block. Read the counts, then turn it off and re-time.
comptime NEWTON_ITER_REPORT: Bool = False

# ⚠⚠ STAGE BISECT OF THE SETUP. The MIN_ITER sweep put the Newton LOOP at 5.2%
# of newton and the per-solve SETUP at 94.8%; the repeat probe then accounted
# for only ~8.6% of it and left ~19 ms/step at k=13 unlocated. Repetition cannot
# find the rest, and the reason is on record: three of those four terms touch
# GLOBAL memory, where repeating a term measures its WARM cost and under-reports
# it. So this is a different instrument — cumulative timing by early return.
#
#   1 = after the workspace init + edge zeroing
#   2 = after the contact normal/friction precompute
#   3 = after the cooperative M_sh / Je loads   (i.e. before the tid-0 setup)
#   4 = after the whole tid-0 setup             (i.e. before the Newton loop)
#   5 = after the Newton LOOP                   (i.e. before the write-back tail)
#
# `t(4) - t(3)` is the 660-line tid-0 serial block, `t(3) - t(2)` the
# cooperative loads, and so on. `t(4)` should land near the ~20 ms/step the
# MIN_ITER sweep attributes to setup — **that agreement is the instrument's own
# validity check**, and if it fails the bisect is measuring something else.
#
# ⚠ WHY AN EARLY RETURN IS SAFE HERE, WHEN CAPPING ITERATIONS WAS NOT. On entry
# `qacc_constrained` already holds the SMOOTH (unconstrained) acceleration —
# the tid-0 setup reads it as `qacc`/`qacc_smooth` at the top. Returning early
# therefore leaves exactly the answer the kernel itself writes when `nefc == 0`,
# which it documents as a no-op rather than an approximation. The sim runs on
# with no constraint forces: WRONG, but stable and bounded, not divergent.
#
# ⚠ THE CONTROLS ARE THE OTHER CHECK. If `collision` moves between stages the
# workload has changed and the whole comparison is void — an unconstrained sim
# is still a DIFFERENT sim, and only its stability is being relied on here.
#
# ⚠ Every return is comptime-selected and unconditional, so all threads take it
# together and no `barrier()` is left half-reached. Timing instrument only.
comptime NEWTON_STOP_AFTER: Int = 0

# ⚠⚠ WITHOUT THIS THE PROBE MEASURES NOTHING AND SAYS SO CONVINCINGLY. Every
# extra block writes memory the real pass overwrites on the very next lines, so
# dead-store elimination is entitled to delete the whole thing — and the probe
# would then report ~0 for every term, which reads as "these terms are free"
# rather than as "the probe did not run". So each block folds its work into a
# checksum and consumes it against a value it cannot produce, which the compiler
# cannot prove unreachable: the stores stay live, and so do the loads and the
# arithmetic that are the cost being measured.
#
# ⚠ THE NON-VACUITY CHECK IS THE SWEEP ITSELF: at `REPEAT = 10` a term that is
# 5% of newton must move newton by ~45%. **A term that comes back inside the
# 1.7% noise floor has either been optimised away or is genuinely free, and
# those are NOT the same answer** — re-run that term at `REPEAT = 100` before
# believing it is free.
@always_inline
def _probe_sentinel[DTYPE: DType]() -> Scalar[DTYPE]:
    """A value the checksums above cannot produce, in the kernel's own dtype.
    One definition, four call sites — the constant must not drift between them,
    and a per-site literal is exactly how that happens."""
    return Scalar[DTYPE](-1.0e30)


# =============================================================================


@no_inline

def _chol_factor_coop[
    DTYPE: DType,
    D: DimsLike,
    L_H_SH: Layout,
    L_CTRL_SH: Layout,
    L_SEG: Layout,
](
    tid: Int,
    n_threads: Int,
    dims: D,
    H_sh: LayoutTensor[
        DTYPE,
        L_H_SH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    L_sh: LayoutTensor[
        DTYPE,
        L_H_SH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    ctrl_sh: LayoutTensor[
        DTYPE,
        L_CTRL_SH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    # ⚠ THE DIAGONAL BLOCKS. `seg0[j]`/`seg1[j]` are the half-open dof range
    # of the block containing column j; a whole-matrix partition (one segment
    # spanning [0, nv)) reproduces the previous body exactly.
    seg0: LayoutTensor[
        DTYPE, L_SEG, MutAnyOrigin, address_space=AddressSpace.SHARED,
    ],
    seg1: LayoutTensor[
        DTYPE, L_SEG, MutAnyOrigin, address_space=AddressSpace.SHARED,
    ],
):
    """Cooperative column-parallel Cholesky of shared H_sh -> L_sh (verbatim
    from chol_factor_coop_gpu). Bit-identical to chol_factor_inline."""
    var nv = dims.get_nv()
    for _attempt in range(2):
        if tid == 0:
            ctrl_sh[2] = Scalar[DTYPE](0)
        # ⚠⚠ PER BLOCK, AND HERE THE ZERO IS LOAD-BEARING — unlike `H_sh`.
        # It is what makes the segmented factor equal the dense one: the terms
        # the restricted `k` loops drop are `L[.,k]*L[.,k]` outside the block,
        # and the dense version reads them AS ZERO. Within a block that
        # property is preserved exactly. Outside, nothing reads `L_sh` at all
        # — audited: every read is `[j*nv+k]`, `[i*nv+k]`, `[j*nv+j]` inside
        # the factor's segment-restricted loops, and the tid-0 copy, which
        # PN2c already restricted to the blocks.
        var zp = 0
        while zp < nv:
            var ze = Int(rebind[Scalar[DTYPE]](seg1[zp]))
            if ze <= zp:
                ze = nv
            for q in range(tid, (ze - zp) * (ze - zp), n_threads):
                L_sh[(zp + q // (ze - zp)) * nv + zp + q % (ze - zp)] = (
                    Scalar[DTYPE](0)
                )
            zp = ze
        barrier()
        for j in range(nv):
            # ⚠ THE BLOCK OF COLUMN j. The OUTER loop and both barriers are
            # untouched, so the cooperative schedule — and the per-column
            # bit-identity with `chol_factor_inline` — is exactly what it was;
            # only the two inner ranges shrink. Every term dropped is
            # `L[.,k] * L[.,k]` with k outside the block, where L is exactly 0
            # (zeroed above, never written, since no segment owns that column),
            # so a sequential accumulation returns the identical bits.
            var b0 = Int(rebind[Scalar[DTYPE]](seg0[j]))
            var b1 = Int(rebind[Scalar[DTYPE]](seg1[j]))
            if tid == 0:
                comptime if NEWTON_SERIAL_PROBE == 3:
                    # Pure recompute of the reduction. It reads `L_sh` columns
                    # strictly BELOW j, which are final at this point and which
                    # it does not write, so repeating it cannot move its own
                    # input. The `ctrl_sh[2]` flag and `L_sh[j*nv+j]` write are
                    # left to the real pass below.
                    for _r in range(SERIAL_PROBE_REPEAT - 1):
                        var p_sd: Scalar[DTYPE] = 0
                        for k in range(b0, j):
                            var p_l = rebind[Scalar[DTYPE]](L_sh[j * nv + k])
                            p_sd += p_l * p_l
                        # Consume it so the reduction cannot be folded away.
                        if p_sd == _probe_sentinel[DTYPE]():
                            ctrl_sh[2] = Scalar[DTYPE](0)
                var s_d: Scalar[DTYPE] = 0
                for k in range(b0, j):
                    var ljk = rebind[Scalar[DTYPE]](L_sh[j * nv + k])
                    s_d += ljk * ljk
                var diag = rebind[Scalar[DTYPE]](H_sh[j * nv + j]) - s_d
                # `mjMINVAL`, matching `chol_factor_inline` — see the long
                # note in `cholesky.mojo`. This third copy exists because the
                # cooperative GPU factorization is documented as BIT-IDENTICAL
                # to that one, and a threshold that drifted between them would
                # break exactly that property.
                if diag < Scalar[DTYPE](_CHOL_MJMINVAL):
                    ctrl_sh[2] = Scalar[DTYPE](1)
                    diag = Scalar[DTYPE](_CHOL_MJMINVAL)
                L_sh[j * nv + j] = sqrt(diag)
            barrier()
            var ljj = rebind[Scalar[DTYPE]](L_sh[j * nv + j])
            for i in range(j + 1 + tid, b1, n_threads):
                var s: Scalar[DTYPE] = 0
                for k in range(b0, j):
                    s += rebind[Scalar[DTYPE]](L_sh[i * nv + k]) * rebind[
                        Scalar[DTYPE]
                    ](L_sh[j * nv + k])
                L_sh[i * nv + j] = (
                    rebind[Scalar[DTYPE]](H_sh[i * nv + j]) - s
                ) / ljj
            barrier()
        if Int(rebind[Scalar[DTYPE]](ctrl_sh[2])) == 0:
            break
        # Rank-deficient: add 1e-6 to the H diagonal and refactor once.
        if tid == 0:
            for i in range(nv):
                H_sh[i * nv + i] += Scalar[DTYPE](1e-6)
        barrier()


@no_inline
def _matvec_mv_jve_coop[
    DTYPE: DType,
    D: DimsLike,
    L_M_SH: Layout,
    L_JE_SH: Layout,
    L_SEARCH_SH: Layout,
    L_JV_E_SH: Layout,
    L_SEG: Layout,
    JE_AS: AddressSpace = AddressSpace.SHARED,
](
    tid: Int,
    n_threads: Int,
    num_edges: Int,
    dims: D,
    M_sh: LayoutTensor[
        DTYPE,
        L_M_SH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    # ⚠ `Je` is the ONE array whose address space varies — see JE_IN_SHARED at
    # the allocation site. Everything else stays in threadgroup memory.
    Je_sh: LayoutTensor[
        DTYPE,
        L_JE_SH,
        MutAnyOrigin,
        address_space=JE_AS,
    ],
    search_sh: LayoutTensor[
        DTYPE,
        L_SEARCH_SH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    Mv_sh: LayoutTensor[
        DTYPE,
        L_SEARCH_SH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    Jv_e_sh: LayoutTensor[
        DTYPE,
        L_JV_E_SH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    seg0: LayoutTensor[
        DTYPE, L_SEG, MutAnyOrigin, address_space=AddressSpace.SHARED,
    ],
    seg1: LayoutTensor[
        DTYPE, L_SEG, MutAnyOrigin, address_space=AddressSpace.SHARED,
    ],
):
    """Cooperative Mv = M·search and Jv_e = Je·search (verbatim from
    matvec_mv_jve_coop). Ascending inner sums → bit-identical.

    ⚠ `Mv` IS RESTRICTED TO ROW i'S BLOCK, and that is exact rather than
    approximate: `M`'s off-tree entries are STRUCTURALLY zero — both CRBA
    paths only ever write within a tree — and a segment is a UNION of trees,
    so `[seg0[i], seg1[i])` is a superset of row i's nonzeros. Dropping exact
    zeros from an ascending sum leaves the bits unchanged. `Jv_e` below is a
    row sweep over `Je` and has no block structure to use."""
    var nv = dims.get_nv()
    for i in range(tid, nv, n_threads):
        var s: Scalar[DTYPE] = 0
        var j0 = Int(rebind[Scalar[DTYPE]](seg0[i]))
        var j1 = Int(rebind[Scalar[DTYPE]](seg1[i]))
        if j1 <= j0:
            j0 = 0
            j1 = nv
        for j in range(j0, j1):
            s += rebind[Scalar[DTYPE]](M_sh[i * nv + j]) * rebind[
                Scalar[DTYPE]
            ](search_sh[j])
        Mv_sh[i] = s
    for e in range(tid, num_edges, n_threads):
        var s: Scalar[DTYPE] = 0
        for i in range(nv):
            s += rebind[Scalar[DTYPE]](Je_sh[e * nv + i]) * rebind[
                Scalar[DTYPE]
            ](search_sh[i])
        Jv_e_sh[e] = s


@no_inline
def _recompute_jfq_coop[
    DTYPE: DType,
    D: DimsLike,
    L_JE_SH: Layout,
    L_DE_SH: Layout,
    L_QACC_SH: Layout,
    JE_AS: AddressSpace = AddressSpace.SHARED,
](
    tid: Int,
    n_threads: Int,
    num_edges: Int,
    dims: D,
    # ⚠ address space varies — see JE_IN_SHARED at the allocation site.
    Je_sh: LayoutTensor[
        DTYPE, L_JE_SH, MutAnyOrigin,
        address_space=JE_AS,
    ],
    De_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    bias_e_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    kind_e_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    R_e_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    floss_e_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    state_e_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    qacc_sh: LayoutTensor[
        DTYPE, L_QACC_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    jar_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    force_sh: LayoutTensor[
        DTYPE, L_DE_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    qfrc_sh: LayoutTensor[
        DTYPE, L_QACC_SH, MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
):
    """Cooperative jar/force/qfrc recompute (verbatim from recompute_jfq_coop).
    Two phases separated by a barrier; ascending inner sums → bit-identical."""
    var nv = dims.get_nv()
    for e in range(tid, num_edges, n_threads):
        var j = rebind[Scalar[DTYPE]](bias_e_sh[e])
        for i in range(nv):
            j += rebind[Scalar[DTYPE]](Je_sh[e * nv + i]) * rebind[
                Scalar[DTYPE]
            ](qacc_sh[i])
        jar_sh[e] = j
        var st = scalar_row_state[DTYPE](
            Int(rebind[Scalar[DTYPE]](kind_e_sh[e])),
            j,
            rebind[Scalar[DTYPE]](R_e_sh[e]),
            rebind[Scalar[DTYPE]](floss_e_sh[e]),
        )
        state_e_sh[e] = Scalar[DTYPE](st)
        force_sh[e] = scalar_row_force[DTYPE](
            st, j, rebind[Scalar[DTYPE]](De_sh[e]),
            rebind[Scalar[DTYPE]](floss_e_sh[e]),
        )
    barrier()
    for i in range(tid, nv, n_threads):
        var q: Scalar[DTYPE] = 0
        for e in range(num_edges):
            q += rebind[Scalar[DTYPE]](Je_sh[e * nv + i]) * rebind[
                Scalar[DTYPE]
            ](force_sh[e])
        qfrc_sh[i] = q


# =============================================================================
# Newton contact solve — single-source per-env body (port of
# NewtonSolver.solve_gpu)
# =============================================================================


@always_inline
def _newton_solve_env[
    DTYPE: DType,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
    D: DimsLike,
    L_QPOS: Layout,
    L_QVEL: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_CONTACTS: Layout,
    L_SMETA: Layout,
    L_JOINTS: Layout,
    L_BODIES: Layout,
    L_MMETA: Layout,
    L_TREES: Layout,
    L_EQUALITY: Layout,
    L_TENDONS: Layout,
    L_SITES: Layout,
    L_GEOMS_W: Layout,
    L_BODY_INVWEIGHT0: Layout,
    L_DOF_INVWEIGHT0: Layout,
    L_CDOF: Layout,
    L_M: Layout,
    L_SOLVER: Layout,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
    # ⚠ CPU ONLY (the per-env dispatcher passes True). Under it the PYRAMIDAL
    # path walks each row's nonzero dofs and factors `H` per kinematic-tree
    # segment — the arithmetic the blocked GPU kernel got in PN2a-e, and that
    # this function never did: `PERFORMANCE.md` §13 measured the dense walks
    # at 60-75% of every step past 20 dofs. Bit-exact by the exact-zero
    # argument in `cholesky.chol_factor_seg`. Off, the body is byte-identical
    # to what every GPU leg compiles today, which is why the GPU legs keep
    # the default rather than take a per-thread index list they cannot afford.
    TREE_AWARE: Bool = False,
](
    env: Int,
    dims: D,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    xpos: LayoutTensor[
        DTYPE, L_XPOS, MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, L_XQUAT, MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, L_XPOS, MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE,
        L_CONTACTS,
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, L_SMETA, MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, L_MMETA, MutAnyOrigin
    ],
    # ⚠ `Model.trees`, FLAT — `[t*MODEL_TREE_SIZE + col]`, matching
    # `newton_blocks.build_dof_segments`. `H`'s diagonal blocks are the
    # kinematic trees merged by whatever a constraint row couples; see that
    # module. The live row count is `mmeta[MODEL_META_IDX_NTREE]`, not `NV`.
    trees: LayoutTensor[
        DTYPE, L_TREES, MutAnyOrigin
    ],
    equality: LayoutTensor[
        DTYPE, L_EQUALITY, MutAnyOrigin
    ],
    tendons: LayoutTensor[
        DTYPE, L_TENDONS, MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, L_SITES, MutAnyOrigin
    ],
    # ⚠ FOR TENDON WRAP GEOMS ONLY — see `dynamics/tendon._geom_world_frame`.
    # Named `..._W` so it cannot be confused with the CONTACT geom tensors,
    # which this solver does not take (contacts arrive pre-detected).
    geoms_w: LayoutTensor[
        DTYPE, L_GEOMS_W, MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, L_BODY_INVWEIGHT0, MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, L_DOF_INVWEIGHT0, MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    M: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, L_M, MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, L_QVEL, MutAnyOrigin
    ],
    # `mjData.qacc_warmstart` — the previous `mj_forward`'s constrained
    # acceleration. READ ONLY here; `solver/warmstart.mojo` writes it.
    qacc_warmstart: LayoutTensor[
        DTYPE, L_QVEL, MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, L_SOLVER, MutAnyOrigin
    ],
):
    """Full primal Newton contact solve for one env (verbatim from
    NewtonSolver.solve_gpu, serialized per env — see module docstring)."""
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var max_contacts = dims.get_max_contacts()
    var ngeom = dims.get_ngeom()
    var nequality = dims.get_nequality()
    var ntendon = dims.get_ntendon()
    var nsite = dims.get_nsite()
    # ⚠ CAPS SIZE CONTAINERS ONLY. Strides, loop bounds and capacity guards
    # all read the live `nv` / `max_contacts` above. On the static leg the cap
    # and the live value are the same integer, so nothing in the 124-file
    # suite can tell a mix-up from correct code.
    comptime MC_CAP = cap[D.CAP_MAX_CONTACTS]()
    comptime V_CAP = cap[D.CAP_NV]()
    comptime M_CAP = cap[D.CAP_NV * D.CAP_NV]()

    # Common normal block offsets (row-relative; the legacy `solver_ws_idx`
    # base is gone)
    var ws_c_dist_idx = sw_c_dist(max_contacts)
    var ws_pos_bias_idx = sw_pos_bias(max_contacts)
    var ws_J_n_idx = sw_j_n(max_contacts)

    # Primal-specific offsets (after common normal block). ⚠ ONE SOURCE OF
    # TRUTH — `solver/elliptic_layout` — because the region is now
    # `MAX_CONDIM`-dependent and the producer indexes the same slots. `NT` is
    # the tangential rows per contact: 2 at condim 3, 3 at 4, 5 at 6.
    comptime NT = ell_nt[MAX_CONDIM]()
    var ws_Jt_idx = sw_ell_jt(max_contacts, nv)
    var ws_mu_idx = sw_ell_mu(max_contacts, nv, MAX_CONDIM)
    var ws_D_n_idx = sw_ell_dn(max_contacts, nv, MAX_CONDIM)
    var ws_Dt_idx = sw_ell_dt(max_contacts, nv, MAX_CONDIM)
    var ws_fr_idx = sw_ell_fr(max_contacts, nv, MAX_CONDIM)
    var ws_bt_idx = sw_ell_bt(max_contacts, nv, MAX_CONDIM)
    var ws_ntc_idx = sw_ell_ntc(max_contacts, nv, MAX_CONDIM)
    # ⚠ THE ONE FAILURE MODE THIS LAYOUT HAS IS OVERRUNNING THE ROW, and it
    # would not crash: `solver` is `[BATCH, row]`, so writing past the row
    # lands in the NEXT ENV's workspace.
    #
    # ⚠⚠ COMPARE AGAINST THE ROW THAT WAS ACTUALLY ALLOCATED, NOT THE COMPTIME
    # `SOLVER_WS`. This read the comptime parameter until 2026-08-19, and on a
    # DYNAMIC provider that parameter is `81*MC + 12*MC*D.NV` with
    # `D.NV = DIM_POISON = -1` and `MC` floored to 1 — i.e. **69 scalars for
    # every model**, the same 69 that `ContactScratch` allocated before its own
    # fix. The guard therefore fired on every runtime-loaded model and took the
    # `return` below, so `_newton_solve_env` COMPUTED NO CONTACT FORCE AT ALL:
    # a sphere dropped on a plane fell straight through to -43.87 m (exactly
    # free fall) while the identical model under PGS matched MuJoCo to six
    # digits, and the studio — which is the runtime-dims path — could not use
    # MuJoCo's DEFAULT solver at all.
    #
    # ⚠ IT PRINTED "FATAL" ON EVERY STEP AND STILL WENT UNNOTICED, because the
    # message names a `MAX_CONDIM` cause that had nothing to do with it and the
    # symptom read as a physics bug. `ws_budget` is the SAME formula
    # `ContactScratch.__init__` allocates with, so the guard now cannot
    # disagree with the buffer on either leg.
    var solver_row = ws_budget(_max_one_rt(max_contacts), nv)
    if ws_end_elliptic(max_contacts, nv, MAX_CONDIM) > solver_row:
        print(
            "FATAL: the ELLIPTIC contact region (",
            ws_end_elliptic(max_contacts, nv, MAX_CONDIM),
            ") does not fit ContactScratch.solver (", solver_row,
            ") at max_contacts", max_contacts, "nv", nv,
            "MAX_CONDIM", MAX_CONDIM,
        )
        return

    # === Initialize workspace (legacy: parallel, one thread per slot; the
    # legacy `contact_tid < MC` guard is vacuous with block_dim.y = MC) ===
    #
    # The `jar_*` / `f*` / `cstate` slots that used to be zeroed here are gone:
    # they were written by nothing and read by nothing after this loop (the
    # solve keeps that state in InlineArrays), and the tangent Jacobian region
    # now extends over the two `MinvJt` blocks they followed.
    #
    # ⚠ THE ROW COUNT IS CONE-SPECIFIC. A pyramidal contact owns `2*(dim-1)`
    # Jacobian blocks, an elliptic one `dim-1`; zeroing the elliptic count on
    # the pyramidal path would leave half the edge list holding the previous
    # step's Jacobian for a slot the producer skips. The old loop zeroed a
    # fixed FOUR blocks and got away with it only because the pyramidal
    # producer re-zeros every edge itself.
    comptime NZ = 2 * NT if CONE_TYPE == ConeType.PYRAMIDAL else NT
    for contact_tid in range(max_contacts):
        _init_common_normal_ws[
            DTYPE](env, contact_tid, dims, solver)
        # Zero primal workspace for this contact slot
        for t in range(NZ):
            for d in range(nv):
                solver[env, ws_Jt_idx + t * max_contacts * nv + contact_tid * nv + d] = 0
        comptime if CONE_TYPE == ConeType.ELLIPTIC:
            for t in range(NT):
                solver[env, ws_Dt_idx + t * max_contacts + contact_tid] = 0
                solver[env, ws_fr_idx + t * max_contacts + contact_tid] = 0
                solver[env, ws_bt_idx + t * max_contacts + contact_tid] = 0
            solver[env, ws_mu_idx + contact_tid] = 0
            solver[env, ws_D_n_idx + contact_tid] = 0
            solver[env, ws_ntc_idx + contact_tid] = 0

    # Read metadata (legacy `dt` read dropped — only the unused-arg limits
    # call consumed it)
    var nc = 0
    var K_spring: Scalar[DTYPE] = 0
    var B_damp: Scalar[DTYPE] = 0
    var si_dmin: Scalar[DTYPE] = 0
    var si_dmax: Scalar[DTYPE] = 0
    var si_width: Scalar[DTYPE] = 1
    var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
    var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)
    var impratio: Scalar[DTYPE] = Scalar[DTYPE](1.0)

    nc = Int(rebind[Scalar[DTYPE]](smeta[env, META_IDX_NUM_CONTACTS]))
    if nc > max_contacts:
        nc = max_contacts
    var sr_tc = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLREF_CONTACT_0]
    )
    var sr_dr = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLREF_CONTACT_1]
    )
    si_dmin = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_0])
    si_dmax = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_1])
    si_width = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_2])
    si_midpoint = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLIMP_CONTACT_3]
    )
    si_power = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_4])
    if si_width < Scalar[DTYPE](1e-6):
        si_width = Scalar[DTYPE](1e-6)
    # MuJoCo clamps BOTH ends of solimp to [mjMINIMP, mjMAXIMP] before
    # interpolating (engine_core_constraint.c:1284-1287). The dmin floor is
    # the one that bites: R = (1-imp)/imp * diagApprox, so dmin=0 asks for an
    # infinitely soft contact at first touch. dm_control's finger is the first
    # model here to set it (`solimp="0 0.9 0.01"`); everything before used the
    # 0.9 default, which is why clamping only dmax survived.
    comptime MJ_MINIMP = Scalar[DTYPE](0.0001)
    comptime MJ_MAXIMP = Scalar[DTYPE](0.9999)
    if si_dmin < MJ_MINIMP:
        si_dmin = MJ_MINIMP
    elif si_dmin > MJ_MAXIMP:
        si_dmin = MJ_MAXIMP
    if si_dmax < MJ_MINIMP:
        si_dmax = MJ_MINIMP
    elif si_dmax > MJ_MAXIMP:
        si_dmax = MJ_MAXIMP
    if si_power < Scalar[DTYPE](1):
        si_power = Scalar[DTYPE](1)
    # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
    # solref. See `constraints/constraint_data.solref_spring_damper` — the
    # formula lived in twelve copy-pasted sites until 2026-08-03.
    (K_spring, B_damp) = solref_spring_damper[DTYPE](
        sr_tc, sr_dr, si_dmax,
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
    )
    impratio = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_IMPRATIO])
    if impratio < Scalar[DTYPE](1e-6):
        impratio = Scalar[DTYPE](1.0)

    # === PHASE 1: normal precompute (legacy: parallel, one thread per
    # contact slot; internal `contact_tid < nc` guard kept in the helper) ===
    for contact_tid in range(max_contacts):
        _precompute_contact_normal[
            DTYPE, V_CAP](
            env,
            contact_tid,
            nc,
            dims,
            qvel,
            subtree_com,
            contacts,
            joints,
            bodies,
            mmeta,
            body_invweight0,
            cdof,
            m_inv,
            qacc_constrained,
            solver,
            K_spring,
            B_damp,
            si_dmin,
            si_dmax,
            si_width,
            si_midpoint,
            si_power,
        )

    # === PHASE 2: Tangent frame + friction data (legacy launch guard
    # `contact_tid < nc`) ===
    for contact_tid in range(nc):
        _precompute_contact_friction[
            DTYPE,
            V_CAP, CONE_TYPE=CONE_TYPE, MAX_CONDIM=MAX_CONDIM](
            env,
            contact_tid,
            nc,
            dims,
            qvel,
            subtree_com,
            contacts,
            joints,
            bodies,
            mmeta,
            cdof,
            solver,
            B_damp,
            impratio,
            K_spring,
        )

    # === SEQUENTIAL: primal Newton (legacy: thread 0) ===
    # ⚠ A CEILING, NOT THE BUDGET — the model's `<option iterations>` is read
    # below and the loops break on it. Raised from 200 when it stopped being
    # the budget: it now only has to be above any count a model can ask for,
    # and a `range()` needs a bound the compiler can see.
    comptime NEWTON_ITER_GPU: Int = 1000
    # ⚠⚠ THE TOLERANCE IS DTYPE-AWARE, AND AT FLOAT32 IT HAS TO BE. Both exit
    # tests — `scale * ||grad||` and `scale * improvement` — are differences of
    # same-magnitude terms, so at float32 their rounding floor sits ORDERS OF
    # MAGNITUDE above 1e-8. Neither test can ever fire, and the solver runs its
    # full `NEWTON_ITER_GPU` budget on every step that has a single constraint
    # row. Measured on SO-ARM100 (one shallow contact, 6 DOF): 1.04 ms/env step
    # against 0.55 ms once the threshold clears the noise — HALF the step spent
    # iterating on rounding error. MuJoCo uses 1e-8 and is float64 throughout,
    # so the deviation is ours to make, not theirs to match.
    #
    # ⚠ THE EXTRA ITERATIONS BUY NOTHING, WHICH IS THE POINT. Measured on a
    # settling sphere: 1e-6 moves the resting penetration by 1.5e-8, while
    # float32's own distance from float64 is 9.8e-9 to 1e-6 depending on the
    # model — i.e. the correction is at or below the dtype's own error. Loosen
    # it much further and that stops being true: at 1e-1 the depth moves 2.7e-6.
    #
    # ⚠ NO FLOAT64 BEHAVIOUR CHANGES — the float64 branch is the literal old
    # constant, so every MuJoCo-parity gate in the tree (all of which run at
    # float64) is bit-identical across this change. That also means NONE of
    # them covers the float32 branch; `test_newton_float32_tracks_float64.mojo`
    # exists for that and is the only float32 convergence gate there is.
    comptime NEWTON_TOL_GPU: Float64 = (
        1e-8 if DTYPE == DType.float64 else 1e-6
    )
    # ⚠⚠ MuJoCo'S LINESEARCH BUDGET IS 50, NOT 20 (`m->opt.ls_iterations`) —
    # and it is the DEFAULT of a model field, not a constant. A ceiling here;
    # `lsiter_rt` below is the count. apollo asks for 10, so101 for 20.
    comptime LINESEARCH_ITER: Int = 50
    # ⚠⚠ AND ITS LINESEARCH TOLERANCE IS `opt.tolerance * opt.ls_tolerance`,
    # NOT `opt.tolerance` alone. `mj_solPrimal` calls
    #     PrimalSearch(&ctx, m->opt.tolerance * m->opt.ls_tolerance, ...)
    # and `PrimalSearch` forms `gtol = tolerance * snorm / scale` from THAT
    # product. `ls_tolerance` defaults to 0.01, so the real threshold is 1e-10
    # and ours was 1e-8 — a HUNDRED times looser, which makes the search accept
    # its first 1-D Newton point instead of refining toward the minimum along
    # the search direction.
    #
    # ⚠ THE SYMPTOM IS A SMALLER STEP, NOT A LOOSER ONE. Measured per iteration
    # on `reassemble_3` at float64 before this: the outer Newton converges
    # QUADRATICALLY while alpha stays ~1 (grad 1.19e-02 -> 8.49e-06 -> 4.25e-08
    # across iterations 5-7, alpha 1.0002 then 1.0029), and then alpha
    # collapses — 0.108, 0.086, 0.072, ... 0.0060 — and the gradient creeps
    # ~0.6% per iteration for another 77 iterations to reach 1e-8. The Hessian
    # is right (quadratic convergence proves it) and the active set stops
    # flipping at iteration 4, so the accepted alpha is what is wrong.
    comptime LS_TOLERANCE: Float64 = 0.01
    comptime ARMIJO: Float64 = 1e-4
    comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

    # ── the model's SOLVER BUDGET, from meta ────────────────────────────────
    #
    # ⚠⚠ THESE WERE THE COMPTIME CONSTANTS BELOW AND THE MODEL WAS IGNORED.
    # `apptronik_apollo` ships `<option iterations="4" ls_iterations="10">` and
    # we ran it to convergence — a DIFFERENT answer, not a better one, because
    # MuJoCo's answer for that model is its 4-iteration iterate. Five
    # Menagerie models set `iterations`, four set `ls_iterations` and rby1's
    # five scenes set `tolerance="1e-6"`.
    #
    # ⚠ THE COMPTIME CONSTANTS SURVIVE AS CAPS, not as the budget. A `range()`
    # needs a bound the compiler can see for the GPU path, so the loops still
    # run to `NEWTON_ITER_GPU` / `LINESEARCH_ITER` and break at the model's
    # count; a model asking for MORE than the cap is truncated at it.
    var niter_rt = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLVER_ITERATIONS])
    )
    if niter_rt <= 0:
        niter_rt = NEWTON_ITER_GPU
    var lsiter_rt = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_LS_ITERATIONS])
    )
    if lsiter_rt <= 0:
        lsiter_rt = LINESEARCH_ITER
    # ⚠ THE FLOOR IS THE DTYPE'S, NOT THE MODEL'S, AND IT ONLY RAISES.
    # `NEWTON_TOL_GPU` is 1e-8 at float64 (MuJoCo's own default, so this is a
    # no-op there) and 1e-6 at float32, where both exit tests are differences
    # of same-magnitude terms whose rounding floor sits above 1e-8 — a model
    # asking for 1e-8 at float32 would never converge and would burn its whole
    # budget on rounding error. A model asking for something LOOSER keeps it.
    var tol_rt = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLVER_TOLERANCE]
    )
    comptime if DTYPE != DType.float64:
        if tol_rt < Scalar[DTYPE](NEWTON_TOL_GPU):
            tol_rt = Scalar[DTYPE](NEWTON_TOL_GPU)
    # ⚠ A MULTIPLIER, NOT A THRESHOLD — `mj_solPrimal` passes
    # `opt.tolerance * opt.ls_tolerance` to `PrimalSearch`.
    var lstol_rt = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_LS_TOLERANCE]
    )
    if lstol_rt <= Scalar[DTYPE](0):
        lstol_rt = Scalar[DTYPE](LS_TOLERANCE)


    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        # =================================================================
        # PYRAMIDAL Newton: iterate over edge rows (all >= 0 constraints)
        # 4 edges per contact for condim=3: J_e = J_n ± mu*J_t
        # No cone coupling — simpler than ELLIPTIC
        # =================================================================
        # Edges per contact = 2*(dim-1): 4 at condim 3, 6 at 4, 10 at 6.
        # Slots are sized for the model's worst condim; the builder zeros the
        # tail per contact, so a condim-3 contact here still spans 4 edges.
        comptime NE = 2 * (MAX_CONDIM - 1)
        # ⚠ TWO SPELLINGS OF THE ROW BUDGET, AND THEY ARE NOT INTERCHANGEABLE.
        # `E_CAP` sizes the arrays and is 0 on a dynamic provider; `me` is the
        # live budget the CAPACITY GUARDS below compare against
        # (`num_edges < me`). Guarding with the cap would admit zero rows on
        # the dynamic leg and silently solve an unconstrained system.
        comptime E_CAP = cap[
            NE * D.CAP_MAX_CONTACTS
            + 2 * D.CAP_NJOINT
            + D.CAP_NV
            + 2 * D.CAP_NTENDON
            + D.CAP_NTENDON
            + 6 * D.CAP_NEQUALITY
        ]()
        var me = (
            NE * max_contacts + 2 * njoint + nv + 2 * ntendon + ntendon
            + 6 * nequality
        )

        # Cache edge data from PYRAMIDAL workspace layout
        var pyr_sc = ws_Jt_idx + NE * max_contacts * nv
        var Je = Scratch[Scalar[DTYPE], E_CAP * V_CAP](me * nv, uninitialized=Scalar[DTYPE](0))
        var De = Scratch[Scalar[DTYPE], E_CAP](me, uninitialized=Scalar[DTYPE](0))
        var bias_e = Scratch[Scalar[DTYPE], E_CAP](me, uninitialized=Scalar[DTYPE](0))
        # Row kind + box data. Contact edges and joint limits are ONE-SIDED;
        # only dry-friction dof rows are box-clamped, and R/floss are read
        # solely on that branch, so the one-sided rows leave them at 0.
        var kind_e = Scratch[Int, E_CAP](me, fill=SROW_LIMIT)
        var R_e = Scratch[Scalar[DTYPE], E_CAP](me, fill=Scalar[DTYPE](0))
        var floss_e = Scratch[Scalar[DTYPE], E_CAP](me, fill=Scalar[DTYPE](0))
        var state_e = Scratch[Int, E_CAP](me, fill=0)
        var num_edges = nc * NE

        # Load contact edges
        for c in range(nc):
            for e in range(NE):
                var idx = c * NE + e
                for i in range(nv):
                    Je[idx * nv + i] = rebind[Scalar[DTYPE]](
                        solver[env, ws_Jt_idx + e * max_contacts * nv + c * nv + i]
                    )
                De[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + e * max_contacts + c]
                )
                bias_e[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + NE * max_contacts + e * max_contacts + c]
                )

        # Detect and add joint limit edges (unified with contacts)
        # Matches CPU build_constraints: per-joint solref/solimp with
        # model-level defaults fallback
        # Model-level defaults for fallback
        var lr_tc_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_0]
        )
        var lr_dr_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_1]
        )
        var li_dmin_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_0]
        )
        var li_dmax_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_1]
        )
        var li_width_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_2]
        )
        var li_midpoint_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_3]
        )
        var li_power_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_4]
        )

        for j in range(njoint):
            var jtype = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])
            )
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            var rmin = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_RANGE_MIN]
            )
            var rmax = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_RANGE_MAX]
            )
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            # Per-joint solref/solimp with model-level defaults fallback
            var lr_tc = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_0]
            )
            var lr_dr = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_1]
            )
            if lr_tc <= Scalar[DTYPE](0):
                lr_tc = lr_tc_def
            if lr_dr <= Scalar[DTYPE](0):
                lr_dr = lr_dr_def
            var li_dmin = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_0]
            )
            var li_dmax = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_1]
            )
            var li_width = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_2]
            )
            var li_midpoint = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_3]
            )
            var li_power = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_4]
            )
            if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](0):
                li_dmin = li_dmin_def
                li_dmax = li_dmax_def
                li_width = li_width_def
                li_midpoint = li_midpoint_def
                li_power = li_power_def
            if li_width < Scalar[DTYPE](1e-6):
                li_width = Scalar[DTYPE](1e-6)
            # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
            # interpolating (engine_core_constraint.c:1284-1287); see the same fix
            # on the contact path above.
            comptime MJL_MINIMP = Scalar[DTYPE](0.0001)
            comptime MJL_MAXIMP = Scalar[DTYPE](0.9999)
            if li_dmin < MJL_MINIMP:
                li_dmin = MJL_MINIMP
            elif li_dmin > MJL_MAXIMP:
                li_dmin = MJL_MAXIMP
            if li_dmax < MJL_MINIMP:
                li_dmax = MJL_MINIMP
            elif li_dmax > MJL_MAXIMP:
                li_dmax = MJL_MAXIMP
            if li_power < Scalar[DTYPE](1):
                li_power = Scalar[DTYPE](1)
            # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
            # solref. See `constraints/constraint_data.solref_spring_damper` — the
            # formula lived in twelve copy-pasted sites until 2026-08-03.
            var (l_K_spring, l_B_damp) = solref_spring_damper[DTYPE](
                lr_tc, lr_dr, li_dmax,
                rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
            )

            var pos = rebind[Scalar[DTYPE]](qpos[env, qpos_adr])
            # Lower limit: dist_lo = pos - rmin < 0 → violated
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0) and num_edges < me:
                var sign = Scalar[DTYPE](1)
# ⚠ NO `K = diag(M^-1)` HERE ANY MORE. MuJoCo's `mj_diagApprox`
                # (engine_core_constraint.c:1720) prices a joint-limit row with
                # `dof_invweight0`, a MODEL-TIME constant (:1880), and its
                # `efc_D` is `1 / R` outright (:2259). This row used to read the
                # per-step `M^-1` diagonal only to round-trip it —
                # `1/(1/(K+R)) - K` — which reproduces R to a few ulp and
                # nothing else, and was the last reason the integrator formed a
                # dense `M^-1` under Newton at all (`PERFORMANCE.md` §13: 24-46%
                # of every step past 20 dofs).
                var pen = -dist_lo
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                # Impedance
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                # Sparse Jacobian: Je[dof] = sign, others 0
                for i in range(nv):
                    Je[num_edges * nv + i] = Scalar[DTYPE](0)
                Je[num_edges * nv + dof] = sign
                De[num_edges] = Scalar[DTYPE](1) / R_lim
                bias_e[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

            # Upper limit: dist_hi = rmax - pos < 0 → violated
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0) and num_edges < me:
                var sign = Scalar[DTYPE](-1)
# ⚠ NO `K = diag(M^-1)` HERE ANY MORE. MuJoCo's `mj_diagApprox`
                # (engine_core_constraint.c:1720) prices a joint-limit row with
                # `dof_invweight0`, a MODEL-TIME constant (:1880), and its
                # `efc_D` is `1 / R` outright (:2259). This row used to read the
                # per-step `M^-1` diagonal only to round-trip it —
                # `1/(1/(K+R)) - K` — which reproduces R to a few ulp and
                # nothing else, and was the last reason the integrator formed a
                # dense `M^-1` under Newton at all (`PERFORMANCE.md` §13: 24-46%
                # of every step past 20 dofs).
                var pen = -dist_hi
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                for i in range(nv):
                    Je[num_edges * nv + i] = Scalar[DTYPE](0)
                Je[num_edges * nv + dof] = sign
                De[num_edges] = Scalar[DTYPE](1) / R_lim
                bias_e[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

        # Tendon limit rows (MuJoCo mjCNSTR_LIMIT_TENDON). Dense J, one row
        # per violated side — see constraints/tendon_limit.mojo for why this
        # is a row here rather than a post-pass.
        comptime if may_exist[D.NTENDON]():
            build_tendon_limit_rows[
                DTYPE, V_CAP, E_CAP, BATCH
            ](
                env, dims, qvel, tendons, sites, geoms_w, bodies, joints,
                mmeta,
                subtree_com, cdof, xpos, xquat, m_inv,
                Je, De, bias_e, me, num_edges,
            )

        # Tendon equality rows (MuJoCo mjEQ_TENDON), FIXED and SPATIAL alike.
        # BILATERAL — always active, never clamped. These used to be a
        # post-solve Gauss-Seidel pass; with contacts live that split cost a
        # standing quadruped two thirds of its ground reaction force. See
        # constraints/tendon_limit.build_tendon_equality_rows.
        comptime if may_exist[D.NTENDON]():
            build_tendon_equality_rows[
                DTYPE, V_CAP, E_CAP,
                BATCH](
                env, dims, qpos, qvel, tendons, sites, geoms_w, bodies,
                joints, mmeta,
                subtree_com, cdof, xpos, xquat, m_inv,
                Je, De, bias_e, kind_e, me, num_edges,
            )

        # connect / weld EQUALITY rows (defect 29a), dense J, BILATERAL.
        #
        # Same conversion the ELLIPTIC path got in `d22144ee`, mirrored here
        # 2026-08-12. As a post-pass these rewrote the dofs the contacts had
        # just balanced: on sawyer the mocap weld left the object 77.6 mm from
        # where MuJoCo rests it, and moving the rows INSIDE the solve brought
        # that to 0.087 mm.
        #
        # ⚠ `eq_D` IS `1/R`, NOT `1/(k+R)`. `build_weld_equality_rows` returns
        # the PGS step size in `we_D`; MuJoCo's Newton cost wants the row
        # STIFFNESS `efc_D = 1/R` (engine_core_constraint.c:1918). Passing the
        # step size instead is what regressed defect 28 from 0.91 mm to
        # 7.86 mm on the first attempt at the elliptic conversion, and it looks
        # exactly like an iteration-budget problem while being nothing of the
        # kind.
        comptime if may_exist[D.NEQUALITY]():
            comptime WR = 6 * cap[D.NEQUALITY]()
            comptime WJ = 6 * cap[D.NEQUALITY]() * cap[D.NV]()
            var w_rows = 6 * nequality
            var w_K = Scratch[Scalar[DTYPE], WR](w_rows, Scalar[DTYPE](1))
            var w_bias = Scratch[Scalar[DTYPE], WR](w_rows, Scalar[DTYPE](0))
            var w_D = Scratch[Scalar[DTYPE], WR](w_rows, Scalar[DTYPE](0))
            var w_J = Scratch[Scalar[DTYPE], WJ](w_rows * nv, Scalar[DTYPE](0))
            var w_MinvJ = Scratch[Scalar[DTYPE], WJ](
                w_rows * nv, Scalar[DTYPE](0)
            )
            var n_w = build_weld_equality_rows[DTYPE, V_CAP](
                env, dims, qpos, qvel, xpos, xquat, subtree_com, joints, bodies,
                mmeta, equality, body_invweight0, dof_invweight0, cdof, m_inv,
                w_K, w_bias, w_D, w_J, w_MinvJ,
            )
            for r in range(n_w):
                if num_edges >= me:
                    break
                for i in range(nv):
                    Je[num_edges * nv + i] = w_J[r * nv + i]
                var R_recov = Scalar[DTYPE](1) / w_D[r] - w_K[r]
                if R_recov < Scalar[DTYPE](1e-14):
                    R_recov = Scalar[DTYPE](1e-14)
                De[num_edges] = Scalar[DTYPE](1) / R_recov
                bias_e[num_edges] = w_bias[r]
                kind_e[num_edges] = SROW_EQ_BILATERAL
                num_edges += 1

        # Dry-friction dof rows (MuJoCo mjCNSTR_FRICTION_DOF). These were
        # MISSING from the pyramidal path entirely — `_friction_env` was only
        # ever called on the elliptic branch, so a pyramidal model with
        # `frictionloss` silently had no dry friction at all. They are box
        # rows, clamped to +-frictionloss, hence kind_e = SROW_FRICTION.
        var f_imp = Scalar[DTYPE](DOF_SOLIMP_DMIN)
        var f_dmax = Scalar[DTYPE](DOF_SOLIMP_DMAX)
        # REFSAFE applies to the hardcoded friction default too — see
        # `refsafe_timeconst`.
        var f_tc_p = refsafe_timeconst[DTYPE](
            Scalar[DTYPE](DOF_SOLREF_TIMECONST),
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        var f_B = Scalar[DTYPE](2.0) / (f_dmax * f_tc_p)
        for j in range(njoint):
            var floss = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_FRICTIONLOSS]
            )
            if floss <= Scalar[DTYPE](0):
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var nd = 1
            if jt == JNT_FREE:
                nd = 6
            elif jt == JNT_BALL:
                nd = 3
            for k in range(nd):
                if num_edges >= me:
                    break
                var dof = dof_adr + k
                # `dof_invweight0`, as MuJoCo (engine_core_constraint.c:1876);
                # the `diag(M^-1)` fallback this carried was dead on any model
                # with a finite mass and is gone with the matrix.
                var diag_f = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                var R_f = (Scalar[DTYPE](1) - f_imp) / f_imp * diag_f
                if R_f < Scalar[DTYPE](1e-14):
                    R_f = Scalar[DTYPE](1e-14)
                for i in range(nv):
                    Je[num_edges * nv + i] = Scalar[DTYPE](0)
                Je[num_edges * nv + dof] = Scalar[DTYPE](1)
                De[num_edges] = Scalar[DTYPE](1) / R_f
                R_e[num_edges] = R_f
                floss_e[num_edges] = floss
                kind_e[num_edges] = SROW_FRICTION
                bias_e[num_edges] = f_B * rebind[Scalar[DTYPE]](
                    qvel[env, dof]
                )
                num_edges += 1

        # ── TREE_AWARE: the rows' nonzero dofs and the tree segments ───────
        # Both are derived from the FINISHED row list, so this sits after the
        # last row builder and before anything reads `Je`. `je_ix[e*nv + a]`,
        # `a < je_n[e]`, lists row `e`'s nonzero dofs; `seg0[i]`/`seg1[i]` is
        # the half-open dof range of the tree segment holding dof `i`
        # (`newton_blocks.build_dof_segments`). Off, both are one-element
        # placeholders so the GPU legs' frames do not grow.
        comptime N_CAP = E_CAP if TREE_AWARE else 1
        comptime IX_CAP = E_CAP * V_CAP if TREE_AWARE else 1
        var je_n = Scratch[Int, N_CAP](me if TREE_AWARE else 1, fill=0)
        var je_ix = Scratch[Int, IX_CAP](me * nv if TREE_AWARE else 1, fill=0)
        var seg0 = Scratch[Scalar[DTYPE], V_CAP](nv, fill=Scalar[DTYPE](0))
        var seg1 = Scratch[Scalar[DTYPE], V_CAP](nv, fill=Scalar[DTYPE](nv))
        comptime if TREE_AWARE:
            for e_idx in range(num_edges):
                var n_e = 0
                for i in range(nv):
                    if Je[e_idx * nv + i] != Scalar[DTYPE](0):
                        je_ix[e_idx * nv + n_e] = i
                        n_e += 1
                je_n[e_idx] = n_e
            _ = build_dof_segments_p[DTYPE](
                nv,
                Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NTREE])),
                num_edges,
                trees.ptr,
                Je.unsafe_ptr(),
                seg0.unsafe_ptr(),
                seg1.unsafe_ptr(),
            )
            # The blocked kernel's guard, kept for the same reason: a segment
            # end at or below its start would never advance the walk below.
            for i in range(nv):
                if Int(seg1[i]) <= i:
                    seg1[i] = Scalar[DTYPE](nv)

        # Initialize qacc from workspace (qacc_smooth set by stage kernel)
        var qacc = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
        var qacc_smooth = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
        var Ma = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))

        # Cache M locally once — M is loop-invariant during Newton iterations.
        # Avoids ~2*NV² workspace (global) reads per iteration (Hessian build
        # + Mv = M*search). Mirrors the ELLIPTIC path's M_local optimization.
        var M_local = Scratch[Scalar[DTYPE], M_CAP](nv * nv, uninitialized=Scalar[DTYPE](0))
        for k in range(nv * nv):
            M_local[k] = rebind[Scalar[DTYPE]](M[env, k])

        for i in range(nv):
            var q_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
            qacc[i] = q_i
            qacc_smooth[i] = q_i
        for i in range(nv):
            Ma[i] = Scalar[DTYPE](0)
            # `M` couples a dof only with its tree, and a segment is a union
            # of trees, so the entries outside `[seg0, seg1)` are exact zeros.
            comptime if TREE_AWARE:
                for j in range(Int(seg0[i]), Int(seg1[i])):
                    Ma[i] += M_local[i * nv + j] * qacc[j]
            else:
                for j in range(nv):
                    Ma[i] += M_local[i * nv + j] * qacc[j]
        # f_smooth = M * qacc (matching CPU's qfrc_smooth = M * qacc_smooth)
        # Using Ma directly avoids LDL round-trip error (f_net ≠ M*M^{-1}*f_net)
        var f_smooth = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
        for i in range(nv):
            f_smooth[i] = Ma[i]

        # ⚠ MuJoCo's CONVERGENCE SCALE IS A MODEL CONSTANT, NOT A POSE ONE.
        # `mj_solPrimal` uses `1 / (stat.meaninertia * max(1, nv))`
        # (`engine_solver.c:1863`), and `stat.meaninertia` is the mean of the
        # mass-matrix diagonal evaluated ONCE at qpos0 in `mj_setConst`. This
        # summed `M[i][i]` at the CURRENT pose instead — the same formula
        # (`sum(diag M)` at qpos0 IS `meaninertia * nv`; measured on dog,
        # 35.635564 both ways) evaluated at the wrong point.
        #
        # It scales BOTH exit tests, `improvement < tol` and `gradient < tol`,
        # so a pose-dependent scale makes the effective tolerance wander with
        # the configuration. Measured on dog at its settled pose: 34.107946
        # against 35.635564, i.e. a tolerance 1.045x looser than MuJoCo's.
        # Unbounded in general — a model that folds up moves its diagonal a lot
        # further than 4.5%.
        #
        # ⚠ THIS IS NOT A FIX FOR THE OPEN DOG RESIDUAL and must not be read as
        # one: tightening `NEWTON_TOL_GPU` to 1e-14 leaves our answer identical
        # to the last digit, so the exit threshold is not what is holding it.
        # This is a fidelity correction on its own merits.
        #
        # `meaninertia` reached the model meta with `mj_solNoSlip`, which needs
        # it for the same reason.
        # ⚠ STAY IN `DTYPE`. Computing this in Float64 makes the enclosing
        # kernel return a double and Metal rejects the module outright
        # ("returns unsupported type 'double'"), which is a BUILD failure on
        # every GPU model, not a dog-only one.
        var scale_d = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_MEANINERTIA]
        ) * Scalar[DTYPE](nv if nv > 1 else 1)
        var scale = (
            Scalar[DTYPE](1) / scale_d
            if scale_d > Scalar[DTYPE](1e-10)
            else Scalar[DTYPE](1)
        )

        # Working arrays
        var jar = Scratch[Scalar[DTYPE], E_CAP](me, uninitialized=Scalar[DTYPE](0))
        var force = Scratch[Scalar[DTYPE], E_CAP](me, uninitialized=Scalar[DTYPE](0))
        var H = Scratch[Scalar[DTYPE], M_CAP](nv * nv, uninitialized=Scalar[DTYPE](0))
        var L_chol = Scratch[Scalar[DTYPE], M_CAP](nv * nv, uninitialized=Scalar[DTYPE](0))
        var grad = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
        var search = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
        var Mv = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))

        # Initial jar + force + qfrc
        var qfrc = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
        pyramidal_edge_forces[
                DTYPE, E_CAP, V_CAP, N_CAP, IX_CAP, SPARSE=TREE_AWARE
            ](
            num_edges, Je, De, bias_e, kind_e, R_e, floss_e,
            qacc, jar, force, state_e, qfrc, nv, je_n, je_ix,
        )

        # ── warmstart(): START AT THE CHEAPER OF `qacc_warmstart` AND
        # `qacc_smooth` (engine_forward.c:786) ───────────────────────────────
        #
        # ⚠⚠ IT IS A COST COMPARISON, NOT A COPY. `qacc_warmstart` carries the
        # previous `mj_forward`'s answer, which after a contact change or a
        # reset can be far worse than the cold start; MuJoCo prices both and
        # keeps the cheaper. Skipping the test is a DIFFERENT algorithm, not a
        # cheaper version of this one.
        #
        # The Gauss term is `0.5*(M*q - qfrc_smooth)·(q - qacc_smooth)`, which
        # is identically 0 at `q = qacc_smooth` — hence `cost_s` carrying the
        # constraint rows only, exactly as MuJoCo's
        # `mj_constraintUpdate(m, d, d->efc_b, &cost_smooth, 0)` does.
        #
        # ⚠ THE TIE GOES TO THE WARM START. MuJoCo falls back only on
        # `cost_warmstart > cost_smooth`, so `<=` here is the reference's `>`
        # negated and not a choice of ours.
        #
        # ⚠ THE TRIAL STATE IS NOT KEPT. Pricing the candidate needs its jar
        # and its row states, and holding those would add two `E_CAP` locals
        # to a frame that Metal already sizes at the edge; instead the cost is
        # accumulated row by row and `pyramidal_edge_forces` is simply re-run
        # once, on whichever `qacc` won. `Ma` is restored from `f_smooth`,
        # which is bit-identically `M * qacc_smooth`.
        if (
            mmeta[MODEL_META_IDX_WARMSTART_DISABLED] == Scalar[DTYPE](0)
            and num_edges > 0
        ):
            var cost_s: Scalar[DTYPE] = 0
            for e_idx in range(num_edges):
                cost_s += scalar_row_cost[DTYPE](
                    state_e[e_idx], jar[e_idx], De[e_idx], R_e[e_idx],
                    floss_e[e_idx],
                )
            var qacc_w = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
            for i in range(nv):
                qacc_w[i] = rebind[Scalar[DTYPE]](qacc_warmstart[env, i])
            var cost_w: Scalar[DTYPE] = 0
            for e_idx in range(num_edges):
                var jar_w = bias_e[e_idx]
                comptime if TREE_AWARE:
                    for a in range(je_n[e_idx]):
                        var i = je_ix[e_idx * nv + a]
                        jar_w += Je[e_idx * nv + i] * qacc_w[i]
                else:
                    for i in range(nv):
                        jar_w += Je[e_idx * nv + i] * qacc_w[i]
                var st_w = scalar_row_state[DTYPE](
                    kind_e[e_idx], jar_w, R_e[e_idx], floss_e[e_idx]
                )
                cost_w += scalar_row_cost[DTYPE](
                    st_w, jar_w, De[e_idx], R_e[e_idx], floss_e[e_idx]
                )
            for i in range(nv):
                var s_i: Scalar[DTYPE] = 0
                comptime if TREE_AWARE:
                    for j in range(Int(seg0[i]), Int(seg1[i])):
                        s_i += M_local[i * nv + j] * qacc_w[j]
                else:
                    for j in range(nv):
                        s_i += M_local[i * nv + j] * qacc_w[j]
                Ma[i] = s_i
                cost_w += (
                    Scalar[DTYPE](0.5)
                    * (s_i - f_smooth[i])
                    * (qacc_w[i] - qacc_smooth[i])
                )
            if cost_w <= cost_s:
                for i in range(nv):
                    qacc[i] = qacc_w[i]
                pyramidal_edge_forces[
                DTYPE, E_CAP, V_CAP, N_CAP, IX_CAP, SPARSE=TREE_AWARE
            ](
                    num_edges, Je, De, bias_e, kind_e, R_e, floss_e,
                    qacc, jar, force, state_e, qfrc, nv, je_n, je_ix,
                )
            else:
                for i in range(nv):
                    Ma[i] = f_smooth[i]

        # Newton iterations
        for iter_n in range(NEWTON_ITER_GPU):
            # ⚠⚠ NO CONSTRAINT ROWS: MUJOCO RETURNS, AND WE USED TO SOLVE.
            # `mj_fwdConstraint` (engine_forward.c:884) is explicit —
            #     if (!nefc) { mju_copy(d->qacc, d->qacc_smooth, nv); return; }
            # — and this loop had no such guard. With no rows the warmstart block
            # above is already skipped (its own `num_edges > 0` test), so `qacc` still
            # holds `qacc_smooth` and the gradient is IDENTICALLY ZERO. Every
            # iteration then factors `H = M` and solves for a search direction of
            # zero: the answer cannot move, so breaking here writes back exactly
            # what a full pass would have — this is a no-op, not an approximation.
            #
            # ⚠ IT IS NOT A MICRO-OPTIMISATION. P0 measured it at 32.3 of 46.9 ms
            # per step on the k=9 park scene — 69% of GPU time, 78% of the whole
            # parked-slot cost — spent re-factorising a matrix `ldl_factor` factored
            # two kernels earlier, for a problem with no constraints in it. The
            # cooperative Cholesky's `d_j` reduction is `if tid == 0: for k in
            # range(j)`, O(NV^2) on ONE thread, so it is also the fastest-growing
            # term in the sweep.
            # See docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md §0.0.1.
            if num_edges == 0:
                break
            # ⚠ THE MODEL'S BUDGET, checked before any work — the comptime
            # bound above is only the ceiling a `range()` needs.
            if iter_n >= niter_rt:
                break
            # Gradient
            var grad_norm: Scalar[DTYPE] = 0
            for i in range(nv):
                grad[i] = Ma[i] - f_smooth[i] - qfrc[i]
                grad_norm += grad[i] * grad[i]

            # ⚠⚠ NOT ON THE FIRST PASS. `mj_solPrimal` evaluates `gradient`
            # only AFTER an update (engine_solver.c:2270-2282): its loop is
            # linesearch -> move -> update -> test, so a solve that STARTS
            # inside the tolerance still takes one line search, and only
            # `PrimalSearch` returning `alpha == 0` can end it with zero
            # updates. Testing here at `iter_n == 0` let a warm start be
            # returned UNTOUCHED — which is exactly the regime the warm start
            # creates, and why it cost `test_frictionless_contact_pyramidal`
            # three orders (4.4e-16 -> 1.2e-12 of qpos against MuJoCo over 60
            # steps) the day `qacc_warmstart` started carrying.
            if iter_n > 0 and scale * sqrt(grad_norm) < tol_rt:
                break

            # Build Hessian H = M + sum_active(D[e] * Je^T * Je)
            for i in range(nv):
                for j in range(nv):
                    H[i * nv + j] = M_local[i * nv + j]
            for e_idx in range(num_edges):
                if state_e[e_idx] == SROW_QUADRATIC:
                    comptime if TREE_AWARE:
                        # `(De * Je_i) * Je_j` is the dense expression's own
                        # evaluation order, so hoisting the left factor is
                        # the same arithmetic.
                        var n_e = je_n[e_idx]
                        for a in range(n_e):
                            var i = je_ix[e_idx * nv + a]
                            var dji = De[e_idx] * Je[e_idx * nv + i]
                            for b in range(n_e):
                                var j = je_ix[e_idx * nv + b]
                                H[i * nv + j] += dji * Je[e_idx * nv + j]
                    else:
                        for i in range(nv):
                            for j in range(nv):
                                H[i * nv + j] += (
                                    De[e_idx]
                                    * Je[e_idx * nv + i]
                                    * Je[e_idx * nv + j]
                                )

            # Cholesky solve
            comptime if TREE_AWARE:
                # One factorisation per tree segment into the same zeroed
                # `L` — `chol_factor_inline` is exactly this over `[0, nv)`.
                # A rank failure anywhere gets the dense path's remedy (the
                # whole diagonal lifted, everything refactored), so the
                # answer stays the dense one bit for bit.
                for k in range(nv * nv):
                    L_chol[k] = Scalar[DTYPE](0)
                var chol_ok = True
                var s0 = 0
                while s0 < nv:
                    var s1 = Int(seg1[s0])
                    var ok_s = chol_factor_seg[DTYPE, M_CAP](
                        H, L_chol, nv, s0, s1
                    )
                    chol_ok = chol_ok and ok_s
                    s0 = s1
                if not chol_ok:
                    for i in range(nv):
                        H[i * nv + i] += Scalar[DTYPE](1e-6)
                    for k in range(nv * nv):
                        L_chol[k] = Scalar[DTYPE](0)
                    s0 = 0
                    while s0 < nv:
                        var s1 = Int(seg1[s0])
                        _ = chol_factor_seg[DTYPE, M_CAP](
                            H, L_chol, nv, s0, s1
                        )
                        s0 = s1
                s0 = 0
                while s0 < nv:
                    var s1 = Int(seg1[s0])
                    chol_solve_seg[DTYPE, M_CAP, V_CAP](
                        L_chol, grad, search, nv, s0, s1
                    )
                    s0 = s1
            else:
                var chol_ok = chol_factor_inline[DTYPE, M_CAP](H, L_chol, nv)
                if not chol_ok:
                    for i in range(nv):
                        H[i * nv + i] += Scalar[DTYPE](1e-6)
                    _ = chol_factor_inline[DTYPE, M_CAP](H, L_chol, nv)
                chol_solve_inline[DTYPE, M_CAP, V_CAP](
                    L_chol, grad, search, nv
                )
            for i in range(nv):
                search[i] = -search[i]

            # Mv = M * search
            for i in range(nv):
                Mv[i] = Scalar[DTYPE](0)
                comptime if TREE_AWARE:
                    for j in range(Int(seg0[i]), Int(seg1[i])):
                        Mv[i] += M_local[i * nv + j] * search[j]
                else:
                    for j in range(nv):
                        Mv[i] += M_local[i * nv + j] * search[j]

            # `PrimalSearch` (engine_solver.c:1692) — an ITERATED search, not
            # a single analytical step. ⚠ `gtol_scale` is
            # `opt.tolerance * opt.ls_tolerance / scale`, the product
            # `mj_solPrimal` passes at engine_solver.c:2236 divided by the
            # convergence scale; the callee multiplies by `|search|`.
            var alpha = pyramidal_linesearch[
                DTYPE, E_CAP, V_CAP, LINESEARCH_ITER,
                PRIMAL_MINVAL_GPU, N_CAP, IX_CAP, SPARSE=TREE_AWARE
            ](
                num_edges, Je, De, kind_e, R_e, floss_e, search, Mv, Ma,
                f_smooth, qacc, qacc_smooth, jar,
                nv, je_n, je_ix,
                lsiter_rt,
                tol_rt * lstol_rt / scale,
            )

            if alpha < Scalar[DTYPE](1e-10):
                break

            # Save old state for cost revert (matching CPU solver)
            var old_qacc = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
            var old_Ma = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
            var old_jar = Scratch[Scalar[DTYPE], E_CAP](me, uninitialized=Scalar[DTYPE](0))
            var old_force = Scratch[Scalar[DTYPE], E_CAP](me, uninitialized=Scalar[DTYPE](0))
            var old_qfrc = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
            for i in range(nv):
                old_qacc[i] = qacc[i]
                old_Ma[i] = Ma[i]
                old_qfrc[i] = qfrc[i]
            for e_idx in range(num_edges):
                old_jar[e_idx] = jar[e_idx]
                old_force[e_idx] = force[e_idx]

            # Compute old cost: gauss + constraint
            var old_cost: Scalar[DTYPE] = 0
            for i in range(nv):
                old_cost += (
                    Scalar[DTYPE](0.5)
                    * (Ma[i] - f_smooth[i])
                    * (qacc[i] - qacc_smooth[i])
                )
            for e_idx in range(num_edges):
                old_cost += scalar_row_cost[DTYPE](
                    state_e[e_idx], jar[e_idx], De[e_idx], R_e[e_idx],
                    floss_e[e_idx],
                )

            # Update qacc, Ma
            for i in range(nv):
                qacc[i] += alpha * search[i]
                Ma[i] += alpha * Mv[i]

            # Recompute jar, force, qfrc
            pyramidal_edge_forces[
                DTYPE, E_CAP, V_CAP, N_CAP, IX_CAP, SPARSE=TREE_AWARE
            ](
                num_edges, Je, De, bias_e, kind_e, R_e, floss_e,
                qacc, jar, force, state_e, qfrc, nv, je_n, je_ix,
            )

            # Compute new cost and check improvement
            var new_cost: Scalar[DTYPE] = 0
            for i in range(nv):
                new_cost += (
                    Scalar[DTYPE](0.5)
                    * (Ma[i] - f_smooth[i])
                    * (qacc[i] - qacc_smooth[i])
                )
            for e_idx in range(num_edges):
                new_cost += scalar_row_cost[DTYPE](
                    state_e[e_idx], jar[e_idx], De[e_idx], R_e[e_idx],
                    floss_e[e_idx],
                )

            var improvement = scale * (old_cost - new_cost)
            comptime if _PYR_TRACE:
                print("  [pyr]", iter_n, "alpha", alpha, "impr", improvement,
                      "tol", tol_rt)
            if improvement < tol_rt and iter_n > 0:
                if improvement < Scalar[DTYPE](0):
                    # Cost increased — revert to old state
                    for i in range(nv):
                        qacc[i] = old_qacc[i]
                        Ma[i] = old_Ma[i]
                        qfrc[i] = old_qfrc[i]
                    for e_idx in range(num_edges):
                        jar[e_idx] = old_jar[e_idx]
                        force[e_idx] = old_force[e_idx]
                break

        # ── mj_solNoSlip ───────────────────────────────────────────────────
        # A friction-only Gauss-Seidel sweep with the NORMAL forces frozen,
        # run after the primal solve. Off unless the model asks for it
        # (`<option noslip_iterations>`); dm_control's dog is the only in-scope
        # model that does, and there it is first-order — 2.9e-2 of qvel on the
        # first contacting step — not a rounding refinement.
        #
        # PYRAMIDAL path only, and that is not an oversight: this is the
        # pyramidal branch of the solver, and `noslip.mojo` implements the
        # matching branch of MuJoCo's routine. The elliptic path below does
        # NOT call it, so an elliptic model with `noslip_iterations` set gets
        # the pass silently skipped — which is exactly why `ModelDefFromXML`
        # makes `noslip_iter > 0` a build error unless the model opts in.
        #
        # ⚠⚠ `NOSLIP_ITER` IS AN ENABLE, NOT THE COUNT — it was the count until
        # 2026-08-25. The number of sweeps is `opt.noslip_iterations`, read
        # from meta at the call below; this guard only decides whether the
        # block is EMITTED, so a model that will never want the pass reserves
        # no `kind_dt` and pays no code for it. The split is what lets a model
        # loaded at RUNTIME run the pass at all: the studio builds its five
        # integrators before it knows which file it will open.
        comptime if NOSLIP_ITER > 0:
            # `max(1, nv)` folded at compile time — see the note on the
            # `scale` argument below for why this must not be an int->float
            # conversion in the kernel body.
            comptime NV_SCALE: Float64 = Float64(D.CAP_NV if D.CAP_NV > 1 else 1)
            # ⚠ DTYPE MIRROR OF `kind_e`. `noslip_pyramidal` takes the row kind
            # as DTYPE so the blocked kernel can hand it `kind_e_sh.ptr`
            # directly (its shared slab is single-dtype, and keeping that
            # caller allocation-free is the point). This path holds `Int`s, so
            # it converts here. Built immediately before the call from the
            # authoritative array and never written after, so it cannot go
            # stale; and it is inside the `comptime if NOSLIP_ITER > 0` above,
            # so a model without the pass reserves nothing for it.
            var kind_dt = Scratch[Scalar[DTYPE], E_CAP](me, fill=Scalar[DTYPE](0))
            for e_k in range(num_edges):
                kind_dt[e_k] = Scalar[DTYPE](kind_e[e_k])
            noslip_pyramidal[
                DTYPE, E_CAP, V_CAP, MC_CAP, D.CAP_MAX_CONTACTS,
                MAX_CONDIM,
            ](
                env,
                nc,
                num_edges,
                contacts,
                m_inv,
                # ⚠ POINTERS, not the arrays. `noslip_pyramidal` takes its row
                # storage as address-space-parameterized pointers so the SAME
                # routine can also be called from the blocked kernel, whose
                # rows live in threadgroup (or, for `Je`, global) memory. Here
                # everything is a per-thread `InlineArray`, so every address
                # space is the GENERIC default.
                Je.unsafe_ptr(),
                bias_e.unsafe_ptr(),
                kind_dt.unsafe_ptr(),
                R_e.unsafe_ptr(),
                floss_e.unsafe_ptr(),
                qacc_smooth,
                # `scale` = 1 / (meaninertia * max(1, nv)) and `tolerance` =
                # opt.noslip_tolerance. Both must be MuJoCo's or the sweep
                # stops on a different iteration — see the note on
                # MODEL_META_IDX_MEANINERTIA.
                #
                # ⚠ BUILT IN DTYPE, NOT Float64. This used to widen the
                # meaninertia read to Float64 and multiply by `Float64(nv)`;
                # both are `double` in the emitted kernel and Metal rejects
                # them — porting `noslip` itself would have been pointless
                # with the conversion still here at the call site. `NV_SCALE`
                # is comptime so no int->float conversion survives either
                # (Metal also rejects `air.convert.f.f64.s.i64`).
                Scalar[DTYPE](1.0)
                / (
                    rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_MEANINERTIA])
                    * Scalar[DTYPE](NV_SCALE)
                ),
                rebind[Scalar[DTYPE]](
                    mmeta[MODEL_META_IDX_NOSLIP_TOLERANCE]
                ),
                # ⚠⚠ THE COUNT COMES FROM THE MODEL, NOT FROM `NOSLIP_ITER`.
                # The comptime parameter is an ENABLE — it decides whether this
                # block is emitted and whether `kind_dt` is reserved — and the
                # number of sweeps is `opt.noslip_iterations`, read from meta
                # beside the tolerance that trims it. Splitting the two is what
                # lets a RUNTIME-loaded model run the pass: the studio and the
                # fidelity harnesses build their integrator long before they
                # know which file they will open, so a comptime count meant the
                # pass was off for every one of them.
                #
                # ⚠ `min` WOULD BE WRONG HERE and there is deliberately none:
                # `NOSLIP_ITER` is not a capacity, nothing is sized by it, and
                # capping the model's count at whatever the caller happened to
                # build would be a silent truncation of the reference's loop.
                # `model_def_from_xml` raises instead when the comptime enable
                # and this slot disagree.
                Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NOSLIP_ITERATIONS])),
                qacc,
                jar,
                force.unsafe_ptr(),
                qfrc,
                nv,
            )

        # Write qacc back
        for i in range(nv):
            qacc_constrained[env, i] = qacc[i]

        # Write forces to state: reconstruct per-contact N/T1/T2
        for c in range(nc):
            var fn_c: Scalar[DTYPE] = 0
            var ft1_c: Scalar[DTYPE] = 0
            var ft2_c: Scalar[DTYPE] = 0
            var mu_c = rebind[Scalar[DTYPE]](
                solver[env, pyr_sc + 2 * NE * max_contacts + c]
            )
            var safe_mu = mu_c
            if safe_mu < Scalar[DTYPE](1e-8):
                safe_mu = Scalar[DTYPE](1e-8)
            # f_n = sum of edge forces / num_tangent_dirs
            # f_tk = (f_edge_pos - f_edge_neg) * mu
            var f_e0 = force[c * NE + 0]
            var f_e1 = force[c * NE + 1]
            var f_e2 = force[c * NE + 2]
            var f_e3 = force[c * NE + 3]
            # `mju_decodePyramid`: the normal force is the SUM of the four edge
            # forces, NOT half of it. Both engines build each edge as
            # `Jn +- mu*Jt` with a FULL Jn (engine_core_constraint.c:1003), so
            # halving it made every pyramidal contact RECORD read half true
            # while qacc stayed correct — the solver works in edge forces and
            # only this write-back was wrong. Its two consumers are cfrc_ext
            # (hence Ant's contact_cost, a squared norm that had been costing a
            # quarter of what it should) and the quadruped force/torque
            # sensors. Fixed 2026-07-31.
            fn_c = f_e0 + f_e1 + f_e2 + f_e3
            var c_off = c * CONTACT_SIZE
            # ⚠ A FRICTIONLESS CONTACT HAS NO TANGENTIAL FORCE, and this
            # decode cannot know that from the edge forces alone. At condim 1
            # only edge 0 is live and edges 1..3 are zero, so `(f_e0 - f_e1)`
            # is `f_e0` and the record picks up a spurious `mu * f_n` of
            # friction. Measured on dog before this guard: `ft1/f_n = 0.9002`
            # on all three of its frictionless contacts — exactly the model's
            # default `friction="0.9"` — against MuJoCo's 0.
            #
            # It only became reachable when condim-1 contacts started producing
            # a row at all (see `_precompute_contact_friction`); before that
            # every edge force was zero and this read 0 for the right value by
            # accident, alongside an `f_n` of 0 that was simply wrong.
            #
            # `qacc` is NOT affected — that row's Jacobian is the pure normal,
            # so the solve stays frictionless. The damage is confined to the
            # record's consumers: `cfrc_ext` (hence contact-cost reward terms)
            # and the force/touch sensors, which is the fourth instance of this
            # write-back failure mode in this file's history.
            var dim_c = Int(
                rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_CONDIM])
            )
            if dim_c > 1:
                ft1_c = (f_e0 - f_e1) * safe_mu
                ft2_c = (f_e2 - f_e3) * safe_mu
            contacts[env, c_off + CONTACT_IDX_FORCE_N] = fn_c
            contacts[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_c
            contacts[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_c

        # NOTHING RUNS AFTER THE SOLVE ON THIS PATH. Joint limits,
        # dry-friction dofs, tendon equalities and connect/weld are all edge
        # rows of the Newton system above; calling `_equality_env` or
        # `_tendon_env` here would double-apply constraints the solve already
        # balanced, not complete them.
        #
        # Both post-passes were removed on 2026-08-12: the tendon one because
        # `build_tendon_equality_rows` covers spatial as well as fixed now, and
        # the weld one because `build_weld_equality_rows` feeds the edge list
        # above — the same defect-29a conversion the ELLIPTIC path got in
        # `d22144ee`.
        return  # PYRAMIDAL path complete

    # === ELLIPTIC path ===
    # === Cache loop-invariant contact data into local InlineArrays ===
    # Jn, the NT tangent Jacobians, mu, D_n, per-row D and friction, dist,
    # pos_bias and per-row bias never change during Newton iterations — load
    # once to avoid ~1000 workspace reads/iter.
    #
    # ⚠ TANGENT ROWS ARE A FLAT `[MC, NT]` BLOCK, NOT TWO NAMED ARRAYS. The
    # old `Jt1_c`/`Jt2_c`/`bt1_cache`/`bt2_cache` pairs WERE the condim-3
    # restriction — there was nowhere to put a torsional row. Index is
    # `c*NT + t` for scalars and `(c*NT + t)*nv + i` for Jacobians, i.e.
    # CONTACT-major, unlike the workspace's block-major `t*MC + c`; the solve
    # touches all of one contact's rows together and none of the arrays outlive
    # this function.
    var Jn_c = Scratch[Scalar[DTYPE], MC_CAP * V_CAP](max_contacts * nv, uninitialized=Scalar[DTYPE](0))
    # ⚠ DECLARED HERE, NOT AT FIRST NEED. `T_CAP` is part of the TYPE of the
    # caches below and of `elliptic_cone`'s parameters, so both must be the
    # SAME expression -- `cap[D.CAP_MAX_CONTACTS * NT]` and
    # `cap[D.CAP_MAX_CONTACTS] * NT` are numerically equal and distinct types.
    comptime T_CAP = cap[D.CAP_MAX_CONTACTS * NT]()
    var tn = max_contacts * NT
    var Jt_c = Scratch[Scalar[DTYPE], T_CAP * V_CAP](max_contacts * NT * nv, uninitialized=Scalar[DTYPE](0))
    var mu_cache = Scratch[Scalar[DTYPE], MC_CAP](max_contacts, uninitialized=Scalar[DTYPE](0))
    var D_n_cache = Scratch[Scalar[DTYPE], MC_CAP](max_contacts, uninitialized=Scalar[DTYPE](0))
    var D_t_cache = Scratch[Scalar[DTYPE], T_CAP](max_contacts * NT, uninitialized=Scalar[DTYPE](0))
    var fr_cache = Scratch[Scalar[DTYPE], T_CAP](max_contacts * NT, uninitialized=Scalar[DTYPE](0))
    var dist_cache = Scratch[Scalar[DTYPE], MC_CAP](max_contacts, uninitialized=Scalar[DTYPE](0))
    var pb_cache = Scratch[Scalar[DTYPE], MC_CAP](max_contacts, uninitialized=Scalar[DTYPE](0))
    var bt_cache = Scratch[Scalar[DTYPE], T_CAP](max_contacts * NT, uninitialized=Scalar[DTYPE](0))
    # How many of the NT rows this contact actually has (`dim-1`). 0 for a
    # frictionless (`condim="1"`) contact, which is one normal row and nothing
    # else — the cone then degenerates to `T == 0` and the zone logic reduces
    # to the one-sided normal constraint.
    var nt_cache = Scratch[Int, MC_CAP](max_contacts, fill=0)
    for c in range(nc):
        dist_cache[c] = rebind[Scalar[DTYPE]](
            solver[env, ws_c_dist_idx + c]
        )
        mu_cache[c] = rebind[Scalar[DTYPE]](solver[env, ws_mu_idx + c])
        D_n_cache[c] = rebind[Scalar[DTYPE]](solver[env, ws_D_n_idx + c])
        pb_cache[c] = rebind[Scalar[DTYPE]](
            solver[env, ws_pos_bias_idx + c]
        )
        nt_cache[c] = Int(
            rebind[Scalar[DTYPE]](solver[env, ws_ntc_idx + c])
        )
        for t in range(NT):
            D_t_cache[c * NT + t] = rebind[Scalar[DTYPE]](
                solver[env, ws_Dt_idx + t * max_contacts + c]
            )
            fr_cache[c * NT + t] = rebind[Scalar[DTYPE]](
                solver[env, ws_fr_idx + t * max_contacts + c]
            )
            bt_cache[c * NT + t] = rebind[Scalar[DTYPE]](
                solver[env, ws_bt_idx + t * max_contacts + c]
            )
        for i in range(nv):
            Jn_c[c * nv + i] = rebind[Scalar[DTYPE]](
                solver[env, ws_J_n_idx + c * nv + i]
            )
            for t in range(NT):
                Jt_c[(c * NT + t) * nv + i] = rebind[Scalar[DTYPE]](
                    solver[env, ws_Jt_idx + t * max_contacts * nv + c * nv + i]
                )

    # === Scalar rows: joint limits + dry-friction dofs ===
    # These used to be PGS post-passes that ran AFTER this solve, so the
    # contact rows were solved as if they did not exist. They are rows of the
    # same system — see constraints/scalar_rows.mojo for the measurement that
    # established this. J = sign * e_dof, so only (dof, sign) is stored.
    comptime S_CAP = max_scalar_rows_cap[D.CAP_NV, D.CAP_NJOINT]()
    var max_srows = max_scalar_rows(nv, njoint)
    var sr_dof = Scratch[Int, S_CAP](max_srows, fill=0)
    var sr_kind = Scratch[Int, S_CAP](max_srows, fill=0)
    var sr_sign = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_D = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_R = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_bias = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_floss = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var ns = build_scalar_rows[DTYPE, S_CAP](
        env, dims, qpos, qvel, joints, mmeta, dof_invweight0, m_inv,
        sr_dof, sr_kind, sr_sign, sr_D, sr_R, sr_bias, sr_floss,
    )
    var sr_jar = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_f = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_st = Scratch[Int, S_CAP](max_srows, fill=0)
    var sr_Js = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))

    # === Fixed-tendon EQUALITY rows (dense J) ===
    # These ran as a post-pass on this path until 2026-08-01, which is the same
    # defect `build_scalar_rows` above exists to fix, one constraint type over.
    # Measured on dm_control manipulator `bring_ball`, where the `coupling`
    # tendon holds thumb == finger while the grasped ball's two contacts break
    # that symmetry: the post-pass left qacc[thumb]/qacc[finger] at
    # +60.12/-58.11 where MuJoCo has -832.98/-839.37 — equal and OPPOSITE, i.e.
    # the row was not enforced against the contact solve at all. Poses where the
    # contacts are SYMMETRIC (a closed empty hand, 18 contact rows) were exact
    # even then, because the equality had nothing to correct — which is why this
    # needed a domain with an object in the hand to surface.
    #
    # A scalar row is stored as `(dof, sign)` to keep the elliptic core's local
    # memory at O(rows); an equality row needs a full nv Jacobian. That is the
    # cost this deferral was avoiding, and it is `ntendon * nv` floats — 22 for
    # manipulator, 88 for quadruped — next to the contact block's `MC * nv * 6`.
    #
    # The row is built by the SAME function the pyramidal edge list uses, so
    # both cones get bit-identical (J, D, bias).
    # Capacity covers THREE dense-J row kinds now: one equality row per
    # tendon, up to TWO limit rows per tendon (a `range` has two sides; only
    # one can be violated at a time, but the builder tries both), and the
    # connect/weld rows added for defect 29a (6 each).
    comptime EQ_CAP = cap[3 * D.CAP_NTENDON + 6 * D.CAP_NEQUALITY]()
    # the LIVE budget for the capacity guard below -- never the cap
    var max_eq_rows = 3 * ntendon + 6 * nequality
    var eq_J = Scratch[Scalar[DTYPE], EQ_CAP * V_CAP](max_eq_rows * nv, fill=Scalar[DTYPE](0))
    var eq_D = Scratch[Scalar[DTYPE], EQ_CAP](max_eq_rows, fill=Scalar[DTYPE](0))
    var eq_bias = Scratch[Scalar[DTYPE], EQ_CAP](max_eq_rows, fill=Scalar[DTYPE](0))
    var eq_kind = Scratch[Int, EQ_CAP](max_eq_rows, fill=0)
    # Per-row solver STATE (`SROW_QUADRATIC` / `SROW_SATISFIED`), the
    # dense-J twin of `sr_st`. It exists because these rows stopped being
    # unconditionally bilateral when the tendon limit rows joined the list.
    var eq_st = Scratch[Int, EQ_CAP](max_eq_rows, fill=SROW_QUADRATIC)
    var eq_jar = Scratch[Scalar[DTYPE], EQ_CAP](max_eq_rows, fill=Scalar[DTYPE](0))
    var eq_f = Scratch[Scalar[DTYPE], EQ_CAP](max_eq_rows, fill=Scalar[DTYPE](0))
    var eq_Js = Scratch[Scalar[DTYPE], EQ_CAP](max_eq_rows, fill=Scalar[DTYPE](0))
    var neq_rows = 0
    # ⚠⚠ TENDON LIMIT ROWS, WHICH THIS BRANCH DID NOT BUILD AT ALL.
    # `build_tendon_limit_rows` was called from the PYRAMIDAL branch (:1114)
    # and from the blocked kernel (:3363) and from nowhere else, so an
    # elliptic model's `<spatial limited="true">` produced no force —
    # `robotiq_2f85/scene.xml` hangs a free box from a 2 cm string and the
    # box fell straight through it. Two of three call sites is the failure
    # mode this file has had before (see the `build_weld_equality_rows`
    # conversion note below); the gate is
    # `test_tendon_rows_live_budget_vs_mujoco`.
    #
    # ⚠ THESE ROWS ARE ONE-SIDED. The `eq_*` list was BILATERAL-ONLY until
    # now — `eq_kind` was written and never read, and every consumer below
    # assumed `f = -D*jar` unconditionally. Each of those is now routed
    # through `scalar_row_state`/`scalar_row_force`, which for
    # `SROW_EQ_BILATERAL` returns exactly the old expression, so the
    # equality rows are bit-identical and only the limit rows can switch
    # off.
    comptime if may_exist[D.NTENDON]():
        build_tendon_limit_rows[
            DTYPE, V_CAP, EQ_CAP, BATCH
        ](
            env, dims, qvel, tendons, sites, geoms_w, bodies, joints, mmeta,
            subtree_com, cdof, xpos, xquat, m_inv,
            eq_J, eq_D, eq_bias, max_eq_rows, neq_rows,
        )
        # `build_tendon_limit_rows` does not take `kind_e` — the pyramidal
        # edge list seeds it to `SROW_LIMIT`, which is what these rows are.
        # `eq_kind` seeds to 0, and `SROW_LIMIT` IS 0, so this loop is
        # belt-and-braces; it is here because the seed being the right
        # constant is a coincidence of the enum's ordering, not a contract.
        for e in range(neq_rows):
            eq_kind[e] = SROW_LIMIT
    comptime if may_exist[D.NTENDON]():
        build_tendon_equality_rows[
            DTYPE, V_CAP, EQ_CAP, BATCH
        ](
            env, dims, qpos, qvel, tendons, sites, geoms_w, bodies, joints,
            mmeta,
            subtree_com, cdof, xpos, xquat, m_inv,
            eq_J, eq_D, eq_bias, eq_kind, max_eq_rows, neq_rows,
        )

    # === connect/weld EQUALITY rows (dense J) — defect 29a ===
    comptime if may_exist[D.NEQUALITY]():
        comptime EQR = 6 * cap[D.NEQUALITY]()
        comptime EQJ = 6 * cap[D.NEQUALITY]() * cap[D.NV]()
        var we_rows = 6 * nequality
        var we_K = Scratch[Scalar[DTYPE], EQR](we_rows, Scalar[DTYPE](1))
        var we_bias = Scratch[Scalar[DTYPE], EQR](we_rows, Scalar[DTYPE](0))
        var we_D = Scratch[Scalar[DTYPE], EQR](we_rows, Scalar[DTYPE](0))
        var we_J = Scratch[Scalar[DTYPE], EQJ](we_rows * nv, Scalar[DTYPE](0))
        var we_MinvJ = Scratch[Scalar[DTYPE], EQJ](we_rows * nv, Scalar[DTYPE](0))
        var nwe = build_weld_equality_rows[DTYPE, V_CAP](
            env, dims, qpos, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
            equality, body_invweight0, dof_invweight0, cdof, m_inv,
            we_K, we_bias, we_D, we_J, we_MinvJ,
        )
        for r in range(nwe):
            if neq_rows >= max_eq_rows:
                break
            for d in range(nv):
                eq_J[neq_rows * nv + d] = we_J[r * nv + d]
            # ⚠⚠ D IS 1/R, NOT 1/(k+R). `build_weld_equality_rows` returns the
            # PGS STEP SIZE 1/(k+R) in `we_D` because that is what the post-pass
            # iterates with; the Newton cost needs the row's STIFFNESS, which
            # MuJoCo defines as `efc_D = 1/R` (engine_core_constraint.c:1918).
            # Passing the step size instead left the weld unenforced at
            # |jar| ~ 60 with a converged gradient — docs 24.8. Recovered by the
            # same round-trip `build_tendon_equality_rows` uses, so both dense-J
            # row kinds get bit-identical D from identical (K, R).
            var R_recov = Scalar[DTYPE](1) / we_D[r] - we_K[r]
            if R_recov < Scalar[DTYPE](1e-14):
                R_recov = Scalar[DTYPE](1e-14)
            eq_D[neq_rows] = Scalar[DTYPE](1) / R_recov
            eq_bias[neq_rows] = we_bias[r]
            eq_kind[neq_rows] = SROW_EQ_BILATERAL
            neq_rows += 1

    # === Step 2: Initialize local InlineArrays from workspace ===
    var H = Scratch[Scalar[DTYPE], M_CAP](nv * nv, uninitialized=Scalar[DTYPE](0))
    var L_chol = Scratch[Scalar[DTYPE], M_CAP](nv * nv, uninitialized=Scalar[DTYPE](0))
    var qacc = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var qacc_sm = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var qfrc_sm = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var Ma = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var grad = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var search = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var Mv = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))

    # Load M into H (primal Hessian starts as M_hat)
    for k in range(nv * nv):
        H[k] = rebind[Scalar[DTYPE]](M[env, k])

    # Cache M locally — saves NV² workspace reads per Newton iteration (for Mv = M*search)
    var M_local = Scratch[Scalar[DTYPE], M_CAP](nv * nv, uninitialized=Scalar[DTYPE](0))
    for k in range(nv * nv):
        M_local[k] = H[k]

    # qacc_sm = unconstrained qacc (set by integrator), save a copy
    for i in range(nv):
        var q_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        qacc[i] = q_i
        qacc_sm[i] = q_i

    # Ma = M_local * qacc (uses cached M — no workspace reads)
    for i in range(nv):
        var s: Scalar[DTYPE] = 0
        for j in range(nv):
            s += M_local[i * nv + j] * qacc[j]
        Ma[i] = s

    # qfrc_sm = M * qacc (matching CPU's qfrc_smooth = M * qacc_smooth)
    # Using Ma directly avoids LDL round-trip error
    for i in range(nv):
        qfrc_sm[i] = Ma[i]

    # Same model-constant scale as the PYRAMIDAL path; see the note there.
    # `mj_solPrimal` is shared by both cones in MuJoCo, so the ELLIPTIC leg
    # took the identical pose-dependent-trace deviation and is corrected with
    # it rather than left as the odd one out.
    var scale_de = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_MEANINERTIA]
    ) * Scalar[DTYPE](nv if nv > 1 else 1)
    var scale = (
        Scalar[DTYPE](1) / scale_de
        if scale_de > Scalar[DTYPE](1e-10)
        else Scalar[DTYPE](1)
    )

    # === Mutable per-contact state: kept in InlineArrays, written to state buffer at end ===
    # Tangential quantities are flat `[MC, NT]`, indexed `c*NT + t`.
    var fn_arr = Scratch[Scalar[DTYPE], MC_CAP](max_contacts, uninitialized=Scalar[DTYPE](0))
    var ft_arr = Scratch[Scalar[DTYPE], T_CAP](tn, fill=Scalar[DTYPE](0))
    var jar_n_arr = Scratch[Scalar[DTYPE], MC_CAP](max_contacts, uninitialized=Scalar[DTYPE](0))
    var jar_t_arr = Scratch[Scalar[DTYPE], T_CAP](tn, fill=Scalar[DTYPE](0))
    var cs_arr = Scratch[Int, MC_CAP](max_contacts, uninitialized=0)

    # ── warmstart(): START AT THE CHEAPER OF `qacc_warmstart` AND
    # `qacc_smooth` (engine_forward.c:786) ───────────────────────────────────
    #
    # The PYRAMIDAL twin above carries the reasoning; two things differ here.
    #
    # 1. THE COST FUNCTION. `mj_constraintUpdate` prices an elliptic contact as
    #    ONE block (`ell_row_cost`), not as `dim` independent scalar rows.
    # 2. IT RUNS BEFORE STEP 3. Step 3 fills the state/force arrays the Hessian
    #    and the whole iteration read, so the choice has to be made while
    #    `qacc` is the only thing that has been decided; the alternative is a
    #    second copy of every one of those arrays to swap in.
    #
    # `jar_t_arr` / `ft_arr` are borrowed as the trial scratch precisely
    # because Step 3 overwrites both unconditionally one loop later.
    if mmeta[MODEL_META_IDX_WARMSTART_DISABLED] == Scalar[DTYPE](0) and (
        nc > 0 or ns > 0 or neq_rows > 0
    ):
        var qacc_w = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
        for i in range(nv):
            qacc_w[i] = rebind[Scalar[DTYPE]](qacc_warmstart[env, i])
        var cost_w = Scalar[DTYPE](0)
        var cost_s = Scalar[DTYPE](0)
        for cand in range(2):
            var acc_cost = Scalar[DTYPE](0)
            for c in range(nc):
                if dist_cache[c] >= Scalar[DTYPE](0):
                    continue
                var nt_c = nt_cache[c]
                var jn = pb_cache[c]
                for t in range(nt_c):
                    jar_t_arr[c * NT + t] = bt_cache[c * NT + t]
                for i in range(nv):
                    var qa_i = qacc_w[i] if cand == 0 else qacc_sm[i]
                    jn += Jn_c[c * nv + i] * qa_i
                    for t in range(nt_c):
                        jar_t_arr[c * NT + t] += (
                            Jt_c[(c * NT + t) * nv + i] * qa_i
                        )
                var f_n_try = Scalar[DTYPE](0)
                var zone = ell_state_force[DTYPE, NT, T_CAP](
                    nt_c, c * NT, jn, jar_t_arr,
                    mu_cache[c], D_n_cache[c], D_t_cache, fr_cache,
                    f_n_try, ft_arr,
                )
                acc_cost += ell_row_cost[DTYPE, NT, T_CAP](
                    zone, nt_c, c * NT, jn, jar_t_arr,
                    mu_cache[c], D_n_cache[c], D_t_cache, fr_cache,
                )
            for sr in range(ns):
                var qa_s = (
                    qacc_w[sr_dof[sr]] if cand == 0 else qacc_sm[sr_dof[sr]]
                )
                var jar_s = sr_bias[sr] + sr_sign[sr] * qa_s
                var st_s = scalar_row_state[DTYPE](
                    sr_kind[sr], jar_s, sr_R[sr], sr_floss[sr]
                )
                acc_cost += scalar_row_cost[DTYPE](
                    st_s, jar_s, sr_D[sr], sr_R[sr], sr_floss[sr]
                )
            for e in range(neq_rows):
                var jar_e = eq_bias[e]
                for dd in range(nv):
                    var qa_e = qacc_w[dd] if cand == 0 else qacc_sm[dd]
                    jar_e += eq_J[e * nv + dd] * qa_e
                var st_e = scalar_row_state[DTYPE](
                    eq_kind[e], jar_e, Scalar[DTYPE](0), Scalar[DTYPE](0)
                )
                acc_cost += scalar_row_cost[DTYPE](
                    st_e, jar_e, eq_D[e], Scalar[DTYPE](0), Scalar[DTYPE](0)
                )
            if cand == 0:
                cost_w = acc_cost
            else:
                cost_s = acc_cost
        # Gauss, identically 0 at `qacc_smooth`. `Ma` is written tentatively
        # and restored from `qfrc_sm`, which IS `M * qacc_sm` bit for bit.
        for i in range(nv):
            var s_i = Scalar[DTYPE](0)
            for j in range(nv):
                s_i += M_local[i * nv + j] * qacc_w[j]
            Ma[i] = s_i
            cost_w += (
                Scalar[DTYPE](0.5)
                * (s_i - qfrc_sm[i])
                * (qacc_w[i] - qacc_sm[i])
            )
        if cost_w <= cost_s:
            for i in range(nv):
                qacc[i] = qacc_w[i]
        else:
            for i in range(nv):
                Ma[i] = qfrc_sm[i]

    # === Step 3: Compute initial jar and forces via 3-zone cone logic ===
    for c in range(nc):
        var nt_c = nt_cache[c]
        if dist_cache[c] >= Scalar[DTYPE](0):
            fn_arr[c] = 0
            jar_n_arr[c] = 0
            for t in range(NT):
                ft_arr[c * NT + t] = 0
                jar_t_arr[c * NT + t] = 0
            cs_arr[c] = ELL_SATISFIED
            continue

        var jar_n: Scalar[DTYPE] = pb_cache[c]
        for t in range(nt_c):
            jar_t_arr[c * NT + t] = bt_cache[c * NT + t]
        for i in range(nv):
            var qa_i = qacc[i]
            jar_n += Jn_c[c * nv + i] * qa_i
            for t in range(nt_c):
                jar_t_arr[c * NT + t] += Jt_c[(c * NT + t) * nv + i] * qa_i
        jar_n_arr[c] = jar_n

        var f_n_c = Scalar[DTYPE](0)
        cs_arr[c] = ell_state_force[DTYPE, NT, T_CAP](
            nt_c, c * NT, jar_n, jar_t_arr,
            mu_cache[c], D_n_cache[c], D_t_cache, fr_cache,
            f_n_c, ft_arr,
        )
        fn_arr[c] = f_n_c

    # Scalar rows: same 3-zone logic, one dof each.
    for s in range(ns):
        var jar_s = sr_bias[s] + sr_sign[s] * qacc[sr_dof[s]]
        sr_jar[s] = jar_s
        var st = scalar_row_state[DTYPE](
            sr_kind[s], jar_s, sr_R[s], sr_floss[s]
        )
        sr_st[s] = st
        sr_f[s] = scalar_row_force[DTYPE](st, jar_s, sr_D[s], sr_floss[s])

    # Dense-J rows. `SROW_EQ_BILATERAL` is unconditionally QUADRATIC with
    # `f = -D*jar`, which is what this loop used to hardcode; `SROW_LIMIT`
    # (the tendon limit rows above) switches off once `jar >= 0`. Routing
    # both through the shared classifier keeps the bilateral answer
    # bit-identical — `scalar_row_state(SROW_EQ_BILATERAL, ...)` is
    # `SROW_QUADRATIC` and `scalar_row_force(SROW_QUADRATIC, ...)` is
    # `-D*jar` — while giving the one-sided rows their state.
    #
    # ⚠ `R` AND `floss` ARE PASSED AS ZERO ON PURPOSE. Both are read only on
    # the `SROW_FRICTION` branch, and no dense-J row is ever friction; the
    # dry-friction dofs are SCALAR rows (`sr_*`) with their own arrays.
    for e in range(neq_rows):
        var jar_e = eq_bias[e]
        for d in range(nv):
            jar_e += eq_J[e * nv + d] * qacc[d]
        eq_jar[e] = jar_e
        var st_e = scalar_row_state[DTYPE](
            eq_kind[e], jar_e, Scalar[DTYPE](0), Scalar[DTYPE](0)
        )
        eq_st[e] = st_e
        eq_f[e] = scalar_row_force[DTYPE](
            st_e, jar_e, eq_D[e], Scalar[DTYPE](0)
        )

    # === Step 4: Build Hessian H = M + J^T*D*J (cone-aware, using cached Jacobians) ===
    # Scalar rows contribute D only on their own dof (J = sign*e_dof, so
    # J^T*J = e_dof*e_dof^T — the sign squares away).
    for s in range(ns):
        if sr_st[s] == SROW_QUADRATIC:
            var d = sr_dof[s]
            H[d * nv + d] += sr_D[s]
    # Dense-J rows contribute a full rank-1 outer product rather than a
    # diagonal bump — and only while QUADRATIC. A satisfied one-sided row
    # carries no force and no curvature.
    for e in range(neq_rows):
        if eq_st[e] != SROW_QUADRATIC:
            continue
        for a in range(nv):
            var Ja = eq_J[e * nv + a]
            if Ja == Scalar[DTYPE](0):
                continue
            for b in range(nv):
                H[a * nv + b] += eq_D[e] * Ja * eq_J[e * nv + b]
    comptime HN = (NT + 1) * (NT + 1)
    ell_add_contact_hessian[
        DTYPE, MC_CAP, NT, T_CAP, V_CAP, M_CAP, HN
    ](
        nc, cs_arr, nt_cache, Jn_c, Jt_c, jar_n_arr, jar_t_arr,
        mu_cache, D_n_cache, D_t_cache, fr_cache, H, nv,
    )

    # Cholesky factorize H (with regularization on rank deficiency)
    var chol_ok_gpu = chol_factor_inline[DTYPE, M_CAP](H, L_chol, nv)
    if not chol_ok_gpu:
        for i in range(nv):
            H[i * nv + i] = H[i * nv + i] + Scalar[DTYPE](1e-6)
        _ = chol_factor_inline[DTYPE, M_CAP](H, L_chol, nv)

    # === Precompute qfrc_c = J^T * force (replaces per-iteration gradient workspace reads) ===
    # Updated after each force update instead of recomputing from workspace each gradient step.
    var qfrc_c = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    for i in range(nv):
        qfrc_c[i] = Scalar[DTYPE](0)
    for c in range(nc):
        if cs_arr[c] == ELL_SATISFIED:
            continue
        for i in range(nv):
            var acc = Jn_c[c * nv + i] * fn_arr[c]
            for t in range(nt_cache[c]):
                acc += Jt_c[(c * NT + t) * nv + i] * ft_arr[c * NT + t]
            qfrc_c[i] += acc
    for s in range(ns):
        qfrc_c[sr_dof[s]] += sr_sign[s] * sr_f[s]
    for e in range(neq_rows):
        for d in range(nv):
            qfrc_c[d] += eq_J[e * nv + d] * eq_f[e]

    # === Step 5: Newton iteration loop ===
    for _iter in range(NEWTON_ITER_GPU):
        # ⚠⚠ NO CONSTRAINT ROWS: MUJOCO RETURNS, AND WE USED TO SOLVE.
        # `mj_fwdConstraint` (engine_forward.c:884) is explicit —
        #     if (!nefc) { mju_copy(d->qacc, d->qacc_smooth, nv); return; }
        # — and this loop had no such guard. With no rows the warmstart block
        # above is already skipped (its own `nc > 0 or ns > 0 or neq_rows > 0` test), so `qacc` still
        # holds `qacc_smooth` and the gradient is IDENTICALLY ZERO. Every
        # iteration then factors `H = M` and solves for a search direction of
        # zero: the answer cannot move, so breaking here writes back exactly
        # what a full pass would have — this is a no-op, not an approximation.
        #
        # ⚠ IT IS NOT A MICRO-OPTIMISATION. P0 measured it at 32.3 of 46.9 ms
        # per step on the k=9 park scene — 69% of GPU time, 78% of the whole
        # parked-slot cost — spent re-factorising a matrix `ldl_factor` factored
        # two kernels earlier, for a problem with no constraints in it. The
        # cooperative Cholesky's `d_j` reduction is `if tid == 0: for k in
        # range(j)`, O(NV^2) on ONE thread, so it is also the fastest-growing
        # term in the sweep.
        # See docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md §0.0.1.
        if nc == 0 and ns == 0 and neq_rows == 0:
            break
        if _iter >= niter_rt:
            break
        # Gradient = Ma - qfrc_sm - qfrc_c (pure InlineArray reads — no workspace access)
        var grad_norm_sq: Scalar[DTYPE] = 0
        for i in range(nv):
            grad[i] = Ma[i] - qfrc_sm[i] - qfrc_c[i]
            grad_norm_sq += grad[i] * grad[i]

        # Convergence check — see the PYRAMIDAL twin for why not on pass 0.
        if _iter > 0 and scale * sqrt(grad_norm_sq) < tol_rt:
            break
        comptime if _ELL_TRACE:
            var _nst = 0
            for _c in range(nc):
                if cs_arr[_c] == ELL_CONE:
                    _nst += 1
            print("  [ell]", _iter, "g", scale * sqrt(grad_norm_sq),
                  "cone", _nst, "of", nc)

        # Newton direction: search = -H^{-1} * grad
        chol_solve_inline[DTYPE, M_CAP, V_CAP](L_chol, grad, search, nv)
        var search_ok_gpu = True
        for i in range(nv):
            search[i] = -search[i]
            if search[i] != search[i]:
                search_ok_gpu = False
        if not search_ok_gpu:
            break

        # Mv = M_local * search (InlineArray reads only — no workspace access)
        for i in range(nv):
            var s: Scalar[DTYPE] = 0
            for j in range(nv):
                s += M_local[i * nv + j] * search[j]
            Mv[i] = s

        # Precompute J * search per contact (using cached Jacobians — no workspace access)
        var Js_n = Scratch[Scalar[DTYPE], MC_CAP](max_contacts, uninitialized=Scalar[DTYPE](0))
        var Js_t = Scratch[Scalar[DTYPE], T_CAP](tn, fill=Scalar[DTYPE](0))
        for c in range(nc):
            var nt_c = nt_cache[c]
            if dist_cache[c] >= Scalar[DTYPE](0):
                Js_n[c] = 0
                for t in range(NT):
                    Js_t[c * NT + t] = 0
                continue
            var js_n: Scalar[DTYPE] = 0
            for t in range(NT):
                Js_t[c * NT + t] = 0
            for i in range(nv):
                var s_i = search[i]
                js_n += Jn_c[c * nv + i] * s_i
                for t in range(nt_c):
                    Js_t[c * NT + t] += Jt_c[(c * NT + t) * nv + i] * s_i
            Js_n[c] = js_n
        for s in range(ns):
            sr_Js[s] = sr_sign[s] * search[sr_dof[s]]
        for e in range(neq_rows):
            var jv = Scalar[DTYPE](0)
            for d in range(nv):
                jv += eq_J[e * nv + d] * search[d]
            eq_Js[e] = jv

        # Analytical Newton linesearch (matches CPU primal_linesearch_with_D)
        # Gauss coefficients for derivative: d_gauss/dalpha = ga*alpha + gb
        var ga: Scalar[DTYPE] = 0
        var gb: Scalar[DTYPE] = 0
        for i in range(nv):
            ga += Mv[i] * search[i]
            gb += (Ma[i] - qfrc_sm[i]) * search[i]

        # Evaluate d1, d2 at alpha=0
        var p0_d1 = gb
        var p0_d2 = ga
        for c in range(nc):
            if dist_cache[c] >= Scalar[DTYPE](0):
                continue
            ell_line_deriv[DTYPE, NT, T_CAP](
                nt_cache[c], c * NT, Scalar[DTYPE](0),
                jar_n_arr[c], jar_t_arr, Js_n[c], Js_t,
                mu_cache[c], D_n_cache[c], D_t_cache, fr_cache,
                p0_d1, p0_d2,
            )
        # Scalar rows. d(cost)/dalpha = -f*Jv in EVERY state, and the second
        # derivative is D*Jv^2 only where the row is quadratic.
        for s in range(ns):
            p0_d1 += -sr_f[s] * sr_Js[s]
            if sr_st[s] == SROW_QUADRATIC:
                p0_d2 += sr_D[s] * sr_Js[s] * sr_Js[s]
        for e in range(neq_rows):
            p0_d1 += -eq_f[e] * eq_Js[e]
            if eq_st[e] == SROW_QUADRATIC:
                p0_d2 += eq_D[e] * eq_Js[e] * eq_Js[e]
        # ⚠ MuJoCo FLOORS `deriv[1]` ONLY WHEN IT IS <= 0 (engine_solver.c:1648,
        # a should-not-occur convexity violation) and to `mjMINVAL` = 1e-15.
        # Testing `< PRIMAL_MINVAL_GPU` inflates a legitimately SMALL POSITIVE
        # curvature — which is what the line's second derivative IS near the
        # optimum — and crushes `alpha = -d1/d2`. See the PYRAMIDAL twin in
        # `primal.mojo` for the measurement.
        if p0_d2 <= Scalar[DTYPE](0):
            p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

        var alpha: Scalar[DTYPE] = 0
        if p0_d1 < Scalar[DTYPE](0):
            # Phase 1: initial Newton step
            var p1_alpha = -p0_d1 / p0_d2

            var snorm_sq: Scalar[DTYPE] = 0
            for i in range(nv):
                snorm_sq += search[i] * search[i]
            var gtol = (
                tol_rt * lstol_rt
                * sqrt(snorm_sq)
                / scale
            )
            var gtol_sq = gtol * gtol

            # Inline eval at p1_alpha
            var p1_d1 = ga * p1_alpha + gb
            var p1_d2_v = ga
            for c in range(nc):
                if dist_cache[c] >= Scalar[DTYPE](0):
                    continue
                ell_line_deriv[DTYPE, NT, T_CAP](
                    nt_cache[c], c * NT, p1_alpha,
                    jar_n_arr[c], jar_t_arr, Js_n[c], Js_t,
                    mu_cache[c], D_n_cache[c], D_t_cache, fr_cache,
                    p1_d1, p1_d2_v,
                )
            for s in range(ns):
                var tj = sr_jar[s] + p1_alpha * sr_Js[s]
                var tst = scalar_row_state[DTYPE](
                    sr_kind[s], tj, sr_R[s], sr_floss[s]
                )
                p1_d1 += (
                    -scalar_row_force[DTYPE](tst, tj, sr_D[s], sr_floss[s])
                    * sr_Js[s]
                )
                if tst == SROW_QUADRATIC:
                    p1_d2_v += sr_D[s] * sr_Js[s] * sr_Js[s]
            for e in range(neq_rows):
                var tje = eq_jar[e] + p1_alpha * eq_Js[e]
                var tste = scalar_row_state[DTYPE](
                    eq_kind[e], tje, Scalar[DTYPE](0), Scalar[DTYPE](0)
                )
                p1_d1 += (
                    -scalar_row_force[DTYPE](
                        tste, tje, eq_D[e], Scalar[DTYPE](0)
                    )
                    * eq_Js[e]
                )
                if tste == SROW_QUADRATIC:
                    p1_d2_v += eq_D[e] * eq_Js[e] * eq_Js[e]
            # Same rule as `p0_d2` above.
            if p1_d2_v <= Scalar[DTYPE](0):
                p1_d2_v = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

            alpha = p1_alpha
            if p1_d1 * p1_d1 >= gtol_sq:
                # Phase 2: one-sided Newton pursuit
                var dir_s = Scalar[DTYPE](-1) if p1_d1 > Scalar[DTYPE](
                    0
                ) else Scalar[DTYPE](1)
                var p2_alpha: Scalar[DTYPE] = 0
                var p2_d1 = p0_d1
                var bracket = False
                for _ls in range(LINESEARCH_ITER):
                    if _ls >= lsiter_rt:
                        break
                    p2_alpha = p1_alpha
                    p2_d1 = p1_d1
                    if p1_d2_v > Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        p1_alpha = p1_alpha - p1_d1 / p1_d2_v
                    else:
                        p1_alpha = p1_alpha + dir_s
                    # Eval at new p1_alpha
                    p1_d1 = ga * p1_alpha + gb
                    p1_d2_v = ga
                    for c in range(nc):
                        if dist_cache[c] >= Scalar[DTYPE](0):
                            continue
                        ell_line_deriv[DTYPE, NT, T_CAP](
                            nt_cache[c], c * NT, p1_alpha,
                            jar_n_arr[c], jar_t_arr, Js_n[c], Js_t,
                            mu_cache[c], D_n_cache[c], D_t_cache, fr_cache,
                            p1_d1, p1_d2_v,
                        )
                    for s in range(ns):
                        var tj = sr_jar[s] + p1_alpha * sr_Js[s]
                        var tst = scalar_row_state[DTYPE](
                            sr_kind[s], tj, sr_R[s], sr_floss[s]
                        )
                        p1_d1 += (
                            -scalar_row_force[DTYPE](
                                tst, tj, sr_D[s], sr_floss[s]
                            )
                            * sr_Js[s]
                        )
                        if tst == SROW_QUADRATIC:
                            p1_d2_v += sr_D[s] * sr_Js[s] * sr_Js[s]
                    for e in range(neq_rows):
                        var tje = eq_jar[e] + p1_alpha * eq_Js[e]
                        var tste = scalar_row_state[DTYPE](
                            eq_kind[e], tje, Scalar[DTYPE](0),
                            Scalar[DTYPE](0),
                        )
                        p1_d1 += (
                            -scalar_row_force[DTYPE](
                                tste, tje, eq_D[e], Scalar[DTYPE](0)
                            )
                            * eq_Js[e]
                        )
                        if tste == SROW_QUADRATIC:
                            p1_d2_v += eq_D[e] * eq_Js[e] * eq_Js[e]
                    # Same rule as `p0_d2` above.
                    if p1_d2_v <= Scalar[DTYPE](0):
                        p1_d2_v = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    if p1_d1 * p1_d1 < gtol_sq:
                        alpha = p1_alpha
                        break
                    if p1_d1 * dir_s > Scalar[DTYPE](0):
                        bracket = True
                        break
                if bracket:
                    # Phase 3: bracketed bisection
                    for _ls in range(LINESEARCH_ITER):
                        if _ls >= lsiter_rt:
                            break
                        var mid = (p1_alpha + p2_alpha) * Scalar[DTYPE](0.5)
                        var mid_d1 = ga * mid + gb
                        # `mid_d2` is written and discarded — the bisection
                        # only brackets on the sign of `d1`. Kept so the
                        # bracketing evaluates the SAME function as the two
                        # Newton phases above rather than a hand-trimmed copy
                        # of it, which is how the four inlined versions of
                        # this block used to differ from each other.
                        var mid_d2 = Scalar[DTYPE](0)
                        for c in range(nc):
                            if dist_cache[c] >= Scalar[DTYPE](0):
                                continue
                            ell_line_deriv[DTYPE, NT, T_CAP](
                                nt_cache[c], c * NT, mid,
                                jar_n_arr[c], jar_t_arr, Js_n[c], Js_t,
                                mu_cache[c], D_n_cache[c], D_t_cache,
                                fr_cache, mid_d1, mid_d2,
                            )
                        for s in range(ns):
                            var tj = sr_jar[s] + mid * sr_Js[s]
                            var tst = scalar_row_state[DTYPE](
                                sr_kind[s], tj, sr_R[s], sr_floss[s]
                            )
                            mid_d1 += (
                                -scalar_row_force[DTYPE](
                                    tst, tj, sr_D[s], sr_floss[s]
                                )
                                * sr_Js[s]
                            )
                        for e in range(neq_rows):
                            var tje = eq_jar[e] + mid * eq_Js[e]
                            var tste = scalar_row_state[DTYPE](
                                eq_kind[e], tje, Scalar[DTYPE](0),
                                Scalar[DTYPE](0),
                            )
                            mid_d1 += (
                                -scalar_row_force[DTYPE](
                                    tste, tje, eq_D[e], Scalar[DTYPE](0)
                                )
                                * eq_Js[e]
                            )
                        if mid_d1 * mid_d1 < gtol_sq:
                            p1_alpha = mid
                            p1_d1 = mid_d1
                            break
                        if mid_d1 * p1_d1 > Scalar[DTYPE](0):
                            p1_alpha = mid
                            p1_d1 = mid_d1
                        else:
                            p2_alpha = mid
                            p2_d1 = mid_d1
                        if (p1_alpha - p2_alpha) * (
                            p1_alpha - p2_alpha
                        ) < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                            break
                    if p2_d1 * p2_d1 < p1_d1 * p1_d1:
                        alpha = p2_alpha
                    else:
                        alpha = p1_alpha
                elif p1_d1 * p1_d1 >= gtol_sq:
                    alpha = p1_alpha

        comptime if _ELL_TRACE:
            print("       alpha", alpha)
        # If alpha is negligible, stop
        if alpha < Scalar[DTYPE](1e-12):
            break

        # Update qacc and Ma
        for i in range(nv):
            qacc[i] = qacc[i] + alpha * search[i]
            Ma[i] = Ma[i] + alpha * Mv[i]

        # Recompute jar and forces (using cached Jacobians — no workspace reads)
        var state_changed = False
        for c in range(nc):
            if dist_cache[c] >= Scalar[DTYPE](0):
                continue
            var old_cs = cs_arr[c]
            var nt_c = nt_cache[c]
            var jar_n: Scalar[DTYPE] = pb_cache[c]
            for t in range(nt_c):
                jar_t_arr[c * NT + t] = bt_cache[c * NT + t]
            for i in range(nv):
                var qa_i = qacc[i]
                jar_n += Jn_c[c * nv + i] * qa_i
                for t in range(nt_c):
                    jar_t_arr[c * NT + t] += Jt_c[(c * NT + t) * nv + i] * qa_i
            jar_n_arr[c] = jar_n

            var f_n_c = Scalar[DTYPE](0)
            cs_arr[c] = ell_state_force[DTYPE, NT, T_CAP](
                nt_c, c * NT, jar_n, jar_t_arr,
                mu_cache[c], D_n_cache[c], D_t_cache, fr_cache,
                f_n_c, ft_arr,
            )
            fn_arr[c] = f_n_c
            if cs_arr[c] != old_cs:
                state_changed = True
        for s in range(ns):
            var old_st = sr_st[s]
            var jar_s = sr_bias[s] + sr_sign[s] * qacc[sr_dof[s]]
            sr_jar[s] = jar_s
            var st = scalar_row_state[DTYPE](
                sr_kind[s], jar_s, sr_R[s], sr_floss[s]
            )
            sr_st[s] = st
            sr_f[s] = scalar_row_force[DTYPE](st, jar_s, sr_D[s], sr_floss[s])
            if st != old_st:
                state_changed = True
        # Dense-J rows. A BILATERAL row still cannot flip `state_changed` —
        # its state is `SROW_QUADRATIC` at every `jar` — but a tendon LIMIT
        # row can, and the Hessian below has to be rebuilt when it does.
        for e in range(neq_rows):
            var jar_e = eq_bias[e]
            for d in range(nv):
                jar_e += eq_J[e * nv + d] * qacc[d]
            eq_jar[e] = jar_e
            var old_ste = eq_st[e]
            var ste = scalar_row_state[DTYPE](
                eq_kind[e], jar_e, Scalar[DTYPE](0), Scalar[DTYPE](0)
            )
            eq_st[e] = ste
            eq_f[e] = scalar_row_force[DTYPE](
                ste, jar_e, eq_D[e], Scalar[DTYPE](0)
            )
            if ste != old_ste:
                state_changed = True

        # Recompute qfrc_c = J^T * updated forces (all InlineArray ops)
        for i in range(nv):
            qfrc_c[i] = Scalar[DTYPE](0)
        for c in range(nc):
            if cs_arr[c] == ELL_SATISFIED:
                continue
            for i in range(nv):
                var acc = Jn_c[c * nv + i] * fn_arr[c]
                for t in range(nt_cache[c]):
                    acc += Jt_c[(c * NT + t) * nv + i] * ft_arr[c * NT + t]
                qfrc_c[i] += acc
        for s in range(ns):
            qfrc_c[sr_dof[s]] += sr_sign[s] * sr_f[s]
        for e in range(neq_rows):
            for d in range(nv):
                qfrc_c[d] += eq_J[e * nv + d] * eq_f[e]

        # ── Hessian rebuild ──────────────────────────────────────────────
        #
        # ⚠⚠ A CONE BLOCK DEPENDS ON `jar`, NOT ONLY ON THE STATE, AND MuJoCo
        # REBUILDS IT EVERY ITERATION. `HessianIncremental`
        # (engine_solver.c:2118) does incremental cholUpdates for the rows that
        # entered or left QUADRATIC — those carry a constant `D`, so a
        # state-gated rebuild is right for them — and then calls
        # `HessianCone` UNCONDITIONALLY whenever any cone row exists, which
        # recomputes `con->H` from the CURRENT `jar` and re-applies it.
        #
        # Gating the whole rebuild on `state_changed` froze the cone blocks at
        # iteration 0's `jar`, which turns Newton into a quasi-Newton with a
        # stale Hessian. The symptom is not divergence but a PERIOD-2 ZIG-ZAG:
        # measured on `unitree_go1` at `impratio="100"`, `alpha` alternated
        # 0.0895 / 0.2317 while `scale*|grad|` fell ~5% per PAIR of steps —
        # MuJoCo converged that pose in 6 iterations and we needed ~800.
        # `unitree_go1` was board #4 at 3.256e-07 and it was this.
        var cone_live = False
        for c in range(nc):
            if cs_arr[c] == ELL_CONE:
                cone_live = True
                break
        if state_changed or cone_live:
            for k in range(nv * nv):
                H[k] = M_local[k]
            for s in range(ns):
                if sr_st[s] == SROW_QUADRATIC:
                    var d = sr_dof[s]
                    H[d * nv + d] += sr_D[s]
            # …and only the QUADRATIC dense-J rows. Bilateral rows always
            # are; a satisfied tendon limit is not.
            for e in range(neq_rows):
                if eq_st[e] != SROW_QUADRATIC:
                    continue
                for a in range(nv):
                    var Ja = eq_J[e * nv + a]
                    if Ja == Scalar[DTYPE](0):
                        continue
                    for b in range(nv):
                        H[a * nv + b] += eq_D[e] * Ja * eq_J[e * nv + b]
            ell_add_contact_hessian[
                DTYPE, MC_CAP, NT, T_CAP, V_CAP, M_CAP, HN
            ](
                nc, cs_arr, nt_cache, Jn_c, Jt_c, jar_n_arr, jar_t_arr,
                mu_cache, D_n_cache, D_t_cache, fr_cache, H, nv,
            )
            var chol_ok_gpu2 = chol_factor_inline[DTYPE, M_CAP](
                H, L_chol, nv
            )
            if not chol_ok_gpu2:
                for i in range(nv):
                    H[i * nv + i] = H[i * nv + i] + Scalar[DTYPE](1e-6)
                _ = chol_factor_inline[DTYPE, M_CAP](H, L_chol, nv)

    # ── mj_solNoSlip (ELLIPTIC branch) ─────────────────────────────────────
    # The friction-only Gauss-Seidel sweep, with the normal forces frozen, run
    # after the primal solve. Off unless the model asks for it
    # (`<option noslip_iterations>`); EVERY dm_control manipulation model does,
    # and there it is first-order — `reach_site_features` moves `qacc` by
    # 7.4e+2 on step 1 with the option alone. Until 2026-08-13 this path had no
    # call at all and `ModelDefFromXML` refused to build an elliptic model that
    # asked for the pass, rather than let it vanish quietly.
    #
    # ELLIPTIC branch specifically — `noslip_elliptic`, not `noslip_pyramidal`.
    # The two are different algorithms over different row layouts (see
    # `noslip.mojo`), and this is the dispatch the module's header calls the
    # caller's obligation. It sits inside the already-cone-split solve body,
    # so there is no runtime test to get wrong.
    comptime if NOSLIP_ITER > 0:
        noslip_elliptic[
            DTYPE, MC_CAP, NT, T_CAP, V_CAP, S_CAP, EQ_CAP
        ](
            env,
            nc,
            ns,
            neq_rows,
            dims,
            m_inv,
            nt_cache,
            Jn_c, Jt_c,
            fr_cache, D_n_cache, D_t_cache,
            pb_cache, bt_cache,
            sr_dof, sr_kind, sr_sign, sr_R, sr_bias, sr_floss,
            eq_J, eq_D, eq_bias,
            qacc_sm,
            # `scale` is the SAME model constant the primal loop above uses —
            # `1 / (meaninertia * max(1, nv))`, already computed and guarded.
            scale,
            # ⚠ FROM META, NOT `NOSLIP_TOLERANCE`. dm_control's manipulation
            # models set 0; the constant is only the absent-attribute default.
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NOSLIP_TOLERANCE]),
            # ⚠ FROM META TOO — see the per-env pyramidal call for why the
            # comptime `NOSLIP_ITER` is an enable and not the count.
            Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NOSLIP_ITERATIONS])),
            qacc,
            fn_arr, ft_arr,
            jar_n_arr, jar_t_arr,
            sr_f, sr_jar,
            eq_f, eq_jar,
            qfrc_c,
        )

    # Write solved qacc back to workspace
    for i in range(nv):
        qacc_constrained[env, i] = qacc[i]

    # Write forces to state buffer for display/warmstart (directly from
    # InlineArrays).
    #
    # ⚠ THE TORSIONAL AND ROLLING SLOTS ARE WRITTEN NOW. `rne_post` and
    # `cfrc_ext_gpu` have READ `CONTACT_IDX_FORCE_TORSION`/`_ROLL1`/`_ROLL2`
    # since they were added, and NOTHING wrote them — so a condim-4 or -6
    # contact contributed its normal and slide forces to `cfrc_ext` and
    # silently dropped its torque. The pyramidal path still does not write
    # them; its forces live on edge rows, not per-direction ones, so
    # recovering them there is a separate change.
    for c in range(nc):
        var c_off = c * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_FORCE_N] = fn_arr[c]
        contacts[env, c_off + CONTACT_IDX_FORCE_T1] = 0
        contacts[env, c_off + CONTACT_IDX_FORCE_T2] = 0
        contacts[env, c_off + CONTACT_IDX_FORCE_TORSION] = 0
        contacts[env, c_off + CONTACT_IDX_FORCE_ROLL1] = 0
        contacts[env, c_off + CONTACT_IDX_FORCE_ROLL2] = 0
        for t in range(nt_cache[c]):
            var slot = CONTACT_IDX_FORCE_T1 + t
            if t >= 2:
                slot = CONTACT_IDX_FORCE_TORSION + (t - 2)
            contacts[env, c_off + slot] = ft_arr[c * NT + t]

    # NOTHING RUNS AFTER THE SOLVE ON THIS PATH ANY MORE. Joint limits,
    # dry-friction dofs, tendon equalities (`build_scalar_rows` /
    # `build_tendon_equality_rows`) and connect/weld (defect 29a,
    # `build_weld_equality_rows`) are all rows of the Newton system above —
    # solving any of them after the contacts is what made the contact force
    # wrong, twice, on two different constraint types.
    #
    # The last holdout was the SPATIAL tendon equality, kept here by a
    # `SKIP_FIXED` guard that deliberately let it past. The pass it was
    # handed to could not express it (no spatial branch, zero Jacobian), so
    # the constraint was silently absent rather than merely mis-sequenced.


def _newton_solve_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
    equality: LayoutTensor[
        DTYPE, Layout.row_major(NEQUALITY, MODEL_EQ_SIZE), MutAnyOrigin
    ],
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    geoms_w: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, Layout.row_major(NV), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    qacc_warmstart: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _newton_solve_env[
        DTYPE,
        CONE_TYPE,
        BATCH,
        SOLVER_WS, MAX_CONDIM=MAX_CONDIM, NOSLIP_ITER=NOSLIP_ITER](
        env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qpos, qvel, xpos, xquat, subtree_com, contacts, smeta, joints,
        bodies, mmeta, trees, equality, tendons, sites, geoms_w, body_invweight0,
        dof_invweight0, cdof, M, m_inv, qacc_constrained, qacc_warmstart,
        solver,
    )


def solve_newton[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
    # Per-env spill size for `Je`; 0 = it fits threadgroup memory. Comes
    # from `je_budget.je_ws_size` via the integrator — never computed here.
    JE_WS: Int = 0,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    mut cscratch: ContactScratch[DTYPE, D, BATCH, JE_WS],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Primal Newton contact solve into `scratch.qacc_constrained` (+ solved
    forces back into `d.contacts` for warm-starting/display), both targets,
    one body. Standalone entry — same signature family as
    `solve_contacts` so callers can swap solvers later.

    ELLIPTIC: joint limits, equality constraints, and fixed tendons run
    INSIDE at the legacy position (after the Newton core, 50 iterations).
    PYRAMIDAL: limits are edge rows inside the Newton optimization;
    equality/tendon after.

    Newton uses a PREFIX (35*MC + 6*MC*NV) of the PGS-sized
    `cscratch.solver` tensor (81*MC + 12*MC*NV) — no separate scratch.
    """
    comptime MC = _max_one[D.MAX_CONTACTS]()
    comptime SOLVER_WS = 81 * MC + 12 * MC * D.NV

    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
    comptime L_CON = Layout.row_major(BATCH, D.MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_JOINT = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    # ⚠ FLAT, matching `build_dof_segments`. `Model.L_TREE` is the 2-D view.
    comptime L_TREES = Layout.row_major(D.NV * MODEL_TREE_SIZE)
    comptime L_EQ = Layout.row_major(D.NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TEN = Layout.row_major(D.NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(D.NSITE, MODEL_SITE_SIZE)
    comptime L_GEOM_W = Layout.row_major(D.NGEOM, MODEL_GEOM_SIZE)
    comptime L_BW = Layout.row_major(D.NBODY, 2)
    comptime L_CDOF = Layout.row_major(BATCH, D.NV * 6)
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_SOLVER = Layout.row_major(BATCH, SOLVER_WS)

    comptime L_QPOS = Layout.row_major(BATCH, D.NQ)
    comptime L_DW = Layout.row_major(D.NV)

    comptime if target == "cpu":
        var dm = d.dims
        var rl_QPOS = rl2(BATCH, dm.get_nq())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_CON = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
        var rl_SMETA = rl2(BATCH, METADATA_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_MMETA = rl1(MODEL_META_SIZE)
        var rl_EQ = rl2(dm.get_nequality(), MODEL_EQ_SIZE)
        var rl_TEN = rl2(dm.get_ntendon(), MODEL_TENDON_SIZE)
        var rl_SITE = rl2(dm.get_nsite(), MODEL_SITE_SIZE)
        var rl_GEOM_W = rl2(dm.get_ngeom(), MODEL_GEOM_SIZE)
        var rl_BW = rl2(dm.get_nbody(), 2)
        var rl_DW = rl1(dm.get_nv())
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        # ⚠⚠ RUNTIME BUDGET, NOT THE COMPTIME `SOLVER_WS`. On a dynamic
        # provider `D.NV` is DIM_POISON and `D.MAX_CONTACTS` floors to 1,
        # so the comptime literal is 81 - 12 = 69 scalars for EVERY model
        # while the ws_* offsets below are computed from the RUNTIME nv/mc.
        # The spelling was swept to `rl2`/`lt_dyn` in 3a; the VALUE was not.
        var rl_SOLVER = rl2(
            BATCH, ws_budget(_max_one_rt(dm.get_max_contacts()), dm.get_nv())
        )
        var qpos_v = d.qpos.lt_dyn["cpu", DYN2](rl_QPOS)
        var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
        var xpos_v = d.xpos.lt_dyn["cpu", DYN2](rl_B3)
        var xquat_v = d.xquat.lt_dyn["cpu", DYN2](rl_B4)
        var stcom_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_B3)
        var con_v = d.contacts.lt_dyn["cpu", DYN2](rl_CON)
        var smeta_v = d.meta.lt_dyn["cpu", DYN2](rl_SMETA)
        var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
        var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
        var mmeta_v = m.meta.lt_dyn["cpu", DYN1](rl_MMETA)
        var rl_TREES = rl1(dm.get_nv() * MODEL_TREE_SIZE)
        var trees_v = m.trees.lt_dyn["cpu", DYN1](rl_TREES)
        var eq_v = m.equality.lt_dyn["cpu", DYN2](rl_EQ)
        var ten_v = m.tendons.lt_dyn["cpu", DYN2](rl_TEN)
        var site_v = m.sites.lt_dyn["cpu", DYN2](rl_SITE)
        var geomw_v = m.geoms.lt_dyn["cpu", DYN2](rl_GEOM_W)
        var bw_v = m.body_invweight0.lt_dyn["cpu", DYN2](rl_BW)
        var dw_v = m.dof_invweight0.lt_dyn["cpu", DYN1](rl_DW)
        var cdof_v = scratch.cdof.lt_dyn["cpu", DYN2](rl_CDOF)
        var M_v = scratch.M.lt_dyn["cpu", DYN2](rl_M)
        var mi_v = scratch.m_inv.lt_dyn["cpu", DYN2](rl_M)
        var qc_v = scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV)
        var qw_v = d.qacc_warmstart.lt_dyn["cpu", DYN2](rl_NV)
        var sol_v = cscratch.solver.lt_dyn["cpu", DYN2](rl_SOLVER)
        for e in range(BATCH):
            _newton_solve_env[
                DTYPE,
                CONE_TYPE,
                BATCH,
                SOLVER_WS, MAX_CONDIM=MAX_CONDIM, NOSLIP_ITER=NOSLIP_ITER,
                TREE_AWARE=True](
                e, dm, qpos_v, qvel_v, xpos_v, xquat_v, stcom_v, con_v, smeta_v,
                joints_v, bodies_v, mmeta_v, trees_v, eq_v, ten_v, site_v, geomw_v, bw_v, dw_v,
                cdof_v, M_v, mi_v, qc_v, qw_v, sol_v,
            )
    else:
        # GPU. PYRAMIDAL (the production default cone) on NVIDIA uses the
        # one-env-per-block cooperative solver: the big Newton matrices live in
        # SHARED memory + the device workspace instead of a ~60KB per-thread
        # local frame, which fixes the humanoid-scale local-memory OOM. That
        # kernel's threadgroup memory exceeds Metal's 32 KB limit, so Metal —
        # and the ELLIPTIC cone on any device — keep the one-thread-per-env
        # kernel (which only OOMs on NVIDIA, where PYRAMIDAL never takes it).
        var used_blocked = False
        comptime if CONE_TYPE == ConeType.PYRAMIDAL:
            if has_nvidia_gpu_accelerator():
                solve_newton_blocked["gpu", DTYPE, CONE_TYPE=CONE_TYPE, BATCH=BATCH, MAX_CONDIM=MAX_CONDIM, NOSLIP_ITER=NOSLIP_ITER, JE_WS=JE_WS](d, m, scratch, cscratch, ctx)
                used_blocked = True
        if not used_blocked:
            var c = ctx.value()
            comptime BLOCKS = (BATCH + NS_TPB - 1) // NS_TPB
            c.enqueue_function[
                _newton_solve_fields_kernel[
                    DTYPE,
                    D.NQ,
                    D.NV,
                    D.NBODY,
                    D.NJOINT,
                    D.MAX_CONTACTS,
                    D.NGEOM,
                    D.NEQUALITY,
                    D.NTENDON,
                    D.NSITE,
                    CONE_TYPE,
                    BATCH,
                    SOLVER_WS,
                    MAX_CONDIM,
                    NOSLIP_ITER,
                ]
            ](
                d.qpos.lt["gpu", L_QPOS](),
                d.qvel.lt["gpu", L_NV](),
                d.xpos.lt["gpu", L_B3](),
                d.xquat.lt["gpu", L_B4](),
                d.subtree_com.lt["gpu", L_B3](),
                d.contacts.lt["gpu", L_CON](),
                d.meta.lt["gpu", L_SMETA](),
                m.joints.lt["gpu", L_JOINT](),
                m.bodies.lt["gpu", L_BODY](),
                m.meta.lt["gpu", L_MMETA](),
                m.trees.lt["gpu", L_TREES](),
                m.equality.lt["gpu", L_EQ](),
                m.tendons.lt["gpu", L_TEN](),
                m.sites.lt["gpu", L_SITE](),
                m.geoms.lt["gpu", L_GEOM_W](),
                m.body_invweight0.lt["gpu", L_BW](),
                m.dof_invweight0.lt["gpu", L_DW](),
                scratch.cdof.lt["gpu", L_CDOF](),
                scratch.M.lt["gpu", L_M](),
                scratch.m_inv.lt["gpu", L_M](),
                scratch.qacc_constrained.lt["gpu", L_NV](),
                d.qacc_warmstart.lt["gpu", L_NV](),
                cscratch.solver.lt["gpu", L_SOLVER](),
                grid_dim=(BLOCKS,),
                block_dim=(NS_TPB,),
            )


# =============================================================================
# PYRAMIDAL blocked Newton solve — ONE ENV PER BLOCK, cooperative across
# MAX_CONTACTS threads (fields port of NewtonSolver.solve_gpu_blocked,
# newton_solver.mojo:2748). The big Newton matrices live in SHARED memory + the
# device `solver` workspace instead of a per-thread local frame, so the
# per-thread local reservation stays tiny — this is what avoids the humanoid-
# scale OOM the one-thread-per-env kernel hits on NVIDIA. Arithmetic, iteration
# order, constants and cooperative thread distribution are VERBATIM from the
# legacy; only slab addressing → Data/Model/scratch tensors changes.
# SOLVE_COOP_NEWTON / SOLVE_COOP_RECOMPUTE are both True in the legacy production
# default, so only those cooperative code paths are ported (the tid-0 serial
# "oracle" branches are dead in production and dropped).
# =============================================================================


def _newton_blocked_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
    # Per-env spill size for `Je`; 0 = it fits threadgroup memory. Comes
    # from `je_budget.je_ws_size` via the integrator — never computed here.
    JE_WS: Int = 0,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
    equality: LayoutTensor[
        DTYPE, Layout.row_major(NEQUALITY, MODEL_EQ_SIZE), MutAnyOrigin
    ],
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    geoms_w: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, Layout.row_major(NV), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    qacc_warmstart: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
    # Spill buffer for `Je` — [BATCH, 1] and untouched unless JE_WS > 0.
    je_ws: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, JE_WS if JE_WS > 0 else 1),
        MutAnyOrigin,
    ],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var contact_tid = tid
    var valid_env = env < BATCH

    comptime MC = _max_one[MAX_CONTACTS]()
    comptime V_SIZE = _max_one[NV]()
    comptime M_SIZE = _max_one[NV * NV]()
    # ⚠ FROM `je_budget.newton_block_threads`, which the LAUNCH also reads —
    # the stride and `block_dim` must not drift apart.
    comptime THREADS = newton_block_threads[MAX_CONTACTS]()
    # The cooperative STRIDE. Equals `THREADS` in production; see
    # `NEWTON_COOP_DIV`. `block_dim` is `THREADS` either way, so every
    # thread still reaches every `barrier()`.
    comptime COOP = _max_one[THREADS // NEWTON_COOP_DIV]()

    # Common normal block offsets (row-relative; the legacy `solver_ws_idx`
    # base is 0 in the fields solver tensor)
    comptime ws_J_n_idx = 15 * MC

    # Edge-list base, from `solver/elliptic_layout` — the same base both cones
    # start their Jacobian region at (that module owns the arithmetic; only
    # what follows it differs by cone).
    #
    # ⚠ THIS KERNEL IS PYRAMIDAL ONLY. `solve_newton` reaches it exclusively
    # under `comptime if CONE_TYPE == ConeType.PYRAMIDAL` — Metal cannot fit
    # its threadgroup memory and the elliptic cone has no cooperative port —
    # so the elliptic scalar slots are not zeroed here at all. They were
    # before, and it was dead work: the producer's elliptic branch is not
    # reached on this path either.
    comptime NE_ZERO = 2 * (MAX_CONDIM - 1)
    comptime ws_Jt_idx = ell_jt[MC, NV]()
    comptime pyr_sc = ws_Jt_idx + NE_ZERO * MC * NV

    # === PARALLEL: Initialize common normal workspace (one thread/contact) ===
    if valid_env:
        comptime if NEWTON_SERIAL_PROBE == 5:
            # Pure stores of constants; the real pass below rewrites every one.
            for _r in range(SERIAL_PROBE_REPEAT - 1):
                if contact_tid < MC:
                    _init_common_normal_ws[
                        DTYPE](env, contact_tid, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), solver)
                    var p_ck: Scalar[DTYPE] = 0
                    for e in range(NE_ZERO):
                        for d in range(NV):
                            var a = ws_Jt_idx + e * MC * NV + contact_tid * NV + d
                            solver[env, a] = 0
                            p_ck += rebind[Scalar[DTYPE]](solver[env, a])
                    # ⚠ NOT `ctrl_sh` — the shared arrays are not declared
                    # until ~250 lines below, so this site consumes into a
                    # `solver` slot the real pass rewrites regardless. The
                    # branch is unreachable; it only has to be UNPROVABLY so.
                    if p_ck == _probe_sentinel[DTYPE]():
                        solver[env, ws_Jt_idx] = Scalar[DTYPE](0)
        # ⚠⚠ THE GUARD THAT WAS DELETED AS VACUOUS, PUT BACK. `:755` records it
        # being dropped because "the legacy `contact_tid < MC` guard is vacuous
        # with block_dim.y = MC" — true only while `THREADS == MAX_CONTACTS`.
        # `_init_common_normal_ws` has NO internal bound: it writes
        # `solver[env, k*MC + contact_tid]` for k in 0..14 and
        # `[15*MC + contact_tid*nv + i]`, so a thread with `contact_tid >= MC`
        # writes OUTSIDE its slot region by construction, and this layout's one
        # failure mode (see `:718`) is that a row overrun lands in the NEXT
        # ENV's workspace rather than faulting.
        #
        # ⚠ HONEST LIMIT OF THE EVIDENCE: it is restored because it is correct
        # and free, NOT because a test proved it necessary. **The negative
        # control has now failed to fire on FOUR models.** Removing it left
        # every result BIT-IDENTICAL at `THREADS = 2*MC` and `8*MC` on
        # Walker2d and ThreeTrees, and again at `4*MC` on SO101Tabletop —
        # which was chosen precisely to break it: 4 trees, 64 contacts against
        # 30 collision MESHES, and `MAX_CONTACTS = 32` where ThreeTrees is 8,
        # so `ME` moves on both axes at once. At every contact count reachable
        # here the overrun lands in slots the producer rewrites or nothing
        # reads.
        #
        # So a future reader must NOT take this guard's presence as evidence
        # that it fires, and must not delete it on the grounds that a test
        # passes without it — that is exactly the reasoning at `:755` that
        # removed it the first time. What IS verified: the kernel is
        # bit-identical at 1x, 2x, 4x and 8x `MAX_CONTACTS` threads WITH the
        # guards in place, which is the property F3 step 2 depends on.
        if contact_tid < MC:
            _init_common_normal_ws[
                DTYPE](env, contact_tid, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), solver)
            # ⚠ ALL `2*(dim-1)` EDGE BLOCKS, not the four this used to zero.
            # The producer re-zeros every edge of a non-penetrating contact
            # itself, which is the only reason the short version was survivable.
            for e in range(NE_ZERO):
                for d in range(NV):
                    solver[
                        env, ws_Jt_idx + e * MC * NV + contact_tid * NV + d
                    ] = 0

    comptime if NEWTON_STOP_AFTER == 1:
        return

    # === Read metadata (all threads; legacy `dt` read dropped — unused) ===
    var nc = 0
    var K_spring: Scalar[DTYPE] = 0
    var B_damp: Scalar[DTYPE] = 0
    var si_dmin: Scalar[DTYPE] = 0
    var si_dmax: Scalar[DTYPE] = 0
    var si_width: Scalar[DTYPE] = 1
    var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
    var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)
    var impratio: Scalar[DTYPE] = Scalar[DTYPE](1.0)

    if valid_env:
        nc = Int(rebind[Scalar[DTYPE]](smeta[env, META_IDX_NUM_CONTACTS]))
        if nc > MAX_CONTACTS:
            nc = MAX_CONTACTS
        var sr_tc = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_CONTACT_0]
        )
        var sr_dr = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_CONTACT_1]
        )
        si_dmin = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_0])
        si_dmax = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_1])
        si_width = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_2])
        si_midpoint = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_CONTACT_3]
        )
        si_power = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_4])
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        # MuJoCo clamps BOTH ends of solimp to [mjMINIMP, mjMAXIMP] before
        # interpolating (engine_core_constraint.c:1284-1287). The dmin floor is
        # the one that bites: R = (1-imp)/imp * diagApprox, so dmin=0 asks for an
        # infinitely soft contact at first touch. dm_control's finger is the first
        # model here to set it (`solimp="0 0.9 0.01"`); everything before used the
        # 0.9 default, which is why clamping only dmax survived.
        comptime MJ_MINIMP = Scalar[DTYPE](0.0001)
        comptime MJ_MAXIMP = Scalar[DTYPE](0.9999)
        if si_dmin < MJ_MINIMP:
            si_dmin = MJ_MINIMP
        elif si_dmin > MJ_MAXIMP:
            si_dmin = MJ_MAXIMP
        if si_dmax < MJ_MINIMP:
            si_dmax = MJ_MINIMP
        elif si_dmax > MJ_MAXIMP:
            si_dmax = MJ_MAXIMP
        if si_power < Scalar[DTYPE](1):
            si_power = Scalar[DTYPE](1)
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        (K_spring, B_damp) = solref_spring_damper[DTYPE](
            sr_tc, sr_dr, si_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        impratio = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_IMPRATIO])
        if impratio < Scalar[DTYPE](1e-6):
            impratio = Scalar[DTYPE](1.0)

    # === PARALLEL PHASE 1: each thread precomputes one contact's normal data ==
    # ⚠ THE `< nc` IS SEMANTICALLY A NO-OP AND KEPT ANYWAY, because it is one
    # less thing to reason about at a different launch shape. The helper carries
    # its own `contact_tid < nc` and would be correct unguarded at any thread
    # count — but it allocates a `Scratch[V_CAP]` and zeroes `nv` entries BEFORE
    # that test, which every surplus thread would pay for nothing. `nc <= MC <=
    # THREADS`, so this can never hide a slot.
    if valid_env and contact_tid < nc:
        comptime if NEWTON_SERIAL_PROBE == 6:
            # Both are pure functions of contacts/qvel/cdof/m_inv into the
            # solver workspace; the real passes below rewrite the same slots.
            for _r in range(SERIAL_PROBE_REPEAT - 1):
                _precompute_contact_normal[
                    DTYPE, V_SIZE](
                    env, contact_tid, nc, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qvel, subtree_com, contacts, joints, bodies,
                    mmeta, body_invweight0, cdof, m_inv, qacc_constrained, solver,
                    K_spring, B_damp, si_dmin, si_dmax, si_width, si_midpoint,
                    si_power,
                )
                _precompute_contact_friction[
                    DTYPE, V_SIZE, CONE_TYPE=CONE_TYPE, MAX_CONDIM=MAX_CONDIM](
                    env, contact_tid, nc, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qvel, subtree_com, contacts, joints, bodies,
                    mmeta, cdof, solver, B_damp, impratio, K_spring,
                )
        _precompute_contact_normal[
            DTYPE, V_SIZE](
            env, contact_tid, nc, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qvel, subtree_com, contacts, joints, bodies,
            mmeta, body_invweight0, cdof, m_inv, qacc_constrained, solver,
            K_spring, B_damp, si_dmin, si_dmax, si_width, si_midpoint,
            si_power,
        )

    barrier()

    # === PARALLEL PHASE 2: tangent frame + friction data ===
    if valid_env and contact_tid < nc:
        _precompute_contact_friction[
            DTYPE, V_SIZE, CONE_TYPE=CONE_TYPE, MAX_CONDIM=MAX_CONDIM](
            env, contact_tid, nc, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qvel, subtree_com, contacts, joints, bodies,
            mmeta, cdof, solver, B_damp, impratio, K_spring,
        )

    barrier()

    comptime NEWTON_ITER_GPU: Int = 1000
    # ⚠⚠ THE TOLERANCE IS DTYPE-AWARE, AND AT FLOAT32 IT HAS TO BE. Both exit
    # tests — `scale * ||grad||` and `scale * improvement` — are differences of
    # same-magnitude terms, so at float32 their rounding floor sits ORDERS OF
    # MAGNITUDE above 1e-8. Neither test can ever fire, and the solver runs its
    # full `NEWTON_ITER_GPU` budget on every step that has a single constraint
    # row. Measured on SO-ARM100 (one shallow contact, 6 DOF): 1.04 ms/env step
    # against 0.55 ms once the threshold clears the noise — HALF the step spent
    # iterating on rounding error. MuJoCo uses 1e-8 and is float64 throughout,
    # so the deviation is ours to make, not theirs to match.
    #
    # ⚠ THE EXTRA ITERATIONS BUY NOTHING, WHICH IS THE POINT. Measured on a
    # settling sphere: 1e-6 moves the resting penetration by 1.5e-8, while
    # float32's own distance from float64 is 9.8e-9 to 1e-6 depending on the
    # model — i.e. the correction is at or below the dtype's own error. Loosen
    # it much further and that stops being true: at 1e-1 the depth moves 2.7e-6.
    #
    # ⚠ NO FLOAT64 BEHAVIOUR CHANGES — the float64 branch is the literal old
    # constant, so every MuJoCo-parity gate in the tree (all of which run at
    # float64) is bit-identical across this change. That also means NONE of
    # them covers the float32 branch; `test_newton_float32_tracks_float64.mojo`
    # exists for that and is the only float32 convergence gate there is.
    comptime NEWTON_TOL_GPU: Float64 = (
        1e-8 if DTYPE == DType.float64 else 1e-6
    )
    # ⚠⚠ MuJoCo'S LINESEARCH BUDGET IS 50, NOT 20 (`m->opt.ls_iterations`) —
    # and it is the DEFAULT of a model field, not a constant. A ceiling here;
    # `lsiter_rt` below is the count. apollo asks for 10, so101 for 20.
    comptime LINESEARCH_ITER: Int = 50
    # ⚠⚠ AND ITS LINESEARCH TOLERANCE IS `opt.tolerance * opt.ls_tolerance`,
    # NOT `opt.tolerance` alone. `mj_solPrimal` calls
    #     PrimalSearch(&ctx, m->opt.tolerance * m->opt.ls_tolerance, ...)
    # and `PrimalSearch` forms `gtol = tolerance * snorm / scale` from THAT
    # product. `ls_tolerance` defaults to 0.01, so the real threshold is 1e-10
    # and ours was 1e-8 — a HUNDRED times looser, which makes the search accept
    # its first 1-D Newton point instead of refining toward the minimum along
    # the search direction.
    #
    # ⚠ THE SYMPTOM IS A SMALLER STEP, NOT A LOOSER ONE. Measured per iteration
    # on `reassemble_3` at float64 before this: the outer Newton converges
    # QUADRATICALLY while alpha stays ~1 (grad 1.19e-02 -> 8.49e-06 -> 4.25e-08
    # across iterations 5-7, alpha 1.0002 then 1.0029), and then alpha
    # collapses — 0.108, 0.086, 0.072, ... 0.0060 — and the gradient creeps
    # ~0.6% per iteration for another 77 iterations to reach 1e-8. The Hessian
    # is right (quadratic convergence proves it) and the active set stops
    # flipping at iteration 4, so the accepted alpha is what is wrong.
    comptime LS_TOLERANCE: Float64 = 0.01
    comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

    # ── the model's SOLVER BUDGET, from meta ────────────────────────────────
    #
    # ⚠⚠ THESE WERE THE COMPTIME CONSTANTS BELOW AND THE MODEL WAS IGNORED.
    # `apptronik_apollo` ships `<option iterations="4" ls_iterations="10">` and
    # we ran it to convergence — a DIFFERENT answer, not a better one, because
    # MuJoCo's answer for that model is its 4-iteration iterate. Five
    # Menagerie models set `iterations`, four set `ls_iterations` and rby1's
    # five scenes set `tolerance="1e-6"`.
    #
    # ⚠ THE COMPTIME CONSTANTS SURVIVE AS CAPS, not as the budget. A `range()`
    # needs a bound the compiler can see for the GPU path, so the loops still
    # run to `NEWTON_ITER_GPU` / `LINESEARCH_ITER` and break at the model's
    # count; a model asking for MORE than the cap is truncated at it.
    var niter_rt = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLVER_ITERATIONS])
    )
    if niter_rt <= 0:
        niter_rt = NEWTON_ITER_GPU
    var lsiter_rt = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_LS_ITERATIONS])
    )
    if lsiter_rt <= 0:
        lsiter_rt = LINESEARCH_ITER
    # ⚠ THE FLOOR IS THE DTYPE'S, NOT THE MODEL'S, AND IT ONLY RAISES.
    # `NEWTON_TOL_GPU` is 1e-8 at float64 (MuJoCo's own default, so this is a
    # no-op there) and 1e-6 at float32, where both exit tests are differences
    # of same-magnitude terms whose rounding floor sits above 1e-8 — a model
    # asking for 1e-8 at float32 would never converge and would burn its whole
    # budget on rounding error. A model asking for something LOOSER keeps it.
    var tol_rt = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLVER_TOLERANCE]
    )
    comptime if DTYPE != DType.float64:
        if tol_rt < Scalar[DTYPE](NEWTON_TOL_GPU):
            tol_rt = Scalar[DTYPE](NEWTON_TOL_GPU)
    # ⚠ A MULTIPLIER, NOT A THRESHOLD — `mj_solPrimal` passes
    # `opt.tolerance * opt.ls_tolerance` to `PrimalSearch`.
    var lstol_rt = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_LS_TOLERANCE]
    )
    if lstol_rt <= Scalar[DTYPE](0):
        lstol_rt = Scalar[DTYPE](LS_TOLERANCE)


    # PYRAMIDAL-only blocked solver. (Non-PYRAMIDAL never routes here.)
    # 2*(dim-1) edges per contact; see the per-env path for the layout note.
    comptime NE = 2 * (MAX_CONDIM - 1)
    comptime MAX_LIM = _max_one[2 * NJOINT]()
    comptime MAX_FRIC = V_SIZE  # one dry-friction row per dof
    comptime MAX_TLIM = 2 * NTENDON  # lo + hi per tendon
    # contact edges + joint limits + dry friction + tendon limits.
    #
    # ⚠ The last two were MISSING here until 2026-07-31, so on NVIDIA +
    # PYRAMIDAL — the only configuration that takes this kernel — a model with
    # `frictionloss` had NO dry friction and a model with a limited tendon had
    # NO string, both silently. `frictionloss` rows landed in the per-env
    # pyramidal path with 04a7c508 and were simply never mirrored here.
    #
    # ⚠ ME drives `Je_sh`, which is `ME * V_SIZE` DOUBLES of THREADGROUP
    # memory and is the dominant shared-memory term. Growing ME by
    # `V_SIZE + 3*NTENDON` grows Je_sh by `(V_SIZE + 3*NTENDON) * V_SIZE`. On a
    # large model that can push the block over the device's shared-memory
    # limit, which shows up as a LAUNCH FAILURE (loud), not a wrong answer.
    comptime MAX_TEQ = NTENDON  # one bilateral row per equality tendon
    # connect is 3 rows, weld is 6; sized for the worst case per equality.
    comptime MAX_WELD = 6 * NEQUALITY
    comptime ME = (
        NE * MC + MAX_LIM + MAX_FRIC + MAX_TLIM + MAX_TEQ + MAX_WELD
    )

    # ── Je: shared when it fits, spilled to global when it does not ───────
    #
    # ⚠⚠ MEASURED FAILURE THIS GUARD EXISTS FOR (humanoid_CMU, 2026-08-10):
    #     ptxas error : Entry function ... uses too much shared data
    #                   (0x2975c bytes, 0x18c00 max)
    # i.e. 169,820 B requested against a 101,376 B limit. `Je_sh` is
    # `ME * V_SIZE` scalars and dominates everything else combined:
    #
    #     humanoid      NV=27  MC=32  ME~150   Je ~16 KB   -> fits
    #     humanoid_CMU  NV=62  MC=64  ME=432   Je ~107 KB  -> over the limit
    #                                                        BY ITSELF
    #
    # ⚠ HALVING max_contacts DOES NOT FIX IT — 64->32 gives ME=304, Je 75 KB,
    # total ~116 KB, still over. It would cost real contact fidelity and still
    # not compile, so do not reach for that lever.
    #
    # Spilling ONLY Je leaves ~61 KB in threadgroup memory, which fits with
    # room to spare. The spill is GATED because Je is read across up to
    # NEWTON_ITER_GPU (200) iterations: putting it in global memory costs
    # bandwidth on EVERY model taking this kernel, and only the oversized ones
    # need to pay. Models that fit keep the fast path unchanged, bit for bit.
    #
    # ⚠ THE THRESHOLD IS A COMPILE-TIME GUESS AT A RUNTIME LIMIT. Shared
    # memory per block is device-specific (99 KB on this box; 227 KB on an
    # H100), and the kernel is compiled without knowing the target. 64 KB is
    # deliberately conservative — the widely-supported opt-in floor — so a
    # model that fits everywhere keeps shared, and anything near the edge
    # spills rather than failing to compile on the smallest plausible device.
    comptime _JE_ELEMS = ME * V_SIZE

    # ⚠ WHERE A SPILLED Je LIVES: `cscratch.je`, a DEDICATED per-env buffer
    # sized by `je_budget.je_ws_size` — the same function the integrator used
    # to allocate it, so the size the kernel indexes and the size that was
    # allocated cannot drift.
    #
    # An earlier version carved this out of the unused TAIL of the solver
    # workspace instead. That worked for humanoid_CMU (26,784 needed vs 27,264
    # free — 1.8% headroom) but not for dog (38,789 vs 12,672), which forced a
    # third "fits neither" case that silently fell back to shared and left dog
    # uncompilable on NVIDIA. A dedicated buffer is always exactly big enough,
    # so the gate is two-way again: shared when it fits, spill when it does not.
    #
    # ⚠ AND IT DOES NOT TOUCH `SOLVER_WS`. That literal is the row stride of a
    # `[BATCH, SOLVER_WS]` view recomputed in FIVE solver files; growing the
    # tensor without growing every view would make every row after 0 read the
    # wrong memory — silent corruption, not a crash.
    # ⚠ FLOORED AT 1 to match `ContactScratch.JE_ELEMS` and the operand's
    # declared layout — a zero-extent tensor operand segfaults.
    comptime JE_ELEMS = JE_WS if JE_WS > 0 else 1
    comptime JE_IN_SHARED = not je_spills[
        DTYPE, NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
    ]()
    comptime JE_AS = (
        AddressSpace.SHARED if JE_IN_SHARED else AddressSpace.GENERIC
    )

    # Sized to 1 when spilling so the threadgroup allocation disappears.
    comptime JE_SH_ELEMS = _JE_ELEMS if JE_IN_SHARED else 1

    comptime if NEWTON_STOP_AFTER == 2:
        return

    # === SHARED memory (per-block == per-env) ===
    var M_sh = LayoutTensor[
        DTYPE, Layout.row_major(M_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var H_sh = LayoutTensor[
        DTYPE, Layout.row_major(M_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # ⚠ BOTH BRANCHES ARE TYPE-CHECKED even though only one is emitted
    # (measured: an ill-typed untaken `comptime if` branch fails the build).
    # `address_space_cast[JE_AS]()` on each side is what makes them agree —
    # without it the SHARED build rejects the GENERIC branch and vice versa.
    var _je_backing = LayoutTensor[
        DTYPE, Layout.row_major(JE_SH_ELEMS), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var _je_ptr: Pointer[
        Scalar[DTYPE], MutAnyOrigin, address_space=JE_AS
    ]
    comptime if JE_IN_SHARED:
        _je_ptr = _je_backing.ptr.unsafe_address_space_cast[JE_AS]()
    else:
        _je_ptr = (
            je_ws.ptr.unsafe_offset(env * JE_ELEMS)
        ).unsafe_address_space_cast[JE_AS]()
    var Je_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME * V_SIZE), MutAnyOrigin,
        address_space=JE_AS,
    ](_je_ptr)
    var De_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var bias_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var force_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Row kind + box data (written once by thread 0) and the per-iteration row
    # STATE (written by thread 0 with the forces, read by every thread for the
    # Hessian). The state cannot be re-derived from `force_sh` alone: a
    # saturated box row has force > 0 yet contributes NO curvature, which is
    # exactly the misclassification `primal.mojo` carried until 04a7c508.
    var kind_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var R_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var floss_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var state_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var L_sh = LayoutTensor[
        DTYPE, Layout.row_major(M_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # ⚠ `H`'s DIAGONAL BLOCKS, per dof: the half-open range of the segment
    # containing dof i. Shared because the whole threadgroup factors against
    # them. `2 * V_SIZE` scalars — 480 B at nv=60, against the 86,676 B this
    # kernel already asks for at k=9. ⚠ IT IS SHARED FOOTPRINT `je_spills`
    # DOES NOT COUNT, exactly like the three M_SIZE arrays; see P4.
    var seg0_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var seg1_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # ⚠ SHARED, NOT PER-THREAD, BECAUSE THE SOLVE IS PER-BLOCK. `grad` was a
    # tid-0 `Scratch` whose only readers were the norm and the Cholesky solve;
    # once each thread solves a different diagonal block it needs the right-hand
    # side, so the vector moves to threadgroup memory. Counted in
    # `newton_shared_elems` — one term, `1 * NV`.
    var grad_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var search_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var Mv_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var Jv_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var qacc_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var jar_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var qfrc_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Scalar shared state: [0]=num_edges, [1]=done flag, [2]=Cholesky
    # rank-deficient flag.
    var ctrl_sh = LayoutTensor[
        DTYPE, Layout.row_major(3), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # === COOPERATIVE LOAD: M into shared ===
    if valid_env:
        comptime if NEWTON_SERIAL_PROBE == 7:
            # `NV^2` scalars read from GLOBAL every solve — the one setup term
            # that grows quadratically in nv. Pure copy; the real pass rewrites
            # every slot.
            for _r in range(SERIAL_PROBE_REPEAT - 1):
                var p_ck: Scalar[DTYPE] = 0
                for k in range(tid, NV * NV, COOP):
                    var v = rebind[Scalar[DTYPE]](M[env, k])
                    M_sh[k] = v
                    p_ck += v
                if p_ck == _probe_sentinel[DTYPE]():
                    ctrl_sh[2] = Scalar[DTYPE](0)
        for k in range(tid, NV * NV, COOP):
            M_sh[k] = rebind[Scalar[DTYPE]](M[env, k])

        # Cooperative load of contact edges (Je/De/bias_e) into shared. One
        # thread per contact (contact_tid == c), matching serial load order
        # (c ascending, e ascending).
        if contact_tid < nc:
            var c = contact_tid
            for e in range(NE):
                var idx = c * NE + e
                for i in range(NV):
                    Je_sh[idx * NV + i] = rebind[Scalar[DTYPE]](
                        solver[env, ws_Jt_idx + e * MC * NV + c * NV + i]
                    )
                De_sh[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + e * MC + c]
                )
                bias_e_sh[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + NE * MC + e * MC + c]
                )

    barrier()

    comptime if NEWTON_STOP_AFTER == 3:
        return

    # === THREAD 0: joint-limit edge detection + initial setup ===
    var qacc = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var qacc_smooth = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var Ma = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var f_smooth = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var jar = Scratch[Scalar[DTYPE], ME](
        ME, uninitialized=Scalar[DTYPE](0)
    )
    var search = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var Mv = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var Jv_e = Scratch[Scalar[DTYPE], ME](
        ME, uninitialized=Scalar[DTYPE](0)
    )
    var qfrc = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var old_qacc = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var old_Ma = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var old_jar = Scratch[Scalar[DTYPE], ME](
        ME, uninitialized=Scalar[DTYPE](0)
    )
    var old_force = Scratch[Scalar[DTYPE], ME](
        ME, uninitialized=Scalar[DTYPE](0)
    )
    var old_qfrc = Scratch[Scalar[DTYPE], V_SIZE](
        NV, uninitialized=Scalar[DTYPE](0)
    )
    var old_cost: Scalar[DTYPE] = 0
    var scale: Scalar[DTYPE] = 0
    var num_edges = 0

    if valid_env and tid == 0:
        # Contact edges and joint limits are ONE-SIDED, so they leave
        # kind = SROW_LIMIT and R/floss = 0; only the dry-friction rows below
        # override. Must be cleared first — shared memory is uninitialised.
        for e in range(ME):
            kind_e_sh[e] = Scalar[DTYPE](SROW_LIMIT)
            R_e_sh[e] = Scalar[DTYPE](0)
            floss_e_sh[e] = Scalar[DTYPE](0)
            state_e_sh[e] = Scalar[DTYPE](0)
        num_edges = nc * NE

        # Model-level defaults for fallback
        var lr_tc_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_0]
        )
        var lr_dr_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_1]
        )
        var li_dmin_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_0]
        )
        var li_dmax_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_1]
        )
        var li_width_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_2]
        )
        var li_midpoint_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_3]
        )
        var li_power_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_4]
        )

        for j in range(NJOINT):
            var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            var rmin = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MIN])
            var rmax = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MAX])
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            var lr_tc = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_0]
            )
            var lr_dr = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_1]
            )
            if lr_tc <= Scalar[DTYPE](0):
                lr_tc = lr_tc_def
            if lr_dr <= Scalar[DTYPE](0):
                lr_dr = lr_dr_def
            var li_dmin = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_0]
            )
            var li_dmax = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_1]
            )
            var li_width = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_2]
            )
            var li_midpoint = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_3]
            )
            var li_power = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_4]
            )
            if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](0):
                li_dmin = li_dmin_def
                li_dmax = li_dmax_def
                li_width = li_width_def
                li_midpoint = li_midpoint_def
                li_power = li_power_def
            if li_width < Scalar[DTYPE](1e-6):
                li_width = Scalar[DTYPE](1e-6)
            # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
            # interpolating (engine_core_constraint.c:1284-1287); see the same fix
            # on the contact path above.
            comptime MJL_MINIMP = Scalar[DTYPE](0.0001)
            comptime MJL_MAXIMP = Scalar[DTYPE](0.9999)
            if li_dmin < MJL_MINIMP:
                li_dmin = MJL_MINIMP
            elif li_dmin > MJL_MAXIMP:
                li_dmin = MJL_MAXIMP
            if li_dmax < MJL_MINIMP:
                li_dmax = MJL_MINIMP
            elif li_dmax > MJL_MAXIMP:
                li_dmax = MJL_MAXIMP
            if li_power < Scalar[DTYPE](1):
                li_power = Scalar[DTYPE](1)
            # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
            # solref. See `constraints/constraint_data.solref_spring_damper` — the
            # formula lived in twelve copy-pasted sites until 2026-08-03.
            var (l_K_spring, l_B_damp) = solref_spring_damper[DTYPE](
                lr_tc, lr_dr, li_dmax,
                rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
            )

            var pos = rebind[Scalar[DTYPE]](qpos[env, qpos_adr])
            # Lower limit
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0) and num_edges < ME:
                var sign = Scalar[DTYPE](1)
# ⚠ NO `K = diag(M^-1)` HERE ANY MORE. MuJoCo's `mj_diagApprox`
                # (engine_core_constraint.c:1720) prices a joint-limit row with
                # `dof_invweight0`, a MODEL-TIME constant (:1880), and its
                # `efc_D` is `1 / R` outright (:2259). This row used to read the
                # per-step `M^-1` diagonal only to round-trip it —
                # `1/(1/(K+R)) - K` — which reproduces R to a few ulp and
                # nothing else, and was the last reason the integrator formed a
                # dense `M^-1` under Newton at all (`PERFORMANCE.md` §13: 24-46%
                # of every step past 20 dofs).
                var pen = -dist_lo
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                Je_sh[num_edges * NV + dof] = sign
                De_sh[num_edges] = Scalar[DTYPE](1) / R_lim
                bias_e_sh[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

            # Upper limit
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0) and num_edges < ME:
                var sign = Scalar[DTYPE](-1)
# ⚠ NO `K = diag(M^-1)` HERE ANY MORE. MuJoCo's `mj_diagApprox`
                # (engine_core_constraint.c:1720) prices a joint-limit row with
                # `dof_invweight0`, a MODEL-TIME constant (:1880), and its
                # `efc_D` is `1 / R` outright (:2259). This row used to read the
                # per-step `M^-1` diagonal only to round-trip it —
                # `1/(1/(K+R)) - K` — which reproduces R to a few ulp and
                # nothing else, and was the last reason the integrator formed a
                # dense `M^-1` under Newton at all (`PERFORMANCE.md` §13: 24-46%
                # of every step past 20 dofs).
                var pen = -dist_hi
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                Je_sh[num_edges * NV + dof] = sign
                De_sh[num_edges] = Scalar[DTYPE](1) / R_lim
                bias_e_sh[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

        # Tendon limit rows (mjCNSTR_LIMIT_TENDON). Dense J — the same builder
        # the per-env pyramidal path uses, so the two cones cannot drift.
        comptime if NTENDON > 0:
            # Staging buffers sized MAX_TLIM, NOT ME: these are per-thread
            # LOCAL memory, and `ME * V_SIZE` doubles would be tens of KB —
            # precisely the local-memory OOM this cooperative kernel exists to
            # avoid. The builder fills from index 0, so tendon capacity is all
            # it can ever need.
            var t_je = Scratch[Scalar[DTYPE], MAX_TLIM * V_SIZE](
                MAX_TLIM * NV, fill=Scalar[DTYPE](0)
            )
            var t_de = Scratch[Scalar[DTYPE], MAX_TLIM](
                MAX_TLIM, fill=Scalar[DTYPE](0)
            )
            var t_bias = Scratch[Scalar[DTYPE], MAX_TLIM](
                MAX_TLIM, fill=Scalar[DTYPE](0)
            )
            var t_n = 0
            build_tendon_limit_rows[
                DTYPE, V_SIZE, MAX_TLIM,
                BATCH](
                env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qvel, tendons, sites, geoms_w, bodies, joints, mmeta,
                subtree_com, cdof, xpos, xquat, m_inv,
                t_je, t_de, t_bias, MAX_TLIM, t_n,
            )
            for r in range(t_n):
                if num_edges >= ME:
                    break
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = t_je[r * NV + i]
                De_sh[num_edges] = t_de[r]
                bias_e_sh[num_edges] = t_bias[r]
                num_edges += 1

            # Tendon equality rows (fixed and spatial) — same staging, and the
            # same reason they are rows: see the CPU pyramidal path.
            var q_je = Scratch[Scalar[DTYPE], MAX_TEQ * V_SIZE](
                MAX_TEQ * NV, fill=Scalar[DTYPE](0)
            )
            var q_de = Scratch[Scalar[DTYPE], MAX_TEQ](
                MAX_TEQ, fill=Scalar[DTYPE](0)
            )
            var q_bias = Scratch[Scalar[DTYPE], MAX_TEQ](
                MAX_TEQ, fill=Scalar[DTYPE](0)
            )
            var q_kind = Scratch[Int, MAX_TEQ](MAX_TEQ, fill=SROW_EQ_BILATERAL)
            var q_n = 0
            build_tendon_equality_rows[
                DTYPE, V_SIZE, MAX_TEQ,
                BATCH](
                env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qpos, qvel, tendons, sites, geoms_w, bodies, joints, mmeta,
                subtree_com, cdof, xpos, xquat, m_inv,
                q_je, q_de, q_bias, q_kind, MAX_TEQ, q_n,
            )
            for r in range(q_n):
                if num_edges >= ME:
                    break
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = q_je[r * NV + i]
                De_sh[num_edges] = q_de[r]
                bias_e_sh[num_edges] = q_bias[r]
                kind_e_sh[num_edges] = Scalar[DTYPE](q_kind[r])
                num_edges += 1

        # connect / weld EQUALITY rows (defect 29a) — the same conversion the
        # per-env paths have, mirrored here 2026-08-12. Dense J, BILATERAL,
        # `De = 1/R` recovered from the builder's PGS step size (see the
        # per-env pyramidal path for why that distinction is load-bearing).
        #
        # ⚠ STAGED BY `WR`/`WJ` — THE ROWS BEING BUILT — NOT BY `ME`. These are
        # PER-THREAD local arrays, and sizing one by total edge capacity is
        # exactly the tens-of-KB local-memory blowout this cooperative kernel
        # exists to avoid; the tendon-limit rows made that mistake first.
        comptime if NEQUALITY > 0:
            comptime WR = 6 * cap[NEQUALITY]()
            comptime WJ = 6 * cap[NEQUALITY]() * cap[NV]()
            var w_K = Scratch[Scalar[DTYPE], WR](6 * NEQUALITY, Scalar[DTYPE](1))
            var w_bias = Scratch[Scalar[DTYPE], WR](6 * NEQUALITY, Scalar[DTYPE](0))
            var w_D = Scratch[Scalar[DTYPE], WR](6 * NEQUALITY, Scalar[DTYPE](0))
            var w_J = Scratch[Scalar[DTYPE], WJ](6 * NEQUALITY * NV, Scalar[DTYPE](0))
            var w_MinvJ = Scratch[Scalar[DTYPE], WJ](
                6 * NEQUALITY * NV, Scalar[DTYPE](0)
            )
            var n_w = build_weld_equality_rows[DTYPE, V_SIZE](
                env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qpos, qvel, xpos, xquat, subtree_com, joints, bodies,
                mmeta, equality, body_invweight0, dof_invweight0, cdof, m_inv,
                w_K, w_bias, w_D, w_J, w_MinvJ,
            )
            for r in range(n_w):
                if num_edges >= ME:
                    break
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = w_J[r * NV + i]
                var R_recov = Scalar[DTYPE](1) / w_D[r] - w_K[r]
                if R_recov < Scalar[DTYPE](1e-14):
                    R_recov = Scalar[DTYPE](1e-14)
                De_sh[num_edges] = Scalar[DTYPE](1) / R_recov
                bias_e_sh[num_edges] = w_bias[r]
                kind_e_sh[num_edges] = Scalar[DTYPE](SROW_EQ_BILATERAL)
                num_edges += 1

        # Dry-friction dof rows (mjCNSTR_FRICTION_DOF). BOX rows, clamped to
        # +-frictionloss, so they are the reason this kernel needs row states
        # at all. Arithmetic identical to the per-env pyramidal builder.
        var f_imp = Scalar[DTYPE](DOF_SOLIMP_DMIN)
        var f_dmax = Scalar[DTYPE](DOF_SOLIMP_DMAX)
        # REFSAFE applies to the hardcoded friction default too — see
        # `refsafe_timeconst`.
        var f_tc_p = refsafe_timeconst[DTYPE](
            Scalar[DTYPE](DOF_SOLREF_TIMECONST),
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        var f_B = Scalar[DTYPE](2.0) / (f_dmax * f_tc_p)
        for j in range(NJOINT):
            var floss = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_FRICTIONLOSS]
            )
            if floss <= Scalar[DTYPE](0):
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var nd = 1
            if jt == JNT_FREE:
                nd = 6
            elif jt == JNT_BALL:
                nd = 3
            for k in range(nd):
                if num_edges >= ME:
                    break
                var dof = dof_adr + k
                # `dof_invweight0`, as MuJoCo (engine_core_constraint.c:1876);
                # the `diag(M^-1)` fallback this carried was dead on any model
                # with a finite mass and is gone with the matrix.
                var diag_f = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                var R_f = (Scalar[DTYPE](1) - f_imp) / f_imp * diag_f
                if R_f < Scalar[DTYPE](1e-14):
                    R_f = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                Je_sh[num_edges * NV + dof] = Scalar[DTYPE](1)
                De_sh[num_edges] = Scalar[DTYPE](1) / R_f
                R_e_sh[num_edges] = R_f
                floss_e_sh[num_edges] = floss
                kind_e_sh[num_edges] = Scalar[DTYPE](SROW_FRICTION)
                bias_e_sh[num_edges] = f_B * rebind[Scalar[DTYPE]](
                    qvel[env, dof]
                )
                num_edges += 1

        # Publish num_edges to shared for all threads.
        ctrl_sh[0] = Scalar[DTYPE](num_edges)

        # ── H's diagonal blocks, from the rows just built ────────────────
        #
        # ⚠ HERE AND NOT INSIDE THE ITERATION LOOP. `Je` is final at this
        # point and does not change across iterations — only the row STATES
        # do — so one partition serves the whole solve and is a superset of
        # every iteration's coupling. Computing it per iteration would let the
        # partition move under the factorisation.
        _ = build_dof_segments[
            DTYPE, J_AS=JE_AS, S_AS = AddressSpace.SHARED
        ](
            NV,
            Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NTREE])),
            num_edges,
            trees,
            Je_sh,
            seg0_sh,
            seg1_sh,
        )

        # Initialize qacc/qacc_smooth from workspace
        for i in range(NV):
            var q_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
            qacc[i] = q_i
            qacc_smooth[i] = q_i
        # Ma = M * qacc (read from M_sh)
        # ⚠⚠ `Ma = M*qacc` OVER ROW i'S BLOCK. `M`'s off-tree entries are
        # STRUCTURALLY zero — both CRBA paths only ever write within a tree —
        # and a segment is a UNION of trees, so `[seg0[i], seg1[i])` is a
        # superset of row i's nonzeros and dropping the rest drops exact
        # zeros. Same argument as `_matvec_mv_jve_coop`.
        #
        # ⚠ THIS IS `NV^2` SERIAL ON THREAD 0 — 3,600 operations at nv=60,
        # against ~1,080 for the whole tid-0 half of one Newton ITERATION.
        # PN2d segmented every dense pass INSIDE the iteration loop and missed
        # both of the ones out here in the setup, which between them cost more
        # than several iterations. `range(tid, ..., THREADS)` made the others
        # easy to spot; a bare `for i in range(NV)` under `tid == 0` did not.
        comptime if NEWTON_SERIAL_PROBE == 8:
            # `Ma` is reset to 0 at the top of each row, so the accumulation is
            # idempotent and the real pass below recomputes it identically.
            for _r in range(SERIAL_PROBE_REPEAT - 1):
                for i in range(NV):
                    Ma[i] = Scalar[DTYPE](0)
                    var j0 = Int(rebind[Scalar[DTYPE]](seg0_sh[i]))
                    var j1 = Int(rebind[Scalar[DTYPE]](seg1_sh[i]))
                    if j1 <= j0:
                        j0 = 0
                        j1 = NV
                    for j in range(j0, j1):
                        Ma[i] += rebind[Scalar[DTYPE]](M_sh[i * NV + j]) * qacc[j]
                    f_smooth[i] = Ma[i]
        for i in range(NV):
            Ma[i] = Scalar[DTYPE](0)
            var j0 = Int(rebind[Scalar[DTYPE]](seg0_sh[i]))
            var j1 = Int(rebind[Scalar[DTYPE]](seg1_sh[i]))
            if j1 <= j0:
                j0 = 0
                j1 = NV
            for j in range(j0, j1):
                Ma[i] += rebind[Scalar[DTYPE]](M_sh[i * NV + j]) * qacc[j]
        for i in range(NV):
            f_smooth[i] = Ma[i]
        # Same model-constant scale as the per-env path above; see the note
        # there for why a pose-dependent trace(M) is wrong and why this is NOT
        # a fix for the open dog residual.
        var scale_db = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_MEANINERTIA]
        ) * Scalar[DTYPE](NV if NV > 1 else 1)
        scale = (
            Scalar[DTYPE](1) / scale_db
            if scale_db > Scalar[DTYPE](1e-10)
            else Scalar[DTYPE](1)
        )

        # Initial jar + force + qfrc; publish force to force_sh
        for i in range(NV):
            qfrc[i] = Scalar[DTYPE](0)
        for e_idx in range(num_edges):
            jar[e_idx] = rebind[Scalar[DTYPE]](bias_e_sh[e_idx])
            for i in range(NV):
                jar[e_idx] += (
                    rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i]) * qacc[i]
                )
            var st_e = scalar_row_state[DTYPE](
                Int(rebind[Scalar[DTYPE]](kind_e_sh[e_idx])),
                jar[e_idx],
                rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
            )
            state_e_sh[e_idx] = Scalar[DTYPE](st_e)
            var f_e = scalar_row_force[DTYPE](
                st_e,
                jar[e_idx],
                rebind[Scalar[DTYPE]](De_sh[e_idx]),
                rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
            )
            force_sh[e_idx] = f_e
            for i in range(NV):
                qfrc[i] += rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i]) * f_e

        # ── warmstart() — the per-env path's twin, see the comment there.
        # Serial on thread 0 like the rest of this init, and re-running the
        # init loop above rather than keeping a trial state is what keeps the
        # thread-0 frame the size that made this kernel exist.
        if (
            mmeta[MODEL_META_IDX_WARMSTART_DISABLED] == Scalar[DTYPE](0)
            and num_edges > 0
        ):
            var cost_s: Scalar[DTYPE] = 0
            for e_idx in range(num_edges):
                cost_s += scalar_row_cost[DTYPE](
                    Int(rebind[Scalar[DTYPE]](state_e_sh[e_idx])),
                    jar[e_idx],
                    rebind[Scalar[DTYPE]](De_sh[e_idx]),
                    rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                    rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                )
            var qacc_w = Scratch[Scalar[DTYPE], V_SIZE](
                NV, uninitialized=Scalar[DTYPE](0)
            )
            for i in range(NV):
                qacc_w[i] = rebind[Scalar[DTYPE]](qacc_warmstart[env, i])
            var cost_w: Scalar[DTYPE] = 0
            for e_idx in range(num_edges):
                var jar_w = rebind[Scalar[DTYPE]](bias_e_sh[e_idx])
                for i in range(NV):
                    jar_w += (
                        rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i])
                        * qacc_w[i]
                    )
                var st_w = scalar_row_state[DTYPE](
                    Int(rebind[Scalar[DTYPE]](kind_e_sh[e_idx])),
                    jar_w,
                    rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                    rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                )
                cost_w += scalar_row_cost[DTYPE](
                    st_w,
                    jar_w,
                    rebind[Scalar[DTYPE]](De_sh[e_idx]),
                    rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                    rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                )
            # The warmstart trial's `Ma = M*qacc_w` — the second NV^2 serial
            # matvec in this setup block, same block restriction and the same
            # exact-zero argument as the one above.
            for i in range(NV):
                var s_i: Scalar[DTYPE] = 0
                var w0 = Int(rebind[Scalar[DTYPE]](seg0_sh[i]))
                var w1 = Int(rebind[Scalar[DTYPE]](seg1_sh[i]))
                if w1 <= w0:
                    w0 = 0
                    w1 = NV
                for j in range(w0, w1):
                    s_i += (
                        rebind[Scalar[DTYPE]](M_sh[i * NV + j]) * qacc_w[j]
                    )
                Ma[i] = s_i
                cost_w += (
                    Scalar[DTYPE](0.5)
                    * (s_i - f_smooth[i])
                    * (qacc_w[i] - qacc_smooth[i])
                )
            if cost_w <= cost_s:
                for i in range(NV):
                    qacc[i] = qacc_w[i]
                    qfrc[i] = Scalar[DTYPE](0)
                for e_idx in range(num_edges):
                    jar[e_idx] = rebind[Scalar[DTYPE]](bias_e_sh[e_idx])
                    for i in range(NV):
                        jar[e_idx] += (
                            rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i])
                            * qacc[i]
                        )
                    var st_c = scalar_row_state[DTYPE](
                        Int(rebind[Scalar[DTYPE]](kind_e_sh[e_idx])),
                        jar[e_idx],
                        rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                        rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                    )
                    state_e_sh[e_idx] = Scalar[DTYPE](st_c)
                    var f_c = scalar_row_force[DTYPE](
                        st_c,
                        jar[e_idx],
                        rebind[Scalar[DTYPE]](De_sh[e_idx]),
                        rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                    )
                    force_sh[e_idx] = f_c
                    for i in range(NV):
                        qfrc[i] += (
                            rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i]) * f_c
                        )
            else:
                for i in range(NV):
                    Ma[i] = f_smooth[i]

    # Make num_edges + force_sh visible to all threads.
    barrier()
    comptime if NEWTON_STOP_AFTER == 4:
        return
    var num_edges_b = Int(rebind[Scalar[DTYPE]](ctrl_sh[0]))

    # === Newton iterations — ALL threads execute the loop ===
    var iters_done = 0
    var ls_evals = 0
    for iter_n in range(NEWTON_ITER_GPU):
        # ⚠⚠ NO CONSTRAINT ROWS: MUJOCO RETURNS, AND WE USED TO SOLVE.
        # `mj_fwdConstraint` (engine_forward.c:884) is explicit —
        #     if (!nefc) { mju_copy(d->qacc, d->qacc_smooth, nv); return; }
        # — and this loop had no such guard. With no rows the warmstart block
        # above is already skipped (its own `num_edges > 0` test), so `qacc` still
        # holds `qacc_smooth` and the gradient is IDENTICALLY ZERO. Every
        # iteration then factors `H = M` and solves for a search direction of
        # zero: the answer cannot move, so breaking here writes back exactly
        # what a full pass would have — this is a no-op, not an approximation.
        #
        # ⚠ IT IS NOT A MICRO-OPTIMISATION. P0 measured it at 32.3 of 46.9 ms
        # per step on the k=9 park scene — 69% of GPU time, 78% of the whole
        # parked-slot cost — spent re-factorising a matrix `ldl_factor` factored
        # two kernels earlier, for a problem with no constraints in it. The
        # cooperative Cholesky's `d_j` reduction is `if tid == 0: for k in
        # range(j)`, O(NV^2) on ONE thread, so it is also the fastest-growing
        # term in the sweep.
        # See docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md §0.0.1.
        #
        # ⚠ UNIFORM ACROSS THE THREADGROUP, so it cannot desynchronise the
        # `barrier()` calls in the loop body: `num_edges_b` is read from
        # `ctrl_sh[0]` AFTER a barrier, so every thread sees the same value and
        # every thread breaks on the same iteration.
        if num_edges_b == 0:
            break
        comptime if NEWTON_MIN_ITER == 0:
            if iter_n >= niter_rt:
                break
        else:
            if iter_n >= niter_rt and iter_n >= NEWTON_MIN_ITER:
                break
        # Counted AFTER every exit test, so it is the number of iterations
        # actually ENTERED, not the number the `range()` offered.
        iters_done = iter_n + 1
        # --- Thread 0: gradient + convergence check ---
        if valid_env and tid == 0:
            comptime if NEWTON_SERIAL_PROBE == 1:
                # Pure recompute: writes only `grad_sh`, which the real loop
                # below overwrites. The norm is deliberately NOT accumulated
                # here — it is the one part that is not idempotent.
                for _r in range(SERIAL_PROBE_REPEAT - 1):
                    var p_ck: Scalar[DTYPE] = 0
                    for i in range(NV):
                        var p_g = Ma[i] - f_smooth[i] - qfrc[i]
                        grad_sh[i] = p_g
                        p_ck += p_g
                    if p_ck == _probe_sentinel[DTYPE]():
                        ctrl_sh[2] = Scalar[DTYPE](0)
            var grad_norm: Scalar[DTYPE] = 0
            for i in range(NV):
                var g = Ma[i] - f_smooth[i] - qfrc[i]
                grad_sh[i] = g
                grad_norm += g * g
            # ⚠ NOT ON THE FIRST PASS — see the per-env twin. `mj_solPrimal`
            # tests `gradient` only after an update.
            comptime if NEWTON_MIN_ITER == 0:
                if iter_n > 0 and scale * sqrt(grad_norm) < tol_rt:
                    ctrl_sh[1] = Scalar[DTYPE](1)  # done
                else:
                    ctrl_sh[1] = Scalar[DTYPE](0)
            else:
                # `>= NEWTON_MIN_ITER` subsumes the original `> 0` guard.
                if (
                    iter_n >= NEWTON_MIN_ITER
                    and scale * sqrt(grad_norm) < tol_rt
                ):
                    ctrl_sh[1] = Scalar[DTYPE](1)  # done
                else:
                    ctrl_sh[1] = Scalar[DTYPE](0)
        barrier()
        if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 1:
            break

        # --- ALL threads: parallel Hessian assembly (inner edge-sum ascending
        # → bit-identical to the serial build) ---
        if valid_env:
            # ⚠⚠ ONLY THE DIAGONAL BLOCKS, AND THIS WAS THE LARGEST TERM LEFT
            # AFTER PN2c. The build ran over every one of `NV*NV` entries with
            # an inner sweep of the rows — `NV^2*(1+E)/THREADS` = 1,575 per
            # thread per iteration at nv=60 with six rows and THREADS=16 —
            # while the segmented factorisation reads only the blocks. Every
            # off-block write was dead. Audited: the only reads of `H_sh` are
            # `[j*nv+j]` and `[i*nv+j]` inside the factor's segment-restricted
            # loops, plus the rank-deficient retry's diagonal bump.
            #
            # ⚠ THE ENTRIES IT NO LONGER WRITES ARE NOW STALE, not zero. That
            # is safe only because nothing reads them; it is NOT the same
            # property as `L_sh` below, where zero is load-bearing.
            var bp = 0
            while bp < NV:
                var be = Int(rebind[Scalar[DTYPE]](seg1_sh[bp]))
                # A malformed partition would hang the walk; a runaway is worse
                # than a wrong answer.
                if be <= bp:
                    be = NV
                var bn = be - bp
                comptime if NEWTON_SERIAL_PROBE == 10:
                    for _r in range(SERIAL_PROBE_REPEAT - 1):
                        for q in range(tid, bn * bn, COOP):
                            var pi = bp + q // bn
                            var pj = bp + q % bn
                            var pidx = pi * NV + pj
                            var ph = rebind[Scalar[DTYPE]](M_sh[pidx])
                            for e in range(num_edges_b):
                                if (
                                    Int(rebind[Scalar[DTYPE]](state_e_sh[e]))
                                    == SROW_QUADRATIC
                                ):
                                    ph += (
                                        rebind[Scalar[DTYPE]](De_sh[e])
                                        * rebind[Scalar[DTYPE]](Je_sh[e * NV + pi])
                                        * rebind[Scalar[DTYPE]](Je_sh[e * NV + pj])
                                    )
                            H_sh[pidx] = ph
                for q in range(tid, bn * bn, COOP):
                    var i = bp + q // bn
                    var j = bp + q % bn
                    var idx = i * NV + j
                    var h = rebind[Scalar[DTYPE]](M_sh[idx])
                    for e in range(num_edges_b):
                        if (
                            Int(rebind[Scalar[DTYPE]](state_e_sh[e]))
                            == SROW_QUADRATIC
                        ):
                            h += (
                                rebind[Scalar[DTYPE]](De_sh[e])
                                * rebind[Scalar[DTYPE]](Je_sh[e * NV + i])
                                * rebind[Scalar[DTYPE]](Je_sh[e * NV + j])
                            )
                    H_sh[idx] = h
                bp = be
        barrier()

        # --- Cooperative Cholesky factor of H into L_sh ---
        _chol_factor_coop[DTYPE](
            tid, COOP, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), H_sh, L_sh, ctrl_sh, seg0_sh, seg1_sh
        )

        # --- ALL threads: one DIAGONAL BLOCK each, Cholesky solve ---
        #
        # ⚠⚠ THE PARALLELISM AXIS CHANGED WHEN THE MATRIX DID, and this is the
        # whole of F3b. `_chol_factor_coop` above parallelizes WITHIN a column
        # (thread per row `i`), which is right for one dense `NV*NV` system and
        # wrong for `1+k` independent 6-dof ones: the solve that consumed it ran
        # `sum(bn^2)` scalars SERIAL ON TID 0, every Newton iteration, and
        # measurement put that serial floor at 67% of the whole GPU excess at
        # k=13. The blocks are INDEPENDENT SYSTEMS — the property this entire
        # campaign established — so one block per THREAD costs `max(bn^2)`
        # instead of `sum(bn^2)`, i.e. it stops scaling with the slot count.
        #
        # ⚠ BIT-EXACT, and for a duller reason than the factorisation's: the
        # per-block solves were ALREADY separate calls over disjoint `[sp, se)`
        # ranges, run back to back. This changes WHICH thread runs each call
        # and nothing inside one. Each writes only `search_sh[sp:se]` and its
        # own `y`; each reads `L_sh` and `grad_sh`, which no thread writes
        # here. Disjoint ranges, so the concurrency is not a race.
        #
        # ⚠ THE WALK IS BY EVERY THREAD, THE SOLVE BY ONE. The segment list is
        # a linked walk (`seg1_sh[sp]` names the next boundary), so a thread
        # cannot jump to its b-th block — it walks all `1+k` boundaries and
        # acts on the ones that are its own. That is `1+k` shared reads per
        # thread against `bn^3` of solve, and it keeps ONE spelling of the walk.
        #
        # ⚠ `bidx % THREADS`, NOT `% COOP`. This is a partition of WORK across
        # real threads, not a strided sweep: at `COOP < THREADS` two threads
        # sharing a residue would solve the same block into the same slots.
        # Same values, but `chol_solve_seg_p` reads `x[j]` back during back
        # substitution, so a concurrent duplicate is a genuine race — unlike
        # every pass `NEWTON_COOP_DIV` touches. See its note at `:307`.
        if valid_env:
            comptime if NEWTON_SERIAL_PROBE == 2:
                # The solve WITHOUT the negate — `search_sh[i] = -search_sh[i]`
                # is the one statement here that is not idempotent, and the real
                # pass below rewrites `search_sh` from scratch anyway.
                for _r in range(SERIAL_PROBE_REPEAT - 1):
                    var psp = 0
                    var pbi = 0
                    while psp < NV:
                        var pse = Int(rebind[Scalar[DTYPE]](seg1_sh[psp]))
                        if pse <= psp:
                            pse = NV
                        if pbi % THREADS == tid:
                            chol_solve_seg_p[
                                DTYPE, V_SIZE,
                                L_AS = AddressSpace.SHARED,
                                B_AS = AddressSpace.SHARED,
                                X_AS = AddressSpace.SHARED,
                            ](L_sh.ptr, grad_sh.ptr, search_sh.ptr, NV, psp, pse)
                            # ⚠ THIS THREAD'S OWN BLOCK ONLY. Summing all of
                            # `search_sh` would read slots other threads are
                            # concurrently writing — a real data race, even for
                            # a value that is discarded.
                            var p_ck: Scalar[DTYPE] = 0
                            for i in range(psp, pse):
                                p_ck += rebind[Scalar[DTYPE]](search_sh[i])
                            if p_ck == _probe_sentinel[DTYPE]():
                                ctrl_sh[2] = Scalar[DTYPE](0)
                        psp = pse
                        pbi += 1
            var sp = 0
            var bidx = 0
            while sp < NV:
                var se = Int(rebind[Scalar[DTYPE]](seg1_sh[sp]))
                # A malformed partition would hang this loop; `seg1 > sp` is
                # guaranteed by `build_dof_segments` (a segment holds at least
                # one dof) but a runaway is worse than a wrong answer.
                if se <= sp:
                    se = NV
                if bidx % THREADS == tid:
                    chol_solve_seg_p[
                        DTYPE, V_SIZE,
                        L_AS = AddressSpace.SHARED,
                        B_AS = AddressSpace.SHARED,
                        X_AS = AddressSpace.SHARED,
                    ](L_sh.ptr, grad_sh.ptr, search_sh.ptr, NV, sp, se)
                    # Negate in place, over this block only. The publish is
                    # gone with the copy: the solve wrote `search_sh` directly.
                    for i in range(sp, se):
                        search_sh[i] = -rebind[Scalar[DTYPE]](search_sh[i])
                sp = se
                bidx += 1

        # --- Cooperative Mv = M·search and Jv_e = Je·search ---
        barrier()
        _matvec_mv_jve_coop[DTYPE, JE_AS=JE_AS](
            tid, COOP, num_edges_b, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), M_sh, Je_sh, search_sh, Mv_sh, Jv_e_sh, seg0_sh, seg1_sh
        )
        barrier()
        if valid_env and tid == 0:
            comptime if NEWTON_SERIAL_PROBE == 4:
                for _r in range(SERIAL_PROBE_REPEAT - 1):
                    var p_ck: Scalar[DTYPE] = 0
                    for i in range(NV):
                        Mv[i] = rebind[Scalar[DTYPE]](Mv_sh[i])
                        search[i] = rebind[Scalar[DTYPE]](search_sh[i])
                        p_ck += Mv[i] + search[i]
                    if p_ck == _probe_sentinel[DTYPE]():
                        ctrl_sh[2] = Scalar[DTYPE](0)
            for i in range(NV):
                Mv[i] = rebind[Scalar[DTYPE]](Mv_sh[i])
                # ⚠ THE READ-BACK RIDES THIS LOOP RATHER THAN ADDING A BARRIER.
                # `search` is now produced in shared memory by whichever thread
                # owned each block, and the tid-0 tail below still needs it as a
                # local (`gauss_a`, `snorm_sq`, the `qacc` update). This point is
                # already past the barrier that publishes `search_sh`, and the
                # cooperative matvec above consumed `search_sh` directly, so no
                # reader is skipped and no new synchronisation is needed.
                search[i] = rebind[Scalar[DTYPE]](search_sh[i])
            for e_idx in range(num_edges_b):
                Jv_e[e_idx] = rebind[Scalar[DTYPE]](Jv_e_sh[e_idx])

        # --- Thread 0: gauss / p0 / line search / update / cost ---
        if valid_env and tid == 0:
            var gauss_a: Scalar[DTYPE] = 0
            var gauss_b: Scalar[DTYPE] = 0
            for i in range(NV):
                gauss_a += Mv[i] * search[i]
                gauss_b += (Ma[i] - f_smooth[i]) * search[i]

            # ── `PrimalSearch` (engine_solver.c:1692), thread-0 serial ──
            #
            # ⚠⚠ THE SAME THREE PHASES AS `primal.mojo`'s
            # `pyramidal_linesearch`, AND IT HAS TO STAY THAT WAY. This kernel
            # cannot call that helper — its rows live in SHARED-memory
            # LayoutTensors, not `Scratch` — so the algorithm is written twice
            # and `test_noslip_blocked_kernel` / `test_newton_blocked_fields`
            # are what hold the two together. A rule written inline twice in
            # this tree has drifted before; read the other copy's docstring
            # for why one analytical step plus halving is not this.
            var snorm_sq: Scalar[DTYPE] = 0
            for i in range(NV):
                snorm_sq += search[i] * search[i]
            var snorm = sqrt(snorm_sq)
            var gtol_b = tol_rt * lstol_rt / scale * snorm

            var ls_budget = LINESEARCH_ITER
            if lsiter_rt > 0 and lsiter_rt < ls_budget:
                ls_budget = lsiter_rt

            # `PrimalEval` (engine_solver.c:1511): the SHIFTED line cost
            # `cost(a) - cost(0)` and BOTH derivatives in one pass, rows
            # RE-CLASSIFIED at the trial point. Mirrors `peval` in
            # `primal.mojo` — see that docstring for why the cost has to be
            # carried at every point rather than computed in the fallback.
            @parameter
            @always_inline
            def _bl_peval(
                a: Scalar[DTYPE],
                mut c: Scalar[DTYPE],
                mut d0: Scalar[DTYPE],
                mut d1: Scalar[DTYPE],
                mut it: Int,
            ):
                c = Scalar[DTYPE](0.5) * gauss_a * a * a + gauss_b * a
                d0 = gauss_a * a + gauss_b
                d1 = gauss_a
                for e_idx in range(num_edges_b):
                    var kd = Int(rebind[Scalar[DTYPE]](kind_e_sh[e_idx]))
                    var Rd = rebind[Scalar[DTYPE]](R_e_sh[e_idx])
                    var fd = rebind[Scalar[DTYPE]](floss_e_sh[e_idx])
                    var Dd = rebind[Scalar[DTYPE]](De_sh[e_idx])
                    var jt = jar[e_idx] + a * Jv_e[e_idx]
                    var st = scalar_row_state[DTYPE](kd, jt, Rd, fd)
                    # ⚠ THE alpha=0 REFERENCE IS RE-DERIVED FROM `jar`, not
                    # read from `state_e_sh`. `PrimalEval` evaluates the line
                    # at `Jaref + alpha*Jv` for alpha=0 like any other alpha,
                    # and the stored state is one Newton move stale.
                    var st0 = scalar_row_state[DTYPE](kd, jar[e_idx], Rd, fd)
                    c += scalar_row_cost[DTYPE](
                        st, jt, Dd, Rd, fd
                    ) - scalar_row_cost[DTYPE](st0, jar[e_idx], Dd, Rd, fd)
                    d0 += (
                        -scalar_row_force[DTYPE](st, jt, Dd, fd) * Jv_e[e_idx]
                    )
                    if st == SROW_QUADRATIC:
                        d1 += Dd * Jv_e[e_idx] * Jv_e[e_idx]
                if d1 <= Scalar[DTYPE](0):
                    d1 = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                it += 1

            # ⚠⚠ p0 GOES THROUGH `_bl_peval` LIKE EVERY OTHER POINT.
            # This block used to read the STORED `state_e_sh` and derive its
            # force from that, where the per-env twin RE-CLASSIFIES from `jar`
            # — and so does `PrimalEval`, which evaluates the line at
            # `Jaref + alpha*Jv` for alpha=0 like any other alpha. Two
            # spellings of the same point is exactly the asymmetry that makes
            # one leg start its search from a different derivative than the
            # other; blocked-GPU vs per-env-CPU `qacc` with the noslip pass OFF
            # went 1.06e-05 -> 6.26e-04 while they disagreed about it.
            var lsiter_b = 0
            var p0_c = Scalar[DTYPE](0)
            var p0_d0 = Scalar[DTYPE](0)
            var p0_d1 = Scalar[DTYPE](0)
            comptime if NEWTON_SERIAL_PROBE == 9:
                # `_bl_peval` is PURE — it writes only its `mut` outputs — so
                # repeating it is idempotent and the real call below overwrites
                # everything it touched. `lsiter_b` is a throwaway counter here.
                var q_c = Scalar[DTYPE](0)
                var q_d0 = Scalar[DTYPE](0)
                var q_d1 = Scalar[DTYPE](0)
                var q_it = 0
                for _r in range(SERIAL_PROBE_REPEAT - 1):
                    _bl_peval(Scalar[DTYPE](0), q_c, q_d0, q_d1, q_it)
                if q_c == _probe_sentinel[DTYPE]():
                    ctrl_sh[2] = Scalar[DTYPE](0)
            _bl_peval(Scalar[DTYPE](0), p0_c, p0_d0, p0_d1, lsiter_b)

            var alpha: Scalar[DTYPE] = 0
            if snorm >= Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                # Phase 1: always attempt one Newton step on the line.
                var p1_a = -p0_d0 / p0_d1
                var p1_c = Scalar[DTYPE](0)
                var p1_d0 = Scalar[DTYPE](0)
                var p1_d1 = Scalar[DTYPE](0)
                _bl_peval(p1_a, p1_c, p1_d0, p1_d1, lsiter_b)
                var done_ls = False
                if abs(p1_d0) < gtol_b:
                    alpha = p1_a
                    done_ls = True

                var dir_b = (
                    Scalar[DTYPE](1) if p1_d0 < Scalar[DTYPE](0)
                    else Scalar[DTYPE](-1)
                )
                var p2_a = Scalar[DTYPE](0)
                var p2_c = p0_c
                var p2_d0 = p0_d0
                var p2_d1 = p0_d1
                if not done_ls:
                    # Phase 2: one-sided Newton search to a sign change.
                    while p1_d0 * dir_b <= -gtol_b and lsiter_b < ls_budget:
                        p2_a = p1_a
                        p2_c = p1_c
                        p2_d0 = p1_d0
                        p2_d1 = p1_d1
                        p1_a = p1_a - p1_d0 / p1_d1
                        _bl_peval(p1_a, p1_c, p1_d0, p1_d1, lsiter_b)
                        if abs(p1_d0) < gtol_b:
                            alpha = p1_a
                            done_ls = True
                            break
                    # Could not bracket within the budget (LSresult 3).
                    if not done_ls and lsiter_b >= ls_budget:
                        alpha = p1_a
                        done_ls = True

                if not done_ls:
                    # Phase 3: the BRACKETED search over {p1next, p2next,
                    # pmid} — NOT a bisection. See `primal.mojo`'s docstring:
                    # the Newton next-point off a bracket end is what lands
                    # the root, and halving from the midpoint spends the whole
                    # budget without reaching `gtol`.
                    var n2_a = p1_a
                    var n2_c = p1_c
                    var n2_d0 = p1_d0
                    var n2_d1 = p1_d1
                    var n1_a = p1_a - p1_d0 / p1_d1
                    var n1_c = Scalar[DTYPE](0)
                    var n1_d0 = Scalar[DTYPE](0)
                    var n1_d1 = Scalar[DTYPE](0)
                    _bl_peval(n1_a, n1_c, n1_d0, n1_d1, lsiter_b)

                    var pm_a: Scalar[DTYPE]
                    var pm_c = Scalar[DTYPE](0)
                    var pm_d0 = Scalar[DTYPE](0)
                    var pm_d1 = Scalar[DTYPE](0)
                    var ZB = Scalar[DTYPE](0)

                    while lsiter_b < ls_budget:
                        pm_a = Scalar[DTYPE](0.5) * (p1_a + p2_a)
                        _bl_peval(pm_a, pm_c, pm_d0, pm_d1, lsiter_b)

                        var best_a = ZB
                        var best_c = ZB
                        var has_best = False
                        if abs(n1_d0) < gtol_b:
                            best_a = n1_a
                            best_c = n1_c
                            has_best = True
                        if abs(n2_d0) < gtol_b and (
                            not has_best or n2_c < best_c
                        ):
                            best_a = n2_a
                            best_c = n2_c
                            has_best = True
                        # ⚠ NO `best_c = pm_c` — see `primal.mojo`'s note
                        # at the same candidate: `engine_solver.c:1842`
                        # writes it inside a LOOP over three candidates, and
                        # `pmid` is the last, so the unrolled write is dead.
                        if abs(pm_d0) < gtol_b and (
                            not has_best or pm_c < best_c
                        ):
                            best_a = pm_a
                            has_best = True
                        if has_best:
                            alpha = best_a
                            done_ls = True
                            break

                        # `updateBracket` (engine_solver.c:1665), per end.
                        var b1 = False
                        if p1_d0 < ZB and n1_d0 < ZB and p1_d0 < n1_d0:
                            p1_a = n1_a; p1_c = n1_c
                            p1_d0 = n1_d0; p1_d1 = n1_d1; b1 = True
                        elif p1_d0 > ZB and n1_d0 > ZB and p1_d0 > n1_d0:
                            p1_a = n1_a; p1_c = n1_c
                            p1_d0 = n1_d0; p1_d1 = n1_d1; b1 = True
                        if p1_d0 < ZB and n2_d0 < ZB and p1_d0 < n2_d0:
                            p1_a = n2_a; p1_c = n2_c
                            p1_d0 = n2_d0; p1_d1 = n2_d1; b1 = True
                        elif p1_d0 > ZB and n2_d0 > ZB and p1_d0 > n2_d0:
                            p1_a = n2_a; p1_c = n2_c
                            p1_d0 = n2_d0; p1_d1 = n2_d1; b1 = True
                        if p1_d0 < ZB and pm_d0 < ZB and p1_d0 < pm_d0:
                            p1_a = pm_a; p1_c = pm_c
                            p1_d0 = pm_d0; p1_d1 = pm_d1; b1 = True
                        elif p1_d0 > ZB and pm_d0 > ZB and p1_d0 > pm_d0:
                            p1_a = pm_a; p1_c = pm_c
                            p1_d0 = pm_d0; p1_d1 = pm_d1; b1 = True

                        var b2 = False
                        if p2_d0 < ZB and n1_d0 < ZB and p2_d0 < n1_d0:
                            p2_a = n1_a; p2_c = n1_c
                            p2_d0 = n1_d0; p2_d1 = n1_d1; b2 = True
                        elif p2_d0 > ZB and n1_d0 > ZB and p2_d0 > n1_d0:
                            p2_a = n1_a; p2_c = n1_c
                            p2_d0 = n1_d0; p2_d1 = n1_d1; b2 = True
                        if p2_d0 < ZB and n2_d0 < ZB and p2_d0 < n2_d0:
                            p2_a = n2_a; p2_c = n2_c
                            p2_d0 = n2_d0; p2_d1 = n2_d1; b2 = True
                        elif p2_d0 > ZB and n2_d0 > ZB and p2_d0 > n2_d0:
                            p2_a = n2_a; p2_c = n2_c
                            p2_d0 = n2_d0; p2_d1 = n2_d1; b2 = True
                        if p2_d0 < ZB and pm_d0 < ZB and p2_d0 < pm_d0:
                            p2_a = pm_a; p2_c = pm_c
                            p2_d0 = pm_d0; p2_d1 = pm_d1; b2 = True
                        elif p2_d0 > ZB and pm_d0 > ZB and p2_d0 > pm_d0:
                            p2_a = pm_a; p2_c = pm_c
                            p2_d0 = pm_d0; p2_d1 = pm_d1; b2 = True

                        if b1:
                            n1_a = p1_a - p1_d0 / p1_d1
                            _bl_peval(n1_a, n1_c, n1_d0, n1_d1, lsiter_b)
                        if b2:
                            n2_a = p2_a - p2_d0 / p2_d1
                            _bl_peval(n2_a, n2_c, n2_d0, n2_d1, lsiter_b)

                        if not b1 and not b2:
                            alpha = pm_a
                            done_ls = True
                            break

                if not done_ls:
                    # No convergence: take the cheaper bracket, and ONLY if it
                    # actually improves — otherwise 0, so the caller breaks
                    # without moving `qacc`.
                    if p1_c <= p2_c and p1_c < Scalar[DTYPE](0):
                        alpha = p1_a
                    elif p2_c <= p1_c and p2_c < Scalar[DTYPE](0):
                        alpha = p2_a

            # ⚠ HERE, NOT AT THE IMPROVEMENT TEST — `lsiter_b` is scoped to this
            # tid-0 block and that test lives further down the loop body. And
            # anchored on `alpha = p2_a` because the bare `if alpha < 1e-10`
            # appears TWICE: the per-env solver carries the same guard.
            comptime if NEWTON_ITER_REPORT:
                ls_evals += lsiter_b
            if alpha < Scalar[DTYPE](1e-10):
                ctrl_sh[1] = Scalar[DTYPE](1)  # done (break next iter)
            else:
                ctrl_sh[1] = Scalar[DTYPE](0)

                # Save old state for revert.
                for i in range(NV):
                    old_qacc[i] = qacc[i]
                    old_Ma[i] = Ma[i]
                    old_qfrc[i] = qfrc[i]
                for e_idx in range(num_edges_b):
                    old_jar[e_idx] = jar[e_idx]
                    old_force[e_idx] = rebind[Scalar[DTYPE]](force_sh[e_idx])

                old_cost = Scalar[DTYPE](0)
                for i in range(NV):
                    old_cost += (
                        Scalar[DTYPE](0.5)
                        * (Ma[i] - f_smooth[i])
                        * (qacc[i] - qacc_smooth[i])
                    )
                for e_idx in range(num_edges_b):
                    old_cost += scalar_row_cost[DTYPE](
                        Int(rebind[Scalar[DTYPE]](state_e_sh[e_idx])),
                        jar[e_idx],
                        rebind[Scalar[DTYPE]](De_sh[e_idx]),
                        rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                        rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                    )

                for i in range(NV):
                    qacc[i] += alpha * search[i]
                    Ma[i] += alpha * Mv[i]

            # Publish qacc unconditionally. When alpha<1e-10 qacc is unchanged,
            # so the cooperative recompute reproduces identical jar/force/qfrc.
            for i in range(NV):
                qacc_sh[i] = qacc[i]

        # Cooperative jar/force/qfrc recompute, then tid 0 reads back and
        # finishes the accept/revert.
        barrier()
        _recompute_jfq_coop[DTYPE, JE_AS=JE_AS](
            tid, COOP, num_edges_b, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), Je_sh, De_sh, bias_e_sh,
            kind_e_sh, R_e_sh, floss_e_sh, state_e_sh, qacc_sh,
            jar_sh, force_sh, qfrc_sh,
        )
        barrier()
        if valid_env and tid == 0:
            for e_idx in range(num_edges_b):
                jar[e_idx] = rebind[Scalar[DTYPE]](jar_sh[e_idx])
            for i in range(NV):
                qfrc[i] = rebind[Scalar[DTYPE]](qfrc_sh[i])
            if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 0:
                var new_cost: Scalar[DTYPE] = 0
                for i in range(NV):
                    new_cost += (
                        Scalar[DTYPE](0.5)
                        * (Ma[i] - f_smooth[i])
                        * (qacc[i] - qacc_smooth[i])
                    )
                for e_idx in range(num_edges_b):
                    new_cost += scalar_row_cost[DTYPE](
                        Int(rebind[Scalar[DTYPE]](state_e_sh[e_idx])),
                        jar[e_idx],
                        rebind[Scalar[DTYPE]](De_sh[e_idx]),
                        rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                        rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                    )

                var improvement = scale * (old_cost - new_cost)
                # ⚠⚠ THE THIRD EXIT, and the one that made the first MIN_ITER
                # sweep read FLAT from 20 to 180. It sets `ctrl_sh[1]` and the
                # loop breaks on it, and the probe did not guard it — so the
                # loop left at the same iteration whatever the minimum asked
                # for. Missed because the search for exits was BOUNDED to the
                # lines around the loop HEAD, and this one lives ~600 lines
                # further down inside the line-search tail. Enumerate a loop's
                # exits over its WHOLE body, not over the part you are reading.
                # ⚠ DECLARED OUTSIDE THE `comptime if` — that construct opens
                # a scope, so a `var` inside either branch is not visible here.
                var _stop = False
                comptime if NEWTON_MIN_ITER == 0:
                    _stop = improvement < tol_rt and iter_n > 0
                else:
                    _stop = (
                        improvement < tol_rt and iter_n >= NEWTON_MIN_ITER
                    )
                if _stop:
                    if improvement < Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc[i] = old_qacc[i]
                            Ma[i] = old_Ma[i]
                            qfrc[i] = old_qfrc[i]
                        for e_idx in range(num_edges_b):
                            jar[e_idx] = old_jar[e_idx]
                            force_sh[e_idx] = old_force[e_idx]
                    ctrl_sh[1] = Scalar[DTYPE](1)  # done

        # force_sh updated; make visible for next assembly.
        barrier()
        if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 1:
            break

    comptime if NEWTON_ITER_REPORT:
        if valid_env and tid == 0 and env == 0:
            # ⚠ TWO NUMBERS, BECAUSE ONE OF THEM IS THE MULTIPLIER. `_bl_peval`
            # is the line search's unit of work; `NEWTON_SERIAL_PROBE == 9`
            # prices ONE of them. Cost = price x count, both measured on the
            # SAME run — which is exactly what the discarded "54%" figure was
            # not: that subtracted an unpinned slope from a pinned fraction.
            print("[niter]", iters_done, "[lseval]", ls_evals)

    # ⚠ STAGE 5 — after the Newton LOOP, before the write-back tail. This is the
    # stage that had to exist: stages 1-4 bisect the SETUP and stop at the loop,
    # so everything after them was one 61% lump containing both the loop and the
    # tail. MIN_ITER says the loop is ~5%, which can only be reconciled with the
    # lump if the TAIL is the bulk — or if a real iteration costs vastly more
    # than the post-convergence ones MIN_ITER was able to force. This separates
    # them, and it is the discriminator between two very different targets.
    comptime if NEWTON_STOP_AFTER == 5:
        return

    # === THREAD 0: write back + reconstruct forces + equality/tendon ===
    if not valid_env or tid != 0:
        return

    # ── mj_solNoSlip (BLOCKED kernel) ──────────────────────────────────────
    # The friction-only Gauss-Seidel sweep with the normal forces frozen, run
    # after the primal solve. Off unless the model asks for it
    # (`<option noslip_iterations>`).
    #
    # ⚠⚠ THIS KERNEL ACCEPTED `NOSLIP_ITER` AND NEVER READ IT until 2026-08-13,
    # so the pass ran on the CPU branch of `solve_newton_blocked` (which
    # delegates to `_newton_solve_env`) and silently vanished on the GPU one.
    # That is not a latent trap: `solve_newton` routes PYRAMIDAL + NVIDIA here,
    # and dm_control's dog is PYRAMIDAL with `noslip_iterations="4"` and is
    # trained batched on GPU — so the two branches of ONE function were
    # computing different physics from identical inputs. Measured on the dog
    # model, MuJoCo against itself with only the option changed moves
    # `max|d(qvel)|` by 2.9e-2 on the FIRST contacting step.
    #
    # PYRAMIDAL branch, matching this kernel — `noslip_pyramidal`, never
    # `noslip_elliptic`. There is no runtime test to get wrong: the elliptic
    # cone has no cooperative port and `solve_newton` cannot route it here.
    #
    # Runs on THREAD 0 ONLY, and safely: every other thread returned at the
    # guard above, so the shared rows it rewrites have no concurrent reader.
    # `mj_solNoSlip` is Gauss-Seidel — sequential by construction — so this
    # costs no parallelism that the algorithm could have used.
    #
    # ⚠ POSITION IS PART OF THE PORT. It must run BEFORE the `qacc` write-back
    # and the contact-force reconstruction below, because it rewrites both
    # `qacc` and `force_sh`. Same placement as the per-env path.
    comptime if NOSLIP_ITER > 0:
        noslip_pyramidal[
            DTYPE, ME, V_SIZE, MC, MAX_CONTACTS, MAX_CONDIM,
            # `Je` is SHARED or GLOBAL depending on whether it fit (see
            # `JE_IN_SHARED`); the other rows are always threadgroup memory.
            # ⚠ BY KEYWORD: the inferred `L_*` layout parameters now sit
            # between the dimensions and these, so a positional `JE_AS`
            # would bind to `L_CONTACTS`.
            JE_AS=JE_AS,
            ROW_AS=AddressSpace.SHARED,
        ](
            env,
            nc,
            num_edges_b,
            contacts,
            m_inv,
            Je_sh.ptr,
            bias_e_sh.ptr,
            kind_e_sh.ptr,
            R_e_sh.ptr,
            floss_e_sh.ptr,
            qacc_smooth,
            # ⚠ THE KERNEL'S OWN `scale`, not a recomputation. It is the same
            # `1 / (meaninertia * max(1, nv))` model constant the primal loop
            # above used, already guarded against a degenerate meaninertia.
            # Recomputing it here would be a second expression for one
            # quantity, and `scale` decides WHEN the sweep stops.
            scale,
            # ⚠ FROM META, NOT the `NOSLIP_TOLERANCE` constant — that constant
            # is only the absent-attribute default. See the per-env path.
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NOSLIP_TOLERANCE]),
            # ⚠ FROM META TOO — see the per-env pyramidal call. On this path
            # the read is uniform across the threadgroup (every thread loads
            # the same model slot), so it costs no divergence.
            Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NOSLIP_ITERATIONS])),
            qacc,
            jar,
            force_sh.ptr,
            qfrc,
            NV,
        )

    for i in range(NV):
        qacc_constrained[env, i] = qacc[i]

    for c in range(nc):
        var fn_c: Scalar[DTYPE] = 0
        var ft1_c: Scalar[DTYPE] = 0
        var ft2_c: Scalar[DTYPE] = 0
        var mu_c = rebind[Scalar[DTYPE]](solver[env, pyr_sc + 2 * NE * MC + c])
        var safe_mu = mu_c
        if safe_mu < Scalar[DTYPE](1e-8):
            safe_mu = Scalar[DTYPE](1e-8)
        var f_e0 = rebind[Scalar[DTYPE]](force_sh[c * NE + 0])
        var f_e1 = rebind[Scalar[DTYPE]](force_sh[c * NE + 1])
        var f_e2 = rebind[Scalar[DTYPE]](force_sh[c * NE + 2])
        var f_e3 = rebind[Scalar[DTYPE]](force_sh[c * NE + 3])
        # `mju_decodePyramid`: the normal force is the SUM of the four edge
        # forces, NOT half of it. Both engines build each edge as
        # `Jn +- mu*Jt` with a FULL Jn (engine_core_constraint.c:1003), so
        # halving it made every pyramidal contact RECORD read half true
        # while qacc stayed correct — the solver works in edge forces and
        # only this write-back was wrong. Its two consumers are cfrc_ext
        # (hence Ant's contact_cost, a squared norm that had been costing a
        # quarter of what it should) and the quadruped force/torque
        # sensors. Fixed 2026-07-31.
        fn_c = f_e0 + f_e1 + f_e2 + f_e3
        var c_off = c * CONTACT_SIZE
        # Frictionless contacts carry no tangential force — see the identical
        # guard in the per-env path above for the measurement and the reason
        # `qacc` is unaffected while the sensors are not.
        var dim_c = Int(
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_CONDIM])
        )
        if dim_c > 1:
            ft1_c = (f_e0 - f_e1) * safe_mu
            ft2_c = (f_e2 - f_e3) * safe_mu
        contacts[env, c_off + CONTACT_IDX_FORCE_N] = fn_c
        contacts[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_c
        contacts[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_c

    # NOTHING RUNS AFTER THE SOLVE ON THIS KERNEL EITHER. Joint limits,
    # dry-friction dofs, tendon equalities (fixed and spatial) and
    # connect/weld are all edge rows above. Both post-passes were removed on
    # 2026-08-12 — the tendon one because `build_tendon_equality_rows` covers
    # both kinds now, and `_equality_env` with the defect-29a conversion that
    # reached this kernel last of the three.


def solve_newton_blocked[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.PYRAMIDAL,
    BATCH: Int = 1,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
    # Per-env spill size for `Je`; 0 = it fits threadgroup memory. Comes
    # from `je_budget.je_ws_size` via the integrator — never computed here.
    JE_WS: Int = 0,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    mut cscratch: ContactScratch[DTYPE, D, BATCH, JE_WS],
    ctx: Optional[DeviceContext] = None,
) raises:
    """PYRAMIDAL-only ONE-ENV-PER-BLOCK Newton contact solve (fields port of
    NewtonSolver.solve_gpu_blocked). Cooperative across MAX_CONTACTS threads,
    big matrices in shared memory — the OOM-safe path at humanoid scale.

    Writes into `scratch.qacc_constrained` (+ solved forces into `d.contacts`).
    Same signature family as `solve_newton`. Only the GPU (blocked)
    launch is meaningful; the CPU branch falls back to the single-source per-env
    body (`_newton_solve_env`, identical PYRAMIDAL math) for parity.
    """
    comptime MC = _max_one[D.MAX_CONTACTS]()
    comptime SOLVER_WS = 81 * MC + 12 * MC * D.NV

    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
    comptime L_CON = Layout.row_major(BATCH, D.MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_JOINT = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    # ⚠ FLAT, matching `build_dof_segments`. `Model.L_TREE` is the 2-D view.
    comptime L_TREES = Layout.row_major(D.NV * MODEL_TREE_SIZE)
    comptime L_EQ = Layout.row_major(D.NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TEN = Layout.row_major(D.NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(D.NSITE, MODEL_SITE_SIZE)
    comptime L_GEOM_W = Layout.row_major(D.NGEOM, MODEL_GEOM_SIZE)
    comptime L_BW = Layout.row_major(D.NBODY, 2)
    comptime L_CDOF = Layout.row_major(BATCH, D.NV * 6)
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_SOLVER = Layout.row_major(BATCH, SOLVER_WS)

    # ⚠ FLOORED AT 1 to match ContactScratch.JE_ELEMS — a zero-extent
    # operand segfaults instead of being an empty tensor.
    comptime JE_ELEMS = JE_WS if JE_WS > 0 else 1
    comptime L_JE_WS = Layout.row_major(BATCH, JE_ELEMS)
    comptime L_QPOS = Layout.row_major(BATCH, D.NQ)
    comptime L_DW = Layout.row_major(D.NV)

    comptime if target == "cpu":
        var dm = d.dims
        var rl_QPOS = rl2(BATCH, dm.get_nq())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_CON = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
        var rl_SMETA = rl2(BATCH, METADATA_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_MMETA = rl1(MODEL_META_SIZE)
        var rl_EQ = rl2(dm.get_nequality(), MODEL_EQ_SIZE)
        var rl_TEN = rl2(dm.get_ntendon(), MODEL_TENDON_SIZE)
        var rl_SITE = rl2(dm.get_nsite(), MODEL_SITE_SIZE)
        var rl_GEOM_W = rl2(dm.get_ngeom(), MODEL_GEOM_SIZE)
        var rl_BW = rl2(dm.get_nbody(), 2)
        var rl_DW = rl1(dm.get_nv())
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        # ⚠⚠ RUNTIME BUDGET, NOT THE COMPTIME `SOLVER_WS`. On a dynamic
        # provider `D.NV` is DIM_POISON and `D.MAX_CONTACTS` floors to 1,
        # so the comptime literal is 81 - 12 = 69 scalars for EVERY model
        # while the ws_* offsets below are computed from the RUNTIME nv/mc.
        # The spelling was swept to `rl2`/`lt_dyn` in 3a; the VALUE was not.
        var rl_SOLVER = rl2(
            BATCH, ws_budget(_max_one_rt(dm.get_max_contacts()), dm.get_nv())
        )
        var qpos_v = d.qpos.lt_dyn["cpu", DYN2](rl_QPOS)
        var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
        var xpos_v = d.xpos.lt_dyn["cpu", DYN2](rl_B3)
        var xquat_v = d.xquat.lt_dyn["cpu", DYN2](rl_B4)
        var stcom_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_B3)
        var con_v = d.contacts.lt_dyn["cpu", DYN2](rl_CON)
        var smeta_v = d.meta.lt_dyn["cpu", DYN2](rl_SMETA)
        var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
        var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
        var mmeta_v = m.meta.lt_dyn["cpu", DYN1](rl_MMETA)
        var rl_TREES = rl1(dm.get_nv() * MODEL_TREE_SIZE)
        var trees_v = m.trees.lt_dyn["cpu", DYN1](rl_TREES)
        var eq_v = m.equality.lt_dyn["cpu", DYN2](rl_EQ)
        var ten_v = m.tendons.lt_dyn["cpu", DYN2](rl_TEN)
        var site_v = m.sites.lt_dyn["cpu", DYN2](rl_SITE)
        var geomw_v = m.geoms.lt_dyn["cpu", DYN2](rl_GEOM_W)
        var bw_v = m.body_invweight0.lt_dyn["cpu", DYN2](rl_BW)
        var dw_v = m.dof_invweight0.lt_dyn["cpu", DYN1](rl_DW)
        var cdof_v = scratch.cdof.lt_dyn["cpu", DYN2](rl_CDOF)
        var M_v = scratch.M.lt_dyn["cpu", DYN2](rl_M)
        var mi_v = scratch.m_inv.lt_dyn["cpu", DYN2](rl_M)
        var qc_v = scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV)
        var qw_v = d.qacc_warmstart.lt_dyn["cpu", DYN2](rl_NV)
        var sol_v = cscratch.solver.lt_dyn["cpu", DYN2](rl_SOLVER)
        for e in range(BATCH):
            _newton_solve_env[
                DTYPE, CONE_TYPE, BATCH, SOLVER_WS, MAX_CONDIM=MAX_CONDIM, NOSLIP_ITER=NOSLIP_ITER](
                e, dm, qpos_v, qvel_v, xpos_v, xquat_v, stcom_v, con_v, smeta_v,
                joints_v, bodies_v, mmeta_v, trees_v, eq_v, ten_v, site_v, geomw_v, bw_v, dw_v,
                cdof_v, M_v, mi_v, qc_v, qw_v, sol_v,
            )
    else:
        var c = ctx.value()
        c.enqueue_function[
            _newton_blocked_fields_kernel[
                DTYPE, D.NQ, D.NV, D.NBODY, D.NJOINT, D.MAX_CONTACTS, D.NGEOM, D.NEQUALITY,
                D.NTENDON, D.NSITE, CONE_TYPE, BATCH, SOLVER_WS,
                MAX_CONDIM,
                NOSLIP_ITER,
                JE_WS,
            ]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            d.subtree_com.lt["gpu", L_B3](),
            d.contacts.lt["gpu", L_CON](),
            d.meta.lt["gpu", L_SMETA](),
            m.joints.lt["gpu", L_JOINT](),
            m.bodies.lt["gpu", L_BODY](),
            m.meta.lt["gpu", L_MMETA](),
            m.trees.lt["gpu", L_TREES](),
            m.equality.lt["gpu", L_EQ](),
            m.tendons.lt["gpu", L_TEN](),
            m.sites.lt["gpu", L_SITE](),
                m.geoms.lt["gpu", L_GEOM_W](),
            m.body_invweight0.lt["gpu", L_BW](),
            m.dof_invweight0.lt["gpu", L_DW](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.M.lt["gpu", L_M](),
            scratch.m_inv.lt["gpu", L_M](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            d.qacc_warmstart.lt["gpu", L_NV](),
            cscratch.solver.lt["gpu", L_SOLVER](),
            cscratch.je.lt["gpu", L_JE_WS](),
            grid_dim=(BATCH,),
            # ⚠ SAME SOURCE AS THE KERNEL'S `THREADS`, not `MC`. They were
            # equal by coincidence of both spelling `_max_one[MAX_CONTACTS]`.
            block_dim=(newton_block_threads[D.MAX_CONTACTS](),),
        )
