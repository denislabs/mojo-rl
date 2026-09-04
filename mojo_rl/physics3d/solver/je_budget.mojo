"""Size of the blocked-Newton constraint Jacobian, and whether it must spill.

⚠⚠ ONE SOURCE OF TRUTH, ON PURPOSE. Two places need this number and they MUST
agree exactly: `ContactScratch` allocates the spill buffer, and the blocked
Newton kernel indexes it. If the allocation is a single scalar smaller than
what the kernel writes, the overrun lands in whatever tensor was allocated
next — silent corruption, not a crash. Hence this module rather than the
formula written twice.

WHY A SPILL EXISTS. `Je_sh` is `ME * NV` scalars of THREADGROUP memory and is
usually the largest single array in the blocked kernel. Measured on NVIDIA
(2026-08-10), humanoid_CMU asked for 169,820 B against a 101,376 B limit and
`ptxas` refused to compile the kernel at all. Models past the budget put `Je`
in a dedicated global buffer instead; models under it keep the threadgroup
array and are untouched, bit for bit.

⚠⚠ THE BUDGET IS THE TOTAL, AND WAS `Je` ALONE UNTIL P4. That is the defect
this file existed with: at the k=12 park scene `Je` is 54 KB — comfortably
under the old 64 KB — so it declined to spill, while the three `NV*NV` matrices
put the block at 136,212 B and `ptxas` refused. `Je` is the biggest array on a
high-CONTACT model, which is what this was tuned on; it is not the biggest on a
high-nv LOW-contact one, which is exactly the shape a fixed scene budget
produces. `newton_shared_elems` below counts all eleven arrays.

Measured sizes (float32). ⚠ THE TOTAL IS WHAT DECIDES; `Je` is shown only
because it is the term that used to:

    model            NV   ME    Je      TOTAL    spills?
    quadruped        22  156   13 KB    25 KB    no
    humanoid         27  199   21 KB    37 KB    no
    quadruped_fetch  28  340   37 KB    59 KB    no   (at its real condim 6)
    humanoid_CMU     62  432  105 KB   166 KB    YES
    dog              79  491  152 KB   244 KB    YES
    dog_fetch        85  539  179 KB   285 KB    YES
    so101_park k=9   60  154   36 KB    85 KB    no   <- the ceiling before P4
    so101_park k=10  66  162   42 KB   100 KB    YES  <- would NOT COMPILE before
    so101_park k=12  78  178   54 KB   134 KB    YES  <- would NOT COMPILE before

None of the first six changes its answer under the new rule — gated in
`tests/physics3d/test_newton_shared_budget.mojo`, arm E, because widening a
budget can silently start spilling models that ran fine, and a spilled `Je` is
re-read from global on every Newton iteration.

⚠ SPILLING REACHES k=13, NOT FURTHER. Past that the three `NV*NV` arrays are
the binding term and the fix is a different one (move the Hessian to global, as
mujoco_warp does, or pack the triple by block).
"""

from std.sys.info import size_of


def newton_block_threads[MAX_CONTACTS: Int]() -> Int:
    """Threads per block for `_newton_blocked_fields_kernel`.

    ⚠⚠ ONE SOURCE FOR TWO PLACES THAT MUST NOT DISAGREE — the kernel's
    cooperative stride (`comptime THREADS`) and the launch's `block_dim`. They
    were two independent spellings of `_max_one[MAX_CONTACTS]()`, which is
    exactly the shape of `_a_rule_written_inline_twice_drifts`: numerically
    equal today, and a silent out-of-range thread the moment one moves.

    ⚠ IT MUST NEVER RETURN LESS THAN `MAX_CONTACTS`. The contact phases map one
    slot to one thread, so a smaller block leaves the tail slots
    UNINITIALISED — `_init_common_normal_ws` never runs for them and the
    workspace keeps the previous step's values. More is safe (every such phase
    is now guarded `< MC` or `< nc`); fewer is a wrong answer.

    Today it returns exactly `MAX_CONTACTS`, so the launch is unchanged. It
    exists so that changing the shape is a one-line edit HERE rather than two
    edits 2,000 lines apart.
    """
    return _max_one[MAX_CONTACTS]()


# ⚠⚠ THE PER-BLOCK SHARED LIMIT THE BLOCKED KERNEL IS COMPILED AGAINST, and it
# is an NVIDIA number ON PURPOSE. `solve_newton` routes PYRAMIDAL + NVIDIA to
# `solve_newton_blocked` and everything else to the one-thread-per-env kernel,
# which holds `Je` as per-thread `InlineArray`s and never consults this file.
# So the only consumer of this budget is a kernel that only ever runs on CUDA.
#
# 0x18c00 is what `ptxas` itself reports as the maximum on an RTX 5090:
#
#     ptxas error : Entry function 'mojo_rl_physics3d_solver_newt...' uses
#                   too much shared data (0x21414 bytes, 0x18c00 max)
#
# ⚠ THE OLD 64 KB WAS INCOHERENT, WHICH IS WHY IT IS GONE. It was justified as
# "the widely-supported opt-in floor", so that a model fitting everywhere kept
# the fast path — but it was compared against `Je` ALONE while the kernel's
# TOTAL was already 87 KB at k=9 and compiling fine. A budget that guards one
# array against a portability figure the whole block has already blown is not
# protecting portability; it is just failing to predict `ptxas`.
comptime SOLVER_SHARED_BUDGET: Int = 0x18C00


def _max_one[N: Int]() -> Int:
    """`max(N, 1)` — a zero-sized dimension is a crash, not an empty tensor."""
    return N if N > 0 else 1


def je_edge_rows[
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Int:
    """`ME` — the blocked solver's constraint-row count.

    ⚠ MUST MATCH `newton_solve.solve_newton_blocked`'s `ME` EXACTLY. The terms,
    in the order that file derives them:

        NE       = 2*(MAX_CONDIM-1)   pyramidal edges per contact
        MAX_LIM  = max(1, 2*NJOINT)   joint limits (lo + hi)
        MAX_FRIC = max(1, NV)         one dry-friction row per dof
        MAX_TLIM = 2*NTENDON          tendon limits (lo + hi)
        MAX_TEQ  = NTENDON            one bilateral row per equality tendon
        MAX_WELD = 6*NEQUALITY        connect (3) / weld (6) rows

    ⚠ The friction and tendon terms were MISSING from the blocked path until
    2026-07-31, so a model with `frictionloss` or a limited tendon silently had
    no such rows. `MAX_WELD` arrived 2026-08-12 with the defect-29a conversion
    of connect/weld from a post-pass into rows. Growing this function grows the
    spill buffer with it — that is the point of routing both through here.

    ⚠ NEQUALITY IS A PARAMETER, not a term folded into another. It was
    tempting to reuse NTENDON's slot since both are "equality" counts; they are
    different models' dimensions and a model can have either without the other.
    """
    return (
        2 * (MAX_CONDIM - 1) * _max_one[MAX_CONTACTS]()
        + _max_one[2 * NJOINT]()
        + _max_one[NV]()
        + 3 * NTENDON
        + 6 * NEQUALITY
    )


def je_elems[
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Int:
    """Scalars in `Je` for ONE env: `ME * V_SIZE`."""
    return (
        je_edge_rows[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
        ]()
        * _max_one[NV]()
    )


def newton_shared_elems[
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
    JE_IN_SHARED: Bool,
]() -> Int:
    """Scalars of THREADGROUP memory `_newton_blocked_fields_kernel` asks for.

    ⚠⚠ MUST MATCH THE KERNEL'S `stack_allocation()` LIST EXACTLY — this is the
    same one-source-of-truth contract `je_elems` has with `ME`, and it is the
    thing the old budget got wrong by counting a single array. In the order the
    kernel declares them (`newton_solve.mojo:3640+`):

        M_sh, H_sh, L_sh          3 * max(1, NV*NV)
        seg0_sh, seg1_sh          2 * max(1, NV)      (PN2c)
        grad_sh                   1 * max(1, NV)      (F3b)
        Je_sh                     ME * max(1, NV), or 1 when spilled
        De/bias_e/force/kind_e/
        R_e/floss_e/state_e/
        Jv_e/jar                  9 * ME
        search/Mv/qacc/qfrc       4 * max(1, NV)
        ctrl_sh                   3

    ⚠ VERIFIED AGAINST `ptxas` ON FOUR POINTS, not derived and hoped for — see
    `tests/physics3d/test_newton_shared_budget.mojo`, which pins it to the byte
    counts the k=6/9/10/12 park scenes produced.
    """
    return (
        3 * _max_one[NV * NV]()
        # 2 seg + 1 grad + search/Mv/qacc/qfrc
        + 7 * _max_one[NV]()
        + (
            je_elems[
                NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
            ]() if JE_IN_SHARED else 1
        )
        + 9
        * je_edge_rows[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
        ]()
        + 3
    )


def je_spills[
    DTYPE: DType,
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Bool:
    """Does the kernel's TOTAL threadgroup footprint force `Je` out?

    ⚠⚠ THE TOTAL, NOT `Je`. This compared `Je` alone against 64 KB until P4,
    and the failure mode is on record: at the k=12 park scene `Je` is 54 KB —
    comfortably under — so it declined to spill, while the three `NV*NV`
    matrices put the block at 136,212 B against a 101,376 B limit and `ptxas`
    refused to compile the kernel at all. Budgeting one array out of eleven
    cannot predict that, and the models it WAS tuned on (humanoid_CMU, dog)
    hid it because they are high-nv AND high-contact, so `Je` dominated. A
    fixed scene budget produces the shape it was never tuned for: high nv, LOW
    contact count.
    """
    return (
        newton_shared_elems[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM, True
        ]()
        * size_of[Scalar[DTYPE]]()
    ) > SOLVER_SHARED_BUDGET


def je_ws_size[
    DTYPE: DType,
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Int:
    """Per-env spill-buffer size: `ME*NV` when spilling, else 0.

    `ContactScratch` allocates `BATCH * max(this, 1)`; a model that does not
    spill pays one scalar per env, not a buffer.
    """
    comptime if je_spills[
        DTYPE, NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
    ]():
        return je_elems[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
        ]()
    return 0
