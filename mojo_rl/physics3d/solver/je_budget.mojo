"""Size of the blocked-Newton constraint Jacobian, and whether it must spill.

⚠⚠ ONE SOURCE OF TRUTH, ON PURPOSE. Two places need this number and they MUST
agree exactly: `ContactScratch` allocates the spill buffer, and the blocked
Newton kernel indexes it. If the allocation is a single scalar smaller than
what the kernel writes, the overrun lands in whatever tensor was allocated
next — silent corruption, not a crash. Hence this module rather than the
formula written twice.

WHY A SPILL EXISTS. `Je_sh` is `ME * NV` scalars of THREADGROUP memory and
dominates the blocked kernel's shared-memory footprint. Measured on NVIDIA
(2026-08-10), humanoid_CMU asked for 169,820 B against a 101,376 B limit and
`ptxas` refused to compile the kernel at all — `Je` alone was 104 KB. Models
past the budget put `Je` in a dedicated global buffer instead; models under it
keep the threadgroup array and are untouched, bit for bit.

⚠ THE BUDGET IS A COMPILE-TIME GUESS AT A RUNTIME LIMIT. Shared memory per
block is device-specific (99 KB on the box this was measured on, 227 KB on an
H100) and the kernel is compiled without knowing the target. 64 KB is
deliberately conservative — the widely-supported opt-in floor — so a model that
fits everywhere keeps the fast path and anything near the edge spills rather
than failing to compile on the smallest plausible device.

Measured sizes (float32):

    model          NV  MC   ME   Je      spills?
    quadruped      22  16  156   13 KB   no
    humanoid       27  32  199   20 KB   no
    quadruped_fetch 28 24  340   37 KB   no   (at its real condim 6)
    humanoid_CMU   62  64  432  104 KB   YES
    dog            79  24  491  151 KB   YES
    dog_fetch      85  28  539  178 KB   YES
"""

from std.sys.info import size_of


comptime JE_SHARED_BUDGET: Int = 64 * 1024


def _max_one[N: Int]() -> Int:
    """`max(N, 1)` — a zero-sized dimension is a crash, not an empty tensor."""
    return N if N > 0 else 1


def je_edge_rows[
    NV: Int, NJOINT: Int, NTENDON: Int, MAX_CONTACTS: Int, MAX_CONDIM: Int
]() -> Int:
    """`ME` — the blocked solver's constraint-row count.

    ⚠ MUST MATCH `newton_solve.solve_newton_blocked`'s `ME` EXACTLY. The terms,
    in the order that file derives them:

        NE       = 2*(MAX_CONDIM-1)   pyramidal edges per contact
        MAX_LIM  = max(1, 2*NJOINT)   joint limits (lo + hi)
        MAX_FRIC = max(1, NV)         one dry-friction row per dof
        MAX_TLIM = 2*NTENDON          tendon limits (lo + hi)
        MAX_TEQ  = NTENDON            one bilateral row per equality tendon

    ⚠ The last three were MISSING from the blocked path until 2026-07-31, so a
    model with `frictionloss` or a limited tendon silently had no such rows.
    Growing this function grows the spill buffer with it — that is the point of
    routing both through here.
    """
    return (
        2 * (MAX_CONDIM - 1) * _max_one[MAX_CONTACTS]()
        + _max_one[2 * NJOINT]()
        + _max_one[NV]()
        + 3 * NTENDON
    )


def je_elems[
    NV: Int, NJOINT: Int, NTENDON: Int, MAX_CONTACTS: Int, MAX_CONDIM: Int
]() -> Int:
    """Scalars in `Je` for ONE env: `ME * V_SIZE`."""
    return (
        je_edge_rows[NV, NJOINT, NTENDON, MAX_CONTACTS, MAX_CONDIM]()
        * _max_one[NV]()
    )


def je_spills[
    DTYPE: DType,
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Bool:
    """Does `Je` exceed the threadgroup budget and need the global buffer?"""
    return (
        je_elems[NV, NJOINT, NTENDON, MAX_CONTACTS, MAX_CONDIM]()
        * size_of[Scalar[DTYPE]]()
    ) > JE_SHARED_BUDGET


def je_ws_size[
    DTYPE: DType,
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Int:
    """Per-env spill-buffer size: `ME*NV` when spilling, else 0.

    `ContactScratch` allocates `BATCH * max(this, 1)`; a model that does not
    spill pays one scalar per env, not a buffer.
    """
    comptime if je_spills[
        DTYPE, NV, NJOINT, NTENDON, MAX_CONTACTS, MAX_CONDIM
    ]():
        return je_elems[NV, NJOINT, NTENDON, MAX_CONTACTS, MAX_CONDIM]()
    return 0
