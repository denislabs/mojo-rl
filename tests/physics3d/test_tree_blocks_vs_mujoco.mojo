"""Our mass-matrix BLOCK PARTITION is MuJoCo's. P1 of the block-diagonal work.

WHY THIS EXISTS
===============
`docs/BLOCK_DIAGONAL_MASS_MATRIX_PLAN.md` turns on one structural claim: M
couples a dof only with its tree ANCESTORS, so M's diagonal blocks are exactly
the kinematic trees and every entry outside a block is zero. A block-restricted
LDL is correct if and only if that partition is right, and a partition that is
wrong is wrong SILENTLY — the factorisation still runs, still returns numbers,
and drops whatever coupling it did not look at.

⚠⚠ THE ORACLE IS MUJOCO, NOT US. Comparing a new partition against our own
mass matrix would pass on a model where both are wrong — the shape recorded as
`feedback_a_gate_that_shares_its_reference_implementation_is_blind`, where two
parsers shared one wrong default.

TWO ARMS, GATED DIFFERENTLY, AND THE ASYMMETRY IS THE DESIGN
============================================================
1. **`dof_adr` / `dof_num`, EXACTLY.** Against `mjModel.tree_dofadr` /
   `tree_dofnum`. A disagreement in either direction is a failure.

2. **`kind`, ONE-DIRECTIONALLY.** Calling a block COMPACT when M is not
   diagonal discards real coupling: a silent wrong answer. Calling it DENSE
   when M is diagonal is only slower. So:

       ours COMPACT, MuJoCo DENSE    -> FAIL
       ours DENSE,  MuJoCo COMPACT   -> pass, COUNTED and printed

   The second is expected and not a defect: our classifier deliberately
   demands a SINGLE-BODY tree, while MuJoCo's `body_simple`
   (`user_model.cc:2814`) also admits a body whose parent is a fixed child of
   world. See `fields_build`'s `kind` comment for why we decline to guess
   there. The count is printed so a REGRESSION in conservatism — a block that
   used to be compact and stopped — is visible rather than silent.

⚠ MuJoCo's own compact column has an artifact baked in: `dof_simplenum`
(`user_model.cc:4100`) is a contiguous-suffix run-length counted from `nv-1`
down, so `M_rownnz` depends on where the simple bodies sit in the body list.
That is another reason the comparison is one-directional rather than exact.

⚠ VACUITY IS THE DEFAULT FAILURE. The tail prints models-COMPARED and
blocks-COMPARED beside the mismatch counts: "0 mismatches" over 0 blocks looks
exactly like a pass.

Regenerate the golden: pixi run python scripts/dump_mujoco_trees.py
Run: pixi run mojo run -I . tests/physics3d/test_tree_blocks_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime,
    dims_from_flat,
    build_model_runtime,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_TREE_SIZE,
    TREE_IDX_DOF_ADR,
    TREE_IDX_DOF_NUM,
    TREE_IDX_KIND,
    TREE_KIND_COMPACT,
    MODEL_META_IDX_NTREE,
)
from tests.physics3d.tree_block_goldens import (
    blk_case_count,
    blk_path,
    blk_nv,
    blk_nC,
    blk,
)

comptime DT = DType.float64


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _split(s: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var c = String(s[byte = i : i + 1])
        if c == " ":
            if cur.byte_length() > 0:
                out.append(cur)
            cur = String("")
        else:
            cur += c
    if cur.byte_length() > 0:
        out.append(cur)
    return out^


def _int(s: String) raises -> Int:
    return Int(atol(s))


def main() raises:
    var t = Tally()
    print("=== M's diagonal blocks vs MuJoCo 3.10.0 ===")

    var models_compared = 0
    var blocks_compared = 0
    var blocks_differing = 0
    var conservative = 0  # ours DENSE where MuJoCo says COMPACT — allowed
    var dense_total = 0
    var compact_ref = 0  # blocks MuJoCo calls COMPACT

    for c in range(blk_case_count()):
        var path = blk_path(c)
        print("---", path, "---")
        var fmd = parse_model_runtime(path)
        # ⚠ `nmesh_verts` IS A BUDGET, NOT A MODEL PROPERTY, and it cannot be
        # derived before the meshes load (`dims_from_flat`'s docstring). A
        # mesh-free model pays nothing at 0; a mesh model raises with the
        # number it needs, so try the cheap size and fall back to a generous
        # one rather than paying 262k vertices on `reacher`.
        var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
        var m = Model[DT, DynDims](dims)
        var built = True
        try:
            build_model_runtime[DT](fmd, dims, m)
        except:
            built = False
        if not built:
            dims = dims_from_flat(
                fmd, max_contacts=16, nmesh_verts=262144, nmesh_tri=0
            )
            m = Model[DT, DynDims](dims)
            build_model_runtime[DT](fmd, dims, m)

        var nv = dims.get_nv()
        t.truth(nv == blk_nv(c), String("nv ", nv, " (MuJoCo ", blk_nv(c), ")"))

        var want = _split(blk(c))
        var want_n = len(want) // 3
        var got_n = Int(m.meta.data[MODEL_META_IDX_NTREE])
        t.truth(
            got_n == want_n,
            String("ntree ", got_n, " (MuJoCo ", want_n, ")"),
        )
        if got_n != want_n:
            continue

        models_compared += 1
        var bad = 0
        var bad_kind = 0
        for b in range(want_n):
            var w_adr = _int(want[b * 3 + 0])
            var w_num = _int(want[b * 3 + 1])
            var w_cpt = _int(want[b * 3 + 2])
            var o = b * MODEL_TREE_SIZE
            var g_adr = Int(m.trees.data[o + TREE_IDX_DOF_ADR])
            var g_num = Int(m.trees.data[o + TREE_IDX_DOF_NUM])
            var g_cpt = (
                1 if Int(m.trees.data[o + TREE_IDX_KIND]) == TREE_KIND_COMPACT
                else 0
            )
            blocks_compared += 1

            if g_adr != w_adr or g_num != w_num:
                bad += 1
                blocks_differing += 1
                if bad <= 4:
                    print(
                        "       block", b, ": (", g_adr, g_num,
                        ") but MuJoCo says (", w_adr, w_num, ")",
                    )
            # ⚠ THE ONE-DIRECTIONAL ARM. Only ours-COMPACT-theirs-DENSE fails.
            if g_cpt == 1 and w_cpt == 0:
                bad_kind += 1
                blocks_differing += 1
                if bad_kind <= 4:
                    print(
                        "       block", b, "dofs[", w_adr, "..",
                        w_adr + w_num - 1,
                        "] : we say COMPACT, MuJoCo says M is NOT diagonal",
                    )
            if w_cpt == 1:
                compact_ref += 1
            if g_cpt == 0:
                dense_total += 1
                if w_cpt == 1:
                    conservative += 1

        t.truth(
            bad == 0,
            String(want_n, " block ranges match (", bad, " wrong)"),
        )
        t.truth(
            bad_kind == 0,
            String(
                want_n, " block kinds are sound (", bad_kind,
                " wrongly COMPACT)",
            ),
        )

    # ⚠ NON-VACUITY. Every arm above is trivially true over an empty golden, an
    # empty split, or a model list that failed to parse.
    print("--- the comparison was not empty ---")
    t.truth(
        models_compared >= 20,
        String("models compared: ", models_compared, " (of ",
               blk_case_count(), ")"),
    )
    t.truth(
        blocks_compared >= 50,
        String("blocks compared: ", blocks_compared, ", differing: ",
               blocks_differing),
    )
    # ⚠⚠ AN ARM, NOT A PRINT — because "conservative" and "broken" look
    # identical from the pass/fail side. A classifier that returns DENSE
    # unconditionally is sound by the arm above and gains nothing, and that is
    # exactly what shipped on the first attempt: the table was built 1,300
    # lines before `BODY_IDX_IQUAT_*` was written, every body read as
    # `iquat == 0`, and all 57 blocks came back DENSE with every other arm
    # green. Demand that we agree with MuJoCo on MOST of what it calls
    # compact, and print the shortfall.
    t.truth(
        conservative * 4 <= compact_ref,
        String("blocks MuJoCo calls COMPACT: ", compact_ref,
               "; we agree on ", compact_ref - conservative,
               ", conservatively decline ", conservative),
    )
    print(
        "  note: declining is allowed (slower, never wrong) — see the",
        "single-body rule in fields_build. DENSE blocks total:", dense_total,
    )

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_tree_blocks_vs_mujoco: " + String(t.fails) + " failed"
        )
