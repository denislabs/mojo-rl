"""Tests for solver island detection.

Tests the union-find island detection algorithm in isolation using manually
constructed ConstraintData with known Jacobian sparsity patterns.

Test cases:
1. No constraints → 0 islands.
2. All constraints share at least one DOF → 1 island.
3. Two independent groups of DOFs and rows → 2 islands.
4. Three independent groups → 3 islands.
5. Chain coupling: A-B, B-C (B bridges A and C) → 1 island.
6. Unconstrained DOFs have dof_island = -1.
7. Negative J values count as non-zero couplings.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_island_detection.mojo
"""

from mojo_rl.physics3d.constraints.constraint_data import (
    ConstraintData,
    ConstraintRow,
)
from mojo_rl.physics3d.solver.island_detection import (
    detect_islands,
    IslandData,
    MAX_ISLANDS,
)
from std.testing import assert_true


# ---------------------------------------------------------------------------
# Helper: set one Jacobian entry in-place
# ---------------------------------------------------------------------------
fn set_j[
    DTYPE: DType, MAX_ROWS: Int, NV: Int
](
    mut cd: ConstraintData[DTYPE, MAX_ROWS, NV],
    row: Int,
    dof: Int,
    val: Float64,
):
    """Set cd.J[row*NV + dof] = val."""
    cd.J[row * NV + dof] = Scalar[DTYPE](val)


# ---------------------------------------------------------------------------
# Test 1: no constraints → 0 islands
# ---------------------------------------------------------------------------
fn test_no_constraints() raises:
    print("Test 1: no constraints → 0 islands ... ", end="")
    var cd = ConstraintData[DType.float64, 4, 4]()
    # num_rows defaults to 0
    var islands = detect_islands[DType.float64, 4, 4](cd)
    if islands.num_islands != 0:
        print("FAIL (got", islands.num_islands, "islands)")
        assert_true(
            False,
            "No constraints test failed: expected 0 islands, got "
            + String(islands.num_islands),
        )
    print("PASS")


# ---------------------------------------------------------------------------
# Test 2: single fully coupled system → 1 island
# ---------------------------------------------------------------------------
fn test_single_island() raises:
    print("Test 2: all rows share DOFs → 1 island ... ", end="")
    # NV=4, 3 rows each touching DOFs {0,1}, {1,2}, {2,3} — all connected
    var cd = ConstraintData[DType.float64, 8, 4]()
    cd.num_rows = 3
    cd.num_normals = 3
    set_j[DType.float64, 8, 4](cd, 0, 0, 1.0)
    set_j[DType.float64, 8, 4](cd, 0, 1, 1.0)
    set_j[DType.float64, 8, 4](cd, 1, 1, 1.0)
    set_j[DType.float64, 8, 4](cd, 1, 2, 1.0)
    set_j[DType.float64, 8, 4](cd, 2, 2, 1.0)
    set_j[DType.float64, 8, 4](cd, 2, 3, 1.0)

    var islands = detect_islands[DType.float64, 8, 4](cd)
    if islands.num_islands != 1:
        print("FAIL (got", islands.num_islands, "islands)")
        assert_true(
            False,
            "Single island test failed: expected 1 island, got "
            + String(islands.num_islands),
        )
    # All rows must be in island 0
    for r in range(3):
        if islands.row_island[r] != 0:
            print("FAIL: row", r, "in island", islands.row_island[r])
            assert_true(
                False,
                "Single island test failed: row "
                + String(r)
                + " not in island 0",
            )
    print("PASS")


# ---------------------------------------------------------------------------
# Test 3: two disjoint groups → 2 islands
# ---------------------------------------------------------------------------
fn test_two_islands() raises:
    print("Test 3: two independent DOF groups → 2 islands ... ", end="")
    # NV=6: DOFs 0,1,2 form island A; DOFs 3,4,5 form island B
    # Rows 0,1 touch only {0,1,2}; rows 2,3 touch only {3,4,5}
    var cd = ConstraintData[DType.float64, 8, 6]()
    cd.num_rows = 4
    cd.num_normals = 4
    # Island A rows
    set_j[DType.float64, 8, 6](cd, 0, 0, 1.0)
    set_j[DType.float64, 8, 6](cd, 0, 1, 1.0)
    set_j[DType.float64, 8, 6](cd, 1, 1, 1.0)
    set_j[DType.float64, 8, 6](cd, 1, 2, 1.0)
    # Island B rows
    set_j[DType.float64, 8, 6](cd, 2, 3, 1.0)
    set_j[DType.float64, 8, 6](cd, 2, 4, 1.0)
    set_j[DType.float64, 8, 6](cd, 3, 4, 1.0)
    set_j[DType.float64, 8, 6](cd, 3, 5, 1.0)

    var islands = detect_islands[DType.float64, 8, 6](cd)
    if islands.num_islands != 2:
        print("FAIL (got", islands.num_islands, "islands)")
        assert_true(
            False,
            "Two islands test failed: expected 2 islands, got "
            + String(islands.num_islands),
        )

    # Rows 0,1 must share an island; rows 2,3 must share a different island
    var ia = islands.row_island[0]
    var ib = islands.row_island[2]
    if ia < 0 or ib < 0:
        print("FAIL: negative island id")
        assert_true(False, "Two islands test failed: negative island id")
    if ia == ib:
        print("FAIL: both groups in same island", ia)
        assert_true(
            False,
            "Two islands test failed: both groups in same island " + String(ia),
        )
    if islands.row_island[1] != ia:
        print("FAIL: row 1 not in island A")
        assert_true(False, "Two islands test failed: row 1 not in island A")
    if islands.row_island[3] != ib:
        print("FAIL: row 3 not in island B")
        assert_true(False, "Two islands test failed: row 3 not in island B")
    # DOF counts
    if islands.island_num_dofs[ia] != 3:
        print(
            "FAIL: island A has", islands.island_num_dofs[ia], "DOFs (want 3)"
        )
        assert_true(
            False, "Two islands test failed: island A has wrong DOF count"
        )
    if islands.island_num_dofs[ib] != 3:
        print(
            "FAIL: island B has", islands.island_num_dofs[ib], "DOFs (want 3)"
        )
        assert_true(
            False, "Two islands test failed: island B has wrong DOF count"
        )
    # Row counts
    if islands.island_num_rows[ia] != 2:
        print(
            "FAIL: island A has", islands.island_num_rows[ia], "rows (want 2)"
        )
        assert_true(
            False, "Two islands test failed: island A has wrong row count"
        )
    if islands.island_num_rows[ib] != 2:
        print(
            "FAIL: island B has", islands.island_num_rows[ib], "rows (want 2)"
        )
        assert_true(
            False, "Two islands test failed: island B has wrong row count"
        )
    print("PASS")


# ---------------------------------------------------------------------------
# Test 4: three independent single-DOF constraints → 3 islands
# ---------------------------------------------------------------------------
fn test_three_islands() raises:
    print("Test 4: three independent constraints → 3 islands ... ", end="")
    # NV=6, 3 rows each touching one distinct DOF
    var cd = ConstraintData[DType.float64, 4, 6]()
    cd.num_rows = 3
    cd.num_normals = 3
    set_j[DType.float64, 4, 6](cd, 0, 0, 1.0)
    set_j[DType.float64, 4, 6](cd, 1, 2, 1.0)
    set_j[DType.float64, 4, 6](cd, 2, 5, 1.0)

    var islands = detect_islands[DType.float64, 4, 6](cd)
    if islands.num_islands != 3:
        print("FAIL (got", islands.num_islands, "islands)")
        assert_true(
            False,
            "Three islands test failed: expected 3 islands, got "
            + String(islands.num_islands),
        )
    # Each row is in a distinct island
    var i0 = islands.row_island[0]
    var i1 = islands.row_island[1]
    var i2 = islands.row_island[2]
    if i0 < 0 or i1 < 0 or i2 < 0:
        print("FAIL: negative island id")
        assert_true(False, "Three islands test failed: negative island id")
    if i0 == i1 or i0 == i2 or i1 == i2:
        print("FAIL: rows share islands unexpectedly:", i0, i1, i2)
        assert_true(
            False, "Three islands test failed: rows share islands unexpectedly"
        )
    print("PASS")


# ---------------------------------------------------------------------------
# Test 5: chain coupling A-B, B-C merges into one island
# ---------------------------------------------------------------------------
fn test_chain_merges() raises:
    print("Test 5: chain A-B, B-C bridges into 1 island ... ", end="")
    # Row 0 couples DOFs 0 and 1 (A-B edge)
    # Row 1 couples DOFs 1 and 2 (B-C edge)
    # Despite DOF 0 and DOF 2 never appearing in the same row,
    # they must end up in the same island (transitivity via DOF 1).
    var cd = ConstraintData[DType.float64, 4, 4]()
    cd.num_rows = 2
    cd.num_normals = 2
    set_j[DType.float64, 4, 4](cd, 0, 0, 1.0)
    set_j[DType.float64, 4, 4](cd, 0, 1, 1.0)
    set_j[DType.float64, 4, 4](cd, 1, 1, 1.0)
    set_j[DType.float64, 4, 4](cd, 1, 2, 1.0)

    var islands = detect_islands[DType.float64, 4, 4](cd)
    if islands.num_islands != 1:
        print("FAIL (got", islands.num_islands, "islands)")
        assert_true(
            False,
            "Chain merges test failed: expected 1 island, got "
            + String(islands.num_islands),
        )
    if islands.row_island[0] != islands.row_island[1]:
        print("FAIL: rows 0 and 1 in different islands")
        assert_true(
            False, "Chain merges test failed: rows 0 and 1 in different islands"
        )
    print("PASS")


# ---------------------------------------------------------------------------
# Test 6: unconstrained DOFs have dof_island == -1
# ---------------------------------------------------------------------------
fn test_unconstrained_dofs() raises:
    print("Test 6: unconstrained DOFs have dof_island = -1 ... ", end="")
    # NV=4, only DOFs 0 and 1 touched; DOFs 2 and 3 are free.
    var cd = ConstraintData[DType.float64, 4, 4]()
    cd.num_rows = 1
    cd.num_normals = 1
    set_j[DType.float64, 4, 4](cd, 0, 0, 1.0)
    set_j[DType.float64, 4, 4](cd, 0, 1, 1.0)

    var islands = detect_islands[DType.float64, 4, 4](cd)
    if islands.num_islands != 1:
        print("FAIL: expected 1 island, got", islands.num_islands)
        assert_true(
            False,
            "Unconstrained DOFs test failed: expected 1 island, got "
            + String(islands.num_islands),
        )
    if islands.dof_island[0] < 0 or islands.dof_island[1] < 0:
        print("FAIL: touched DOFs have negative island id")
        assert_true(
            False,
            (
                "Unconstrained DOFs test failed: touched DOFs have negative"
                " island id"
            ),
        )
    if islands.dof_island[2] != -1 or islands.dof_island[3] != -1:
        print(
            "FAIL: free DOFs should be -1, got",
            islands.dof_island[2],
            islands.dof_island[3],
        )
        assert_true(
            False,
            (
                "Unconstrained DOFs test failed: free DOFs should have"
                " dof_island == -1"
            ),
        )
    print("PASS")


# ---------------------------------------------------------------------------
# Test 7: negative J values are also recognised as non-zero
# ---------------------------------------------------------------------------
fn test_negative_j() raises:
    print("Test 7: negative J entries count as non-zero couplings ... ", end="")
    # Row 0: DOF 0 = +1, DOF 1 = -1  →  DOFs 0,1 in same island
    var cd = ConstraintData[DType.float64, 4, 4]()
    cd.num_rows = 1
    cd.num_normals = 1
    set_j[DType.float64, 4, 4](cd, 0, 0, 1.0)
    set_j[DType.float64, 4, 4](cd, 0, 1, -1.0)

    var islands = detect_islands[DType.float64, 4, 4](cd)
    if islands.num_islands != 1:
        print("FAIL: expected 1 island, got", islands.num_islands)
        assert_true(
            False,
            "Negative J test failed: expected 1 island, got "
            + String(islands.num_islands),
        )
    if islands.dof_island[0] != islands.dof_island[1]:
        print("FAIL: DOFs 0 and 1 in different islands")
        assert_true(
            False,
            "Negative J test failed: DOFs 0 and 1 should be in the same island",
        )
    print("PASS")


fn main() raises:
    # TestSuite.discover_tests[__functions_in_module()]().run()
    test_no_constraints()
    test_single_island()
    test_two_islands()
    test_three_islands()
    test_chain_merges()
    test_unconstrained_dofs()
    test_negative_j()
    print("All island detection tests passed.")
