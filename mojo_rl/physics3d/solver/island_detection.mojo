"""Solver island detection for the physics3d constraint system.

An 'island' is a maximal connected subgraph of the constraint-DOF coupling
graph: two DOFs belong to the same island if they are both touched by at
least one common constraint row (|J[r,d]| > ISLAND_J_THRESH).  Independent
islands can be solved in parallel and can converge at different rates,
enabling per-island early termination in iterative solvers (PGS, CG, Newton).

Algorithm
---------
Union-Find (path-halving) over DOFs.  For every constraint row r we union
all DOFs d with |J[r,d]| > ISLAND_J_THRESH into one component.  After
processing all rows we assign sequential island IDs 0..nisland-1 to
non-trivial components (those that participate in at least one row).
Unconstrained DOFs get dof_island = -1.

Complexity: O(MAX_ROWS * NV * α(NV)) ≈ O(MAX_ROWS * NV).

Usage
-----
    from physics3d.solver.island_detection import detect_islands, IslandData
    var islands = detect_islands(constraints)
    # islands.num_islands  — number of independent sub-problems
    # islands.dof_island[d]  — island id for DOF d (-1 if unconstrained)
    # islands.row_island[r]  — island id for constraint row r
"""

from ..types import _max_one
from ..constraints.constraint_data import ConstraintData

# Jacobian entry magnitude threshold below which an entry is considered zero
comptime ISLAND_J_THRESH: Float64 = 1e-15

# Soft cap on number of tracked islands.  If more connected components
# are found, the excess are merged into the last bucket.
# Increase this if you have more than 64 independent sub-problems.
comptime MAX_ISLANDS: Int = 64


struct IslandData[MAX_ROWS: Int, NV: Int](Movable):
    """Per-DOF and per-row island membership produced by detect_islands().

    Fields:
        num_islands:      Total number of islands found (0 if no constraints).
        dof_island:       Island ID for each DOF;  -1 = unconstrained DOF.
        row_island:       Island ID for each constraint row; -1 = row absent.
        island_num_rows:  Number of constraint rows in each island.
        island_num_dofs:  Number of constrained DOFs in each island.
    """

    var num_islands: Int
    var dof_island: InlineArray[Int, _max_one[Self.NV]()]
    var row_island: InlineArray[Int, _max_one[Self.MAX_ROWS]()]
    var island_num_rows: InlineArray[Int, MAX_ISLANDS]
    var island_num_dofs: InlineArray[Int, MAX_ISLANDS]

    fn __init__(out self):
        self.num_islands = 0
        self.dof_island = InlineArray[Int, _max_one[Self.NV]()](fill=-1)
        self.row_island = InlineArray[Int, _max_one[Self.MAX_ROWS]()](fill=-1)
        self.island_num_rows = InlineArray[Int, MAX_ISLANDS](fill=0)
        self.island_num_dofs = InlineArray[Int, MAX_ISLANDS](fill=0)


fn detect_islands[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
) -> IslandData[
    MAX_ROWS, NV
]:
    """Detect constraint islands via union-find on the DOF coupling graph.

    Two DOFs are coupled (same island) if they both appear as non-zero
    columns in at least one common row of the constraint Jacobian J.

    Returns IslandData mapping each DOF and constraint row to an island ID.
    DOFs that appear in no constraint row have dof_island = -1.

    Args:
        constraints: Pre-built ConstraintData from the constraint builder.

    Returns:
        IslandData with island membership for all DOFs and rows.
    """
    var result = IslandData[MAX_ROWS, NV]()
    var num_rows = constraints.num_rows

    if num_rows == 0:
        return result^

    # ---- Union-Find: initialise each DOF as its own root ----
    var parent = InlineArray[Int, _max_one[NV]()](uninitialized=True)
    # touched[d] = 1 when DOF d appears in at least one constraint row
    var touched = InlineArray[Int, _max_one[NV]()](fill=0)
    for d in range(NV):
        parent[d] = d

    # ---- Build coupling graph: union DOFs that share a constraint row ----
    var j_pos = Scalar[DTYPE](ISLAND_J_THRESH)
    var j_neg = -j_pos

    for r in range(num_rows):
        var first_dof = -1
        for d in range(NV):
            var j_val = constraints.J[r * NV + d]
            if j_val > j_pos or j_val < j_neg:
                touched[d] = 1
                if first_dof < 0:
                    first_dof = d
                else:
                    # Path-halving find for first_dof
                    # Read inner index into a temp to avoid aliasing on parent[parent[x]]
                    var ra = first_dof
                    while parent[ra] != ra:
                        var inner = parent[ra]
                        var gp = parent[inner]
                        parent[ra] = gp
                        ra = gp
                    # Path-halving find for d
                    var rb = d
                    while parent[rb] != rb:
                        var inner = parent[rb]
                        var gp = parent[inner]
                        parent[rb] = gp
                        rb = gp
                    # Union: attach rb's tree under ra
                    if ra != rb:
                        parent[rb] = ra

    # ---- Assign sequential island IDs to distinct roots ----
    var root_island = InlineArray[Int, _max_one[NV]()](fill=-1)
    var num_islands = 0

    for d in range(NV):
        if touched[d] == 1:
            # Ordinary find (root is nearly flat after path-halving above)
            var root = d
            while parent[root] != root:
                root = parent[root]
            if root_island[root] < 0:
                var iid = num_islands
                if iid >= MAX_ISLANDS:
                    iid = MAX_ISLANDS - 1  # clamp excess into last bucket
                root_island[root] = iid
                num_islands += 1

    if num_islands > MAX_ISLANDS:
        num_islands = MAX_ISLANDS
    result.num_islands = num_islands

    # ---- Map each DOF to its island ID ----
    for d in range(NV):
        if touched[d] == 1:
            var root = d
            while parent[root] != root:
                root = parent[root]
            var iid = root_island[root]
            result.dof_island[d] = iid
            result.island_num_dofs[iid] += 1

    # ---- Map each constraint row to an island ID ----
    # (use the island of the first non-zero DOF in that row)
    for r in range(num_rows):
        for d in range(NV):
            var j_val = constraints.J[r * NV + d]
            if j_val > j_pos or j_val < j_neg:
                result.row_island[r] = result.dof_island[d]
                break
        var iid = result.row_island[r]
        if iid >= 0:
            result.island_num_rows[iid] += 1

    return result^
