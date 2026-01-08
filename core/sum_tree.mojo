"""Sum-Tree data structure for Prioritized Experience Replay.

A sum-tree is a complete binary tree where:
- Leaf nodes store priorities
- Internal nodes store the sum of their children
- Root stores the total sum of all priorities

This enables O(log n) sampling proportional to priority.

Reference: Schaul et al., "Prioritized Experience Replay" (2015)
"""


struct SumTree(Movable):
    """Binary sum-tree for efficient priority-based sampling.

    The tree is stored in a flat array where:
    - Index 0 is the root
    - For node i: left child = 2*i + 1, right child = 2*i + 2
    - Leaf nodes start at index (capacity - 1)

    Time complexity:
    - Update priority: O(log n)
    - Sample: O(log n)
    - Get total sum: O(1)

    Usage:
        var tree = SumTree(capacity=1000)
        tree.update(idx=0, priority=1.5)  # Set priority for leaf 0
        var idx = tree.sample(target=0.5)  # Sample proportionally
    """

    var tree: List[Float64]  # Internal tree nodes + leaves
    var capacity: Int  # Number of leaf nodes
    var write_ptr: Int  # Next write position in leaves
    var size: Int  # Current number of valid entries

    fn __init__(out self, capacity: Int):
        """Initialize sum-tree with given leaf capacity.

        Args:
            capacity: Maximum number of priorities to store (leaf nodes).
        """
        self.capacity = capacity
        self.write_ptr = 0
        self.size = 0

        # Total tree size: 2 * capacity - 1 (complete binary tree)
        var tree_size = 2 * capacity - 1
        self.tree = List[Float64]()
        for _ in range(tree_size):
            self.tree.append(0.0)

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.tree = existing.tree^
        self.capacity = existing.capacity
        self.write_ptr = existing.write_ptr
        self.size = existing.size

    fn _propagate(mut self, idx: Int, change: Float64):
        """Propagate priority change up to root.

        Args:
            idx: Tree index where change occurred
            change: Amount of change (new - old priority)
        """
        var parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent > 0:
            self._propagate(parent, change)

    fn _leaf_to_tree_idx(self, leaf_idx: Int) -> Int:
        """Convert leaf index to tree array index.

        Args:
            leaf_idx: Index in leaf space (0 to capacity-1)

        Returns:
            Index in tree array
        """
        return leaf_idx + self.capacity - 1

    fn _tree_to_leaf_idx(self, tree_idx: Int) -> Int:
        """Convert tree array index to leaf index.

        Args:
            tree_idx: Index in tree array

        Returns:
            Index in leaf space (0 to capacity-1)
        """
        return tree_idx - self.capacity + 1

    fn update(mut self, leaf_idx: Int, priority: Float64):
        """Update priority at leaf index.

        Args:
            leaf_idx: Leaf index (0 to capacity-1)
            priority: New priority value (must be positive)
        """
        var tree_idx = self._leaf_to_tree_idx(leaf_idx)
        var change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up to root
        if tree_idx > 0:
            self._propagate(tree_idx, change)

    fn add(mut self, priority: Float64) -> Int:
        """Add a new priority and return its leaf index.

        Uses circular buffer - overwrites oldest entry when full.

        Args:
            priority: Priority value for new entry

        Returns:
            Leaf index where priority was stored
        """
        var leaf_idx = self.write_ptr
        self.update(leaf_idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

        return leaf_idx

    fn get(self, leaf_idx: Int) -> Float64:
        """Get priority at leaf index.

        Args:
            leaf_idx: Leaf index (0 to capacity-1)

        Returns:
            Priority value at that index
        """
        var tree_idx = self._leaf_to_tree_idx(leaf_idx)
        return self.tree[tree_idx]

    fn sample(self, target: Float64) -> Int:
        """Sample a leaf index proportional to priorities.

        Traverses tree from root, going left if target <= left child sum,
        otherwise going right (subtracting left sum from target).

        Args:
            target: Random value in [0, total_sum)

        Returns:
            Leaf index selected proportionally to priorities
        """
        var idx = 0  # Start at root
        var remaining = target

        while True:
            var left = 2 * idx + 1
            var right = 2 * idx + 2

            # If we've reached a leaf node
            if left >= len(self.tree):
                break

            # Go left or right based on remaining target
            if remaining <= self.tree[left]:
                idx = left
            else:
                remaining -= self.tree[left]
                idx = right

        return self._tree_to_leaf_idx(idx)

    fn total_sum(self) -> Float64:
        """Get total sum of all priorities (root value).

        Returns:
            Sum of all priorities in the tree
        """
        return self.tree[0]

    fn max_priority(self) -> Float64:
        """Get maximum priority among all leaves.

        Returns:
            Maximum priority value
        """
        var max_p: Float64 = 0.0
        for i in range(self.size):
            var tree_idx = self._leaf_to_tree_idx(i)
            if self.tree[tree_idx] > max_p:
                max_p = self.tree[tree_idx]
        return max_p

    fn min_priority(self) -> Float64:
        """Get minimum non-zero priority among all leaves.

        Returns:
            Minimum priority value (or large value if empty)
        """
        var min_p: Float64 = 1e10
        for i in range(self.size):
            var tree_idx = self._leaf_to_tree_idx(i)
            var p = self.tree[tree_idx]
            if p > 0 and p < min_p:
                min_p = p
        return min_p if min_p < 1e10 else 1.0

    fn len(self) -> Int:
        """Return number of valid entries."""
        return self.size
