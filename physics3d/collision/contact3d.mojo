"""3D Contact data structures.

Stores collision information for constraint solving.
"""

from math3d import Vec3


struct Contact3D(Copyable, Movable):
    """Single 3D contact point between two bodies."""

    var body_a: Int  # Index of first body
    var body_b: Int  # Index of second body (-1 for static/ground)
    var point: Vec3  # World-space contact point
    var normal: Vec3  # Contact normal (from A to B)
    var depth: Float64  # Penetration depth (positive = penetrating)

    # Cached impulses for warm starting
    var impulse_n: Float64  # Normal impulse
    var impulse_t1: Float64  # Tangent impulse 1
    var impulse_t2: Float64  # Tangent impulse 2

    # Tangent basis for friction
    var tangent1: Vec3
    var tangent2: Vec3

    fn __init__(out self):
        """Initialize empty contact."""
        self.body_a = -1
        self.body_b = -1
        self.point = Vec3.zero()
        self.normal = Vec3.unit_z()
        self.depth = 0.0
        self.impulse_n = 0.0
        self.impulse_t1 = 0.0
        self.impulse_t2 = 0.0
        self.tangent1 = Vec3.unit_x()
        self.tangent2 = Vec3.unit_y()

    fn __init__(
        out self,
        body_a: Int,
        body_b: Int,
        point: Vec3,
        normal: Vec3,
        depth: Float64,
    ):
        """Initialize contact with given properties."""
        self.body_a = body_a
        self.body_b = body_b
        self.point = point
        self.normal = normal.normalized()
        self.depth = depth
        self.impulse_n = 0.0
        self.impulse_t1 = 0.0
        self.impulse_t2 = 0.0

        # Initialize tangent vectors before computing basis
        self.tangent1 = Vec3.unit_x()
        self.tangent2 = Vec3.unit_y()

        # Compute tangent basis orthogonal to normal
        self._compute_tangent_basis()

    fn _compute_tangent_basis(mut self):
        """Compute tangent vectors orthogonal to normal for friction."""
        # Choose a vector not parallel to normal
        var up = Vec3(0.0, 0.0, 1.0)
        if abs(self.normal.dot(up)) > 0.99:
            up = Vec3(1.0, 0.0, 0.0)

        self.tangent1 = self.normal.cross(up).normalized()
        self.tangent2 = self.normal.cross(self.tangent1).normalized()

    fn is_valid(self) -> Bool:
        """Check if contact is valid (positive depth, valid bodies)."""
        return self.body_a >= 0 and self.depth > 0.0


struct ContactManifold(Movable):
    """Collection of contact points between two bodies or body/ground.

    A manifold stores up to MAX_POINTS contacts for the same body pair.
    This allows for more stable contact solving (e.g., box on ground).
    """

    comptime MAX_POINTS: Int = 4

    var body_a: Int
    var body_b: Int
    var contacts: List[Contact3D]

    fn __init__(out self, body_a: Int, body_b: Int):
        """Initialize empty manifold for body pair."""
        self.body_a = body_a
        self.body_b = body_b
        self.contacts = List[Contact3D]()

    fn add_contact(mut self, var contact: Contact3D):
        """Add a contact point to the manifold.

        If manifold is full, replaces the contact with smallest depth.
        """
        if len(self.contacts) < Self.MAX_POINTS:
            self.contacts.append(contact^)
        else:
            # Find contact with smallest depth and replace if new one is deeper
            var min_depth = contact.depth
            var min_idx = -1

            for i in range(len(self.contacts)):
                if self.contacts[i].depth < min_depth:
                    min_depth = self.contacts[i].depth
                    min_idx = i

            if min_idx >= 0:
                self.contacts[min_idx] = contact^

    fn clear(mut self):
        """Clear all contacts."""
        self.contacts.clear()

    fn get_deepest(self) -> Contact3D:
        """Get the contact with maximum penetration depth."""
        if len(self.contacts) == 0:
            return Contact3D()

        var deepest_idx = 0
        var max_depth = self.contacts[0].depth

        for i in range(1, len(self.contacts)):
            if self.contacts[i].depth > max_depth:
                max_depth = self.contacts[i].depth
                deepest_idx = i

        return self.contacts[deepest_idx]

    fn get_average_normal(self) -> Vec3:
        """Get average contact normal across all contacts."""
        if len(self.contacts) == 0:
            return Vec3.unit_z()

        var avg = Vec3.zero()
        for i in range(len(self.contacts)):
            avg += self.contacts[i].normal

        return avg.normalized()
