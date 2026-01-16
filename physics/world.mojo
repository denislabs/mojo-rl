"""
World: Physics Simulation Manager

Central manager that owns all physics objects and runs the simulation.
Implements the Box2D-style simulation loop:
1. Apply forces and integrate velocities
2. Broad phase collision detection
3. Narrow phase collision detection
4. Solve velocity constraints
5. Integrate positions
6. Solve position constraints
7. Update sleep states
"""

from physics.vec2 import Vec2, vec2
from physics.shape import (
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_EDGE,
    PolygonShape,
    CircleShape,
    EdgeShape,
)
from physics.body import Body, Transform, MassData, BODY_STATIC, BODY_DYNAMIC
from physics.fixture import Fixture, Filter, AABB
from physics.collision import (
    Contact,
    ContactManifold,
    collide_edge_polygon,
    collide_edge_circle,
    collide_polygon_polygon,
    solve_contact_velocity,
    solve_contact_position,
)
from physics.joint import RevoluteJoint
from physics.raycast import RaycastResult, raycast_edge, raycast_polygon


# Default simulation parameters
comptime DEFAULT_VELOCITY_ITERATIONS: Int = 8
comptime DEFAULT_POSITION_ITERATIONS: Int = 3


struct ContactListener(Copyable, Movable):
    """Callback interface for contact events.

    Tracks fixture indices for contacts - actual callback handling
    is done by the environment querying begin_contacts/end_contacts.
    """

    var fixture_a_idx: Int  # Set before callback
    var fixture_b_idx: Int  # Set before callback
    var enabled: Bool

    fn __init__(out self):
        self.fixture_a_idx = -1
        self.fixture_b_idx = -1
        self.enabled = False

    fn __copyinit__(out self, other: Self):
        self.fixture_a_idx = other.fixture_a_idx
        self.fixture_b_idx = other.fixture_b_idx
        self.enabled = other.enabled

    fn __moveinit__(out self, deinit other: Self):
        self.fixture_a_idx = other.fixture_a_idx
        self.fixture_b_idx = other.fixture_b_idx
        self.enabled = other.enabled


struct World(Copyable, Movable):
    """Physics world simulation manager."""

    var gravity: Vec2

    # Storage
    var bodies: List[Body]
    var fixtures: List[Fixture]
    var joints: List[RevoluteJoint]
    var contacts: List[Contact]

    # Free lists for reuse
    var free_body_indices: List[Int]
    var free_fixture_indices: List[Int]

    # Contact listener (simplified - stores callbacks separately)
    var contact_listener: ContactListener

    # Begin/end contact tracking
    var begin_contacts: List[Tuple[Int, Int]]  # (fixture_a, fixture_b) pairs
    var end_contacts: List[Tuple[Int, Int]]

    fn __init__(out self, gravity: Vec2 = Vec2(0.0, -10.0)):
        """Create world with given gravity."""
        self.gravity = gravity
        self.bodies = List[Body]()
        self.fixtures = List[Fixture]()
        self.joints = List[RevoluteJoint]()
        self.contacts = List[Contact]()
        self.free_body_indices = List[Int]()
        self.free_fixture_indices = List[Int]()
        self.contact_listener = ContactListener()
        self.begin_contacts = List[Tuple[Int, Int]]()
        self.end_contacts = List[Tuple[Int, Int]]()

    # ===== Body Management =====

    fn create_body(
        mut self,
        body_type: Int,
        position: Vec2,
        angle: Float64 = 0.0,
    ) -> Int:
        """Create a new body and return its index."""
        var body = Body(body_type, position, angle)

        # Reuse freed index if available
        if len(self.free_body_indices) > 0:
            var idx = self.free_body_indices.pop()
            self.bodies[idx] = body^
            return idx

        # Append new body
        var idx = len(self.bodies)
        self.bodies.append(body^)
        return idx

    fn destroy_body(mut self, body_idx: Int):
        """Mark body as destroyed (will be reused)."""
        if body_idx < 0 or body_idx >= len(self.bodies):
            return

        # Clear body data
        self.bodies[body_idx] = Body()
        self.bodies[body_idx].body_type = -1  # Mark as destroyed

        # Add to free list
        self.free_body_indices.append(body_idx)

        # TODO: Also destroy associated fixtures and joints

    fn get_body_copy(self, idx: Int) -> Body:
        """Get body by index (returns copy)."""
        return self.bodies[idx].copy()

    fn get_body_ref(mut self, idx: Int) -> ref [self.bodies] Body:
        """Get mutable reference to body."""
        return self.bodies[idx]

    # ===== Fixture Management =====

    fn create_polygon_fixture(
        mut self,
        body_idx: Int,
        var polygon: PolygonShape,
        density: Float64 = 1.0,
        friction: Float64 = 0.2,
        restitution: Float64 = 0.0,
        filter: Filter = Filter(),
    ) -> Int:
        """Create polygon fixture attached to body."""
        var fixture = Fixture.from_polygon(
            body_idx, polygon^, density, friction, restitution, filter
        )

        var idx: Int
        if len(self.free_fixture_indices) > 0:
            idx = self.free_fixture_indices.pop()
            self.fixtures[idx] = fixture^
        else:
            idx = len(self.fixtures)
            self.fixtures.append(fixture^)

        # Update body's fixture range
        if self.bodies[body_idx].fixture_count == 0:
            self.bodies[body_idx].fixture_start = idx
        self.bodies[body_idx].fixture_count += 1

        # Update body mass
        self._update_body_mass(body_idx)

        return idx

    fn create_circle_fixture(
        mut self,
        body_idx: Int,
        var circle: CircleShape,
        density: Float64 = 1.0,
        friction: Float64 = 0.2,
        restitution: Float64 = 0.0,
        filter: Filter = Filter(),
    ) -> Int:
        """Create circle fixture attached to body."""
        var fixture = Fixture.from_circle(
            body_idx, circle^, density, friction, restitution, filter
        )

        var idx: Int
        if len(self.free_fixture_indices) > 0:
            idx = self.free_fixture_indices.pop()
            self.fixtures[idx] = fixture^
        else:
            idx = len(self.fixtures)
            self.fixtures.append(fixture^)

        # Update body's fixture range
        if self.bodies[body_idx].fixture_count == 0:
            self.bodies[body_idx].fixture_start = idx
        self.bodies[body_idx].fixture_count += 1

        # Update body mass
        self._update_body_mass(body_idx)

        return idx

    fn create_edge_fixture(
        mut self,
        body_idx: Int,
        var edge: EdgeShape,
        friction: Float64 = 0.1,
        restitution: Float64 = 0.0,
        filter: Filter = Filter(),
    ) -> Int:
        """Create edge fixture attached to body."""
        var fixture = Fixture.from_edge(
            body_idx, edge^, 0.0, friction, restitution, filter
        )

        var idx: Int
        if len(self.free_fixture_indices) > 0:
            idx = self.free_fixture_indices.pop()
            self.fixtures[idx] = fixture^
        else:
            idx = len(self.fixtures)
            self.fixtures.append(fixture^)

        # Update body's fixture range (edges don't affect mass)
        if self.bodies[body_idx].fixture_count == 0:
            self.bodies[body_idx].fixture_start = idx
        self.bodies[body_idx].fixture_count += 1

        return idx

    fn _update_body_mass(mut self, body_idx: Int):
        """Recompute body mass from fixtures."""
        if self.bodies[body_idx].body_type != BODY_DYNAMIC:
            return

        var total_mass: Float64 = 0.0
        var total_inertia: Float64 = 0.0
        var center = Vec2.zero()

        var start = self.bodies[body_idx].fixture_start
        var count = self.bodies[body_idx].fixture_count

        for i in range(count):
            var fix_idx = start + i
            if fix_idx >= len(self.fixtures):
                break
            var result = self.fixtures[fix_idx].compute_mass()
            var mass = result[0]
            var inertia = result[1]
            var c = result[2]
            total_mass += mass
            total_inertia += inertia
            center += c * mass

        if total_mass > 0.0:
            center = center / total_mass
            # Shift inertia to body center
            total_inertia -= total_mass * center.length_squared()

        self.bodies[body_idx].set_mass_data(total_mass, total_inertia, center)

    # ===== Joint Management =====

    fn create_revolute_joint(
        mut self,
        body_a_idx: Int,
        body_b_idx: Int,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        enable_motor: Bool = False,
        motor_speed: Float64 = 0.0,
        max_motor_torque: Float64 = 0.0,
        enable_limit: Bool = False,
        lower_angle: Float64 = 0.0,
        upper_angle: Float64 = 0.0,
    ) -> Int:
        """Create revolute joint and return its index."""
        # Compute reference angle from current body angles
        var ref_angle = self.bodies[body_b_idx].angle - self.bodies[body_a_idx].angle

        var joint = RevoluteJoint(
            body_a_idx,
            body_b_idx,
            local_anchor_a,
            local_anchor_b,
            ref_angle,
            enable_motor,
            motor_speed,
            max_motor_torque,
            enable_limit,
            lower_angle,
            upper_angle,
        )

        var idx = len(self.joints)
        self.joints.append(joint^)
        return idx

    # ===== Simulation Step =====

    fn step(
        mut self,
        dt: Float64,
        velocity_iterations: Int = DEFAULT_VELOCITY_ITERATIONS,
        position_iterations: Int = DEFAULT_POSITION_ITERATIONS,
    ):
        """Advance simulation by dt seconds."""
        # Clear contact tracking
        self.begin_contacts.clear()
        self.end_contacts.clear()

        # 1. Apply gravity and integrate velocities
        self._integrate_velocities(dt)

        # 2. Update fixture AABBs
        self._update_aabbs()

        # 3. Collision detection (broad + narrow phase)
        self._find_contacts()

        # 4. Initialize joint constraints
        for i in range(len(self.joints)):
            var body_a_idx = self.joints[i].body_a_idx
            var body_b_idx = self.joints[i].body_b_idx
            self.joints[i].init_velocity_constraints(
                self.bodies[body_a_idx], self.bodies[body_b_idx], dt
            )

        # 5. Solve velocity constraints
        for _ in range(velocity_iterations):
            # Solve joints
            self._solve_joints_velocity()

            # Solve contacts
            self._solve_contacts_velocity()

        # 6. Integrate positions
        self._integrate_positions(dt)

        # 7. Solve position constraints
        for _ in range(position_iterations):
            # Solve joints
            self._solve_joints_position()

            # Solve contacts
            self._solve_contacts_position()

        # 8. Update sleep states
        for i in range(len(self.bodies)):
            if self.bodies[i].body_type >= 0:  # Not destroyed
                self.bodies[i].update_sleep_state(dt)

    fn _integrate_velocities(mut self, dt: Float64):
        """Apply forces and integrate velocities."""
        for i in range(len(self.bodies)):
            if self.bodies[i].body_type == BODY_DYNAMIC and self.bodies[i].awake:
                self.bodies[i].integrate_velocities(self.gravity, dt)

    fn _integrate_positions(mut self, dt: Float64):
        """Integrate positions from velocities."""
        for i in range(len(self.bodies)):
            if self.bodies[i].body_type == BODY_DYNAMIC:
                self.bodies[i].integrate_positions(dt)

    fn _update_aabbs(mut self):
        """Update fixture AABBs."""
        for i in range(len(self.fixtures)):
            var body_idx = self.fixtures[i].body_idx
            if body_idx >= 0 and body_idx < len(self.bodies):
                self.fixtures[i].compute_aabb(self.bodies[body_idx].transform)

    fn _find_contacts(mut self):
        """Find all contacts between fixtures."""
        # Mark existing contacts as not touching
        for i in range(len(self.contacts)):
            self.contacts[i].touching = False

        # O(n^2) broad phase - check all pairs
        # For LunarLander's small number of bodies, this is fine
        for i in range(len(self.fixtures)):
            for j in range(i + 1, len(self.fixtures)):
                self._check_collision(i, j)

        # Remove non-touching contacts
        var new_contacts = List[Contact]()
        for i in range(len(self.contacts)):
            if self.contacts[i].touching:
                new_contacts.append(self.contacts[i].copy())
        self.contacts = new_contacts^

    fn _check_collision(mut self, fix_a_idx: Int, fix_b_idx: Int):
        """Check collision between two fixtures."""
        var fix_a = self.fixtures[fix_a_idx].copy()
        var fix_b = self.fixtures[fix_b_idx].copy()

        # Skip same body
        if fix_a.body_idx == fix_b.body_idx:
            return

        # Check filter
        if not fix_a.filter.should_collide(fix_b.filter):
            return

        # AABB overlap test
        if not fix_a.aabb.overlaps(fix_b.aabb):
            return

        var body_a = self.bodies[fix_a.body_idx].copy()
        var body_b = self.bodies[fix_b.body_idx].copy()

        # Narrow phase
        var manifold = ContactManifold()

        # Track which fixture is edge (for consistent ordering)
        var actual_fix_a_idx = fix_a_idx
        var actual_fix_b_idx = fix_b_idx

        # Edge-Polygon
        if fix_a.shape_type == SHAPE_EDGE and fix_b.shape_type == SHAPE_POLYGON:
            manifold = collide_edge_polygon(
                fix_a.edge, body_a.transform, fix_b.polygon, body_b.transform
            )
        elif fix_b.shape_type == SHAPE_EDGE and fix_a.shape_type == SHAPE_POLYGON:
            manifold = collide_edge_polygon(
                fix_b.edge, body_b.transform, fix_a.polygon, body_a.transform
            )
            # Swap fixture indices for consistent ordering
            actual_fix_a_idx = fix_b_idx
            actual_fix_b_idx = fix_a_idx

        # Edge-Circle
        elif fix_a.shape_type == SHAPE_EDGE and fix_b.shape_type == SHAPE_CIRCLE:
            manifold = collide_edge_circle(
                fix_a.edge, body_a.transform, fix_b.circle, body_b.transform
            )
        elif fix_b.shape_type == SHAPE_EDGE and fix_a.shape_type == SHAPE_CIRCLE:
            manifold = collide_edge_circle(
                fix_b.edge, body_b.transform, fix_a.circle, body_a.transform
            )
            actual_fix_a_idx = fix_b_idx
            actual_fix_b_idx = fix_a_idx

        # Polygon-Polygon
        elif fix_a.shape_type == SHAPE_POLYGON and fix_b.shape_type == SHAPE_POLYGON:
            manifold = collide_polygon_polygon(
                fix_a.polygon, body_a.transform, fix_b.polygon, body_b.transform
            )

        # If collision found, create or update contact
        if manifold.count > 0:
            # Set fixture indices
            manifold.fixture_a_idx = actual_fix_a_idx
            manifold.fixture_b_idx = actual_fix_b_idx

            # Find existing contact
            var found_idx = -1
            for i in range(len(self.contacts)):
                if (
                    self.contacts[i].manifold.fixture_a_idx == actual_fix_a_idx
                    and self.contacts[i].manifold.fixture_b_idx == actual_fix_b_idx
                ):
                    found_idx = i
                    break

            if found_idx >= 0:
                # Update existing contact
                self.contacts[found_idx].manifold = manifold^
                self.contacts[found_idx].touching = True
            else:
                # Create new contact
                var contact = Contact()
                contact.manifold = manifold^
                contact.body_a_idx = self.fixtures[actual_fix_a_idx].body_idx
                contact.body_b_idx = self.fixtures[actual_fix_b_idx].body_idx
                contact.touching = True
                contact.is_new = True
                self.contacts.append(contact^)

                # Track begin contact
                self.begin_contacts.append((actual_fix_a_idx, actual_fix_b_idx))

    fn _solve_joints_velocity(mut self):
        """Solve velocity constraints for all joints."""
        for i in range(len(self.joints)):
            var body_a_idx = self.joints[i].body_a_idx
            var body_b_idx = self.joints[i].body_b_idx
            # Copy bodies, solve, then copy back
            var body_a = self.bodies[body_a_idx].copy()
            var body_b = self.bodies[body_b_idx].copy()
            self.joints[i].solve_velocity_constraints(body_a, body_b)
            self.bodies[body_a_idx] = body_a^
            self.bodies[body_b_idx] = body_b^

    fn _solve_joints_position(mut self):
        """Solve position constraints for all joints."""
        for i in range(len(self.joints)):
            var body_a_idx = self.joints[i].body_a_idx
            var body_b_idx = self.joints[i].body_b_idx
            # Copy bodies, solve, then copy back
            var body_a = self.bodies[body_a_idx].copy()
            var body_b = self.bodies[body_b_idx].copy()
            _ = self.joints[i].solve_position_constraints(body_a, body_b)
            self.bodies[body_a_idx] = body_a^
            self.bodies[body_b_idx] = body_b^

    fn _solve_contacts_velocity(mut self):
        """Solve velocity constraints for all contacts."""
        for i in range(len(self.contacts)):
            if not self.contacts[i].touching:
                continue

            var fix_a_idx = self.contacts[i].manifold.fixture_a_idx
            var fix_b_idx = self.contacts[i].manifold.fixture_b_idx
            var restitution = max_f64(
                self.fixtures[fix_a_idx].restitution,
                self.fixtures[fix_b_idx].restitution,
            )

            var body_a_idx = self.contacts[i].body_a_idx
            var body_b_idx = self.contacts[i].body_b_idx
            var body_a = self.bodies[body_a_idx].copy()
            var body_b = self.bodies[body_b_idx].copy()

            for j in range(self.contacts[i].manifold.count):
                var point = self.contacts[i].manifold.points[j]
                _ = solve_contact_velocity(
                    body_a,
                    body_b,
                    point.point,
                    point.normal,
                    restitution,
                    self.contacts[i].normal_impulse,
                )

            self.bodies[body_a_idx] = body_a^
            self.bodies[body_b_idx] = body_b^

    fn _solve_contacts_position(mut self):
        """Solve position constraints for all contacts."""
        for i in range(len(self.contacts)):
            if not self.contacts[i].touching:
                continue

            var body_a_idx = self.contacts[i].body_a_idx
            var body_b_idx = self.contacts[i].body_b_idx
            var body_a = self.bodies[body_a_idx].copy()
            var body_b = self.bodies[body_b_idx].copy()

            for j in range(self.contacts[i].manifold.count):
                var point = self.contacts[i].manifold.points[j]
                _ = solve_contact_position(
                    body_a,
                    body_b,
                    point.point,
                    point.normal,
                    point.separation,
                )

            self.bodies[body_a_idx] = body_a^
            self.bodies[body_b_idx] = body_b^

    # ===== Query Methods =====

    fn get_contact_count(self) -> Int:
        """Get number of active contacts."""
        return len(self.contacts)

    fn is_body_contacting(self, body_idx: Int) -> Bool:
        """Check if body is in contact with anything."""
        for i in range(len(self.contacts)):
            if (
                self.contacts[i].touching
                and (
                    self.contacts[i].body_a_idx == body_idx
                    or self.contacts[i].body_b_idx == body_idx
                )
            ):
                return True
        return False

    fn get_fixture_contacts(self, fixture_idx: Int) -> List[Int]:
        """Get indices of fixtures in contact with given fixture."""
        var result = List[Int]()
        for i in range(len(self.contacts)):
            if not self.contacts[i].touching:
                continue
            if self.contacts[i].manifold.fixture_a_idx == fixture_idx:
                result.append(self.contacts[i].manifold.fixture_b_idx)
            elif self.contacts[i].manifold.fixture_b_idx == fixture_idx:
                result.append(self.contacts[i].manifold.fixture_a_idx)
        return result^

    # ===== Raycast Methods =====

    fn raycast(
        self,
        ray_start: Vec2,
        ray_end: Vec2,
        filter_mask: UInt16 = 0xFFFF,
    ) -> RaycastResult:
        """Cast ray against all fixtures and return closest hit.

        Args:
            ray_start: Ray origin in world space
            ray_end: Ray end point in world space
            filter_mask: Only test fixtures whose category matches this mask

        Returns:
            RaycastResult with closest hit info
        """
        var best_result = RaycastResult.no_hit()

        for i in range(len(self.fixtures)):
            var fix = self.fixtures[i].copy()

            # Skip invalid fixtures
            if fix.body_idx < 0 or fix.body_idx >= len(self.bodies):
                continue

            # Check filter - fixture category must match our mask
            if (fix.filter.category_bits & filter_mask) == 0:
                continue

            var body = self.bodies[fix.body_idx].copy()
            var result: RaycastResult

            if fix.shape_type == SHAPE_EDGE:
                result = raycast_edge(ray_start, ray_end, fix.edge, body.transform)
            elif fix.shape_type == SHAPE_POLYGON:
                result = raycast_polygon(ray_start, ray_end, fix.polygon, body.transform)
            else:
                # Skip circles for now (lidar doesn't need them)
                continue

            if result.hit and result.fraction < best_result.fraction:
                best_result = result
                best_result.fixture_idx = i

        return best_result^

    fn raycast_fixture_range(
        self,
        ray_start: Vec2,
        ray_end: Vec2,
        fixture_start: Int,
        fixture_count: Int,
    ) -> RaycastResult:
        """Cast ray against a specific range of fixtures.

        Useful for testing against terrain only (for lidar).

        Args:
            ray_start: Ray origin in world space
            ray_end: Ray end point in world space
            fixture_start: First fixture index to test
            fixture_count: Number of fixtures to test

        Returns:
            RaycastResult with closest hit info
        """
        var best_result = RaycastResult.no_hit()

        var end_idx = min_i64(fixture_start + fixture_count, len(self.fixtures))
        for i in range(fixture_start, end_idx):
            var fix = self.fixtures[i].copy()

            # Skip invalid fixtures
            if fix.body_idx < 0 or fix.body_idx >= len(self.bodies):
                continue

            var body = self.bodies[fix.body_idx].copy()
            var result: RaycastResult

            if fix.shape_type == SHAPE_EDGE:
                result = raycast_edge(ray_start, ray_end, fix.edge, body.transform)
            elif fix.shape_type == SHAPE_POLYGON:
                result = raycast_polygon(ray_start, ray_end, fix.polygon, body.transform)
            else:
                continue

            if result.hit and result.fraction < best_result.fraction:
                best_result = result
                best_result.fixture_idx = i

        return best_result^


fn max_f64(a: Float64, b: Float64) -> Float64:
    """Maximum of two floats."""
    return a if a > b else b


fn min_i64(a: Int, b: Int) -> Int:
    """Minimum of two integers."""
    return a if a < b else b
