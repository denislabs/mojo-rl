"""Edge terrain collision detection for environments with varying terrain heights.

This collision system handles:
- Terrain defined as a chain of edge segments
- Polygon/circle bodies colliding with angled terrain
- Proper collision normals for sloped surfaces

Perfect for LunarLander with varying terrain heights.
"""

from math import cos, sin, sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from ..traits.collision import CollisionSystem

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_SHAPE,
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    MAX_POLYGON_VERTS,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_DEPTH,
    CONTACT_NORMAL_IMPULSE,
    CONTACT_TANGENT_IMPULSE,
)

# Maximum number of terrain edges per environment
# Set to 128 to support BipedalWalker's more complex terrain
comptime MAX_TERRAIN_EDGES: Int = 128


struct EdgeTerrainCollision(CollisionSystem):
    """Collision detection against edge-based terrain.

    Terrain is defined as a chain of connected edge segments.
    Each edge has two vertices and an outward normal (pointing up/out from terrain).

    Edge data layout per edge: [x0, y0, x1, y1, nx, ny]
    - (x0, y0): Start vertex
    - (x1, y1): End vertex
    - (nx, ny): Outward normal (typically pointing up)

    For N terrain points, there are N-1 edges.
    """

    # Edge data: [BATCH, MAX_TERRAIN_EDGES, 6] - stored as flat list
    # Layout per edge: [x0, y0, x1, y1, normal_x, normal_y]
    var edges: List[Scalar[dtype]]
    var edge_counts: List[Int]  # Number of edges per environment
    var num_envs: Int

    fn __init__(out self, num_envs: Int):
        """Initialize with number of environments.

        Args:
            num_envs: Number of parallel environments.
        """
        self.num_envs = num_envs
        self.edges = List[Scalar[dtype]](
            capacity=num_envs * MAX_TERRAIN_EDGES * 6
        )
        self.edge_counts = List[Int](capacity=num_envs)

        # Initialize with zeros
        for _ in range(num_envs * MAX_TERRAIN_EDGES * 6):
            self.edges.append(Scalar[dtype](0))
        for _ in range(num_envs):
            self.edge_counts.append(0)

    fn __copyinit__(out self, other: EdgeTerrainCollision):
        self.edges = List[Scalar[dtype]]()
        self.edge_counts = List[Int]()
        self.num_envs = other.num_envs
        for i in range(other.num_envs * MAX_TERRAIN_EDGES * 6):
            self.edges.append(other.edges[i])
        for i in range(other.num_envs):
            self.edge_counts.append(other.edge_counts[i])

    fn set_terrain_from_heights(
        mut self,
        env: Int,
        heights: List[Scalar[dtype]],
        x_start: Float64,
        x_spacing: Float64,
    ):
        """Set terrain edges from height values.

        Creates a chain of edges connecting (x_i, h_i) points.

        Args:
            env: Environment index.
            heights: List of terrain heights (N points).
            x_start: X coordinate of first point.
            x_spacing: Horizontal spacing between points.
        """
        var n_points = len(heights)
        var n_edges = n_points - 1

        if n_edges > MAX_TERRAIN_EDGES:
            n_edges = MAX_TERRAIN_EDGES

        self.edge_counts[env] = n_edges

        var base_offset = env * MAX_TERRAIN_EDGES * 6

        for i in range(n_edges):
            var x0 = x_start + Float64(i) * x_spacing
            var y0 = Float64(heights[i])
            var x1 = x_start + Float64(i + 1) * x_spacing
            var y1 = Float64(heights[i + 1])

            # Compute edge direction and outward normal
            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)

            # Normal perpendicular to edge, pointing "up" (leftward rotation of direction)
            # For a left-to-right edge, this gives upward normal
            var nx = -dy / length
            var ny = dx / length

            # Ensure normal points upward (positive y component)
            if ny < 0:
                nx = -nx
                ny = -ny

            var edge_offset = base_offset + i * 6
            self.edges[edge_offset + 0] = Scalar[dtype](x0)
            self.edges[edge_offset + 1] = Scalar[dtype](y0)
            self.edges[edge_offset + 2] = Scalar[dtype](x1)
            self.edges[edge_offset + 3] = Scalar[dtype](y1)
            self.edges[edge_offset + 4] = Scalar[dtype](nx)
            self.edges[edge_offset + 5] = Scalar[dtype](ny)

    fn set_flat_terrain(
        mut self, env: Int, ground_y: Float64, x_min: Float64, x_max: Float64
    ):
        """Set a single flat terrain edge.

        Args:
            env: Environment index.
            ground_y: Y coordinate of flat ground.
            x_min: Left edge of terrain.
            x_max: Right edge of terrain.
        """
        self.edge_counts[env] = 1

        var base_offset = env * MAX_TERRAIN_EDGES * 6
        self.edges[base_offset + 0] = Scalar[dtype](x_min)
        self.edges[base_offset + 1] = Scalar[dtype](ground_y)
        self.edges[base_offset + 2] = Scalar[dtype](x_max)
        self.edges[base_offset + 3] = Scalar[dtype](ground_y)
        self.edges[base_offset + 4] = Scalar[dtype](0)  # Normal x
        self.edges[base_offset + 5] = Scalar[dtype](1)  # Normal y (up)

    # =========================================================================
    # CPU Implementation
    # =========================================================================

    fn detect[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int = 16,
    ](
        self,
        bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        mut contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        mut contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """Detect terrain contacts for all bodies in all environments."""
        for env in range(BATCH):
            var count = 0
            var n_edges = self.edge_counts[env]
            var env_edge_base = env * MAX_TERRAIN_EDGES * 6

            for body_idx in range(NUM_BODIES):
                # Get body transform
                var body_x = bodies[env, body_idx, IDX_X]
                var body_y = bodies[env, body_idx, IDX_Y]
                var body_angle = bodies[env, body_idx, IDX_ANGLE]

                # Get shape info
                var shape_idx = Int(bodies[env, body_idx, IDX_SHAPE])
                var shape_type = Int(shapes[shape_idx, 0])

                var cos_a = cos(body_angle)
                var sin_a = sin(body_angle)

                if shape_type == SHAPE_POLYGON:
                    var n_verts = Int(shapes[shape_idx, 1])

                    for v in range(n_verts):
                        if v >= MAX_POLYGON_VERTS:
                            break

                        # Local vertex position
                        var local_x = shapes[shape_idx, 2 + v * 2]
                        var local_y = shapes[shape_idx, 3 + v * 2]

                        # Transform to world coordinates
                        var world_x = body_x + local_x * cos_a - local_y * sin_a
                        var world_y = body_y + local_x * sin_a + local_y * cos_a

                        # Check against each terrain edge
                        for edge_idx in range(n_edges):
                            var edge_offset = env_edge_base + edge_idx * 6
                            var e_x0 = self.edges[edge_offset + 0]
                            var e_y0 = self.edges[edge_offset + 1]
                            var e_x1 = self.edges[edge_offset + 2]
                            var e_y1 = self.edges[edge_offset + 3]
                            var e_nx = self.edges[edge_offset + 4]
                            var e_ny = self.edges[edge_offset + 5]

                            # Check if vertex is within edge's x range (with small margin)
                            var margin = Scalar[dtype](0.1)
                            var x_min = e_x0 - margin
                            var x_max = e_x1 + margin
                            if e_x0 > e_x1:
                                x_min = e_x1 - margin
                                x_max = e_x0 + margin

                            if world_x < x_min or world_x > x_max:
                                continue

                            # Project vertex onto edge line
                            # Edge direction: (dx, dy) = (x1-x0, y1-y0)
                            var edge_dx = e_x1 - e_x0
                            var edge_dy = e_y1 - e_y0
                            var edge_len_sq = (
                                edge_dx * edge_dx + edge_dy * edge_dy
                            )

                            if edge_len_sq < Scalar[dtype](1e-6):
                                continue

                            # Parameter t for closest point on edge
                            var t = (
                                (world_x - e_x0) * edge_dx
                                + (world_y - e_y0) * edge_dy
                            ) / edge_len_sq

                            # Clamp t to [0, 1] to stay on edge segment
                            if t < Scalar[dtype](0):
                                t = Scalar[dtype](0)
                            if t > Scalar[dtype](1):
                                t = Scalar[dtype](1)

                            # Closest point on edge
                            var closest_x = e_x0 + t * edge_dx
                            var closest_y = e_y0 + t * edge_dy

                            # Distance from vertex to closest point (signed by normal)
                            var dist_x = world_x - closest_x
                            var dist_y = world_y - closest_y

                            # Penetration depth (negative distance along normal = penetration)
                            var penetration = -(dist_x * e_nx + dist_y * e_ny)

                            if (
                                penetration > Scalar[dtype](0)
                                and count < MAX_CONTACTS
                            ):
                                contacts[env, count, CONTACT_BODY_A] = Scalar[
                                    dtype
                                ](body_idx)
                                contacts[env, count, CONTACT_BODY_B] = Scalar[
                                    dtype
                                ](-1)
                                contacts[env, count, CONTACT_POINT_X] = world_x
                                contacts[env, count, CONTACT_POINT_Y] = world_y
                                contacts[env, count, CONTACT_NORMAL_X] = e_nx
                                contacts[env, count, CONTACT_NORMAL_Y] = e_ny
                                contacts[
                                    env, count, CONTACT_DEPTH
                                ] = penetration
                                contacts[
                                    env, count, CONTACT_NORMAL_IMPULSE
                                ] = Scalar[dtype](0)
                                contacts[
                                    env, count, CONTACT_TANGENT_IMPULSE
                                ] = Scalar[dtype](0)
                                count += 1

                elif shape_type == SHAPE_CIRCLE:
                    var radius = shapes[shape_idx, 1]
                    var center_offset_x = shapes[shape_idx, 2]
                    var center_offset_y = shapes[shape_idx, 3]

                    # Transform center to world
                    var center_world_x = (
                        body_x
                        + center_offset_x * cos_a
                        - center_offset_y * sin_a
                    )
                    var center_world_y = (
                        body_y
                        + center_offset_x * sin_a
                        + center_offset_y * cos_a
                    )

                    # Check against each terrain edge
                    for edge_idx in range(n_edges):
                        var edge_offset = env_edge_base + edge_idx * 6
                        var e_x0 = self.edges[edge_offset + 0]
                        var e_y0 = self.edges[edge_offset + 1]
                        var e_x1 = self.edges[edge_offset + 2]
                        var e_y1 = self.edges[edge_offset + 3]
                        var e_nx = self.edges[edge_offset + 4]
                        var e_ny = self.edges[edge_offset + 5]

                        # Project circle center onto edge
                        var edge_dx = e_x1 - e_x0
                        var edge_dy = e_y1 - e_y0
                        var edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy

                        if edge_len_sq < Scalar[dtype](1e-6):
                            continue

                        var t = (
                            (center_world_x - e_x0) * edge_dx
                            + (center_world_y - e_y0) * edge_dy
                        ) / edge_len_sq

                        if t < Scalar[dtype](0):
                            t = Scalar[dtype](0)
                        if t > Scalar[dtype](1):
                            t = Scalar[dtype](1)

                        var closest_x = e_x0 + t * edge_dx
                        var closest_y = e_y0 + t * edge_dy

                        # Distance from center to closest point
                        var dist_x = center_world_x - closest_x
                        var dist_y = center_world_y - closest_y
                        var dist = sqrt(dist_x * dist_x + dist_y * dist_y)

                        # Penetration = radius - distance
                        var penetration = radius - dist

                        if (
                            penetration > Scalar[dtype](0)
                            and count < MAX_CONTACTS
                        ):
                            # Contact normal points from edge to circle
                            var contact_nx = rebind[Scalar[dtype]](e_nx)
                            var contact_ny = rebind[Scalar[dtype]](e_ny)
                            if dist > Scalar[dtype](1e-6):
                                contact_nx = rebind[Scalar[dtype]](
                                    dist_x
                                ) / rebind[Scalar[dtype]](dist)
                                contact_ny = rebind[Scalar[dtype]](
                                    dist_y
                                ) / rebind[Scalar[dtype]](dist)

                            contacts[env, count, CONTACT_BODY_A] = Scalar[
                                dtype
                            ](body_idx)
                            contacts[env, count, CONTACT_BODY_B] = Scalar[
                                dtype
                            ](-1)
                            contacts[env, count, CONTACT_POINT_X] = rebind[
                                Scalar[dtype]
                            ](closest_x)
                            contacts[env, count, CONTACT_POINT_Y] = rebind[
                                Scalar[dtype]
                            ](closest_y)
                            contacts[env, count, CONTACT_NORMAL_X] = contact_nx
                            contacts[env, count, CONTACT_NORMAL_Y] = contact_ny
                            contacts[env, count, CONTACT_DEPTH] = rebind[
                                Scalar[dtype]
                            ](penetration)
                            contacts[
                                env, count, CONTACT_NORMAL_IMPULSE
                            ] = Scalar[dtype](0)
                            contacts[
                                env, count, CONTACT_TANGENT_IMPULSE
                            ] = Scalar[dtype](0)
                            count += 1

            contact_counts[env] = Scalar[dtype](count)

    # =========================================================================
    # Strided GPU Kernels for 2D State Layout
    # =========================================================================
    #
    # These methods work with 2D [BATCH, STATE_SIZE] layout.
    # Bodies and edges are at offsets within each environment's state.
    # Shapes are shared across all environments (separate buffer).
    # Contacts are workspace output.
    # =========================================================================

    # =========================================================================
    # Single-Environment Methods (can be called from fused kernels)
    # =========================================================================

    # =========================================================================
    # Single-Body Method (for parallel collision detection)
    # =========================================================================

    # =========================================================================
    # Single Body-Edge Pair Method (for fully parallel collision detection)
    # =========================================================================

    @always_inline
    @staticmethod
    fn detect_single_body_edge[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY_EDGE: Int,  # Max contacts for one (body, edge) pair
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        env: Int,
        body_idx: Int,
        edge_idx: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_flags: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE), MutAnyOrigin
        ],
    ):
        """Detect collisions for a single (body, edge) pair.

        Each (body, edge) pair gets pre-allocated contact slots:
        - Slot base = (body_idx * MAX_EDGES + edge_idx) * MAX_CONTACTS_PER_BODY_EDGE

        This allows fully parallel execution over (env × body × edge).
        contact_flags[slot] = 1 if valid contact, 0 otherwise.
        """
        var slot_base = (body_idx * MAX_EDGES + edge_idx) * MAX_CONTACTS_PER_BODY_EDGE
        var count = 0

        var body_off = BODIES_OFFSET + body_idx * BODY_STATE_SIZE
        var body_x = state[env, body_off + IDX_X]
        var body_y = state[env, body_off + IDX_Y]
        var body_angle = state[env, body_off + IDX_ANGLE]
        var shape_idx = Int(state[env, body_off + IDX_SHAPE])
        var shape_type = Int(shapes[shape_idx, 0])

        var cos_a = cos(body_angle)
        var sin_a = sin(body_angle)

        # Read edge data
        var edge_off = EDGES_OFFSET + edge_idx * 6
        var e_x0 = state[env, edge_off + 0]
        var e_y0 = state[env, edge_off + 1]
        var e_x1 = state[env, edge_off + 2]
        var e_y1 = state[env, edge_off + 3]
        var e_nx = state[env, edge_off + 4]
        var e_ny = state[env, edge_off + 5]

        var edge_dx = e_x1 - e_x0
        var edge_dy = e_y1 - e_y0
        var edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy

        # Initialize all slots for this (body, edge) pair as invalid
        for i in range(MAX_CONTACTS_PER_BODY_EDGE):
            contact_flags[env, slot_base + i] = Scalar[dtype](0)

        if edge_len_sq < Scalar[dtype](1e-6):
            return

        if shape_type == SHAPE_POLYGON:
            var n_verts = Int(shapes[shape_idx, 1])

            var x_min = e_x0
            var x_max = e_x1
            if e_x0 > e_x1:
                x_min = e_x1
                x_max = e_x0

            for v in range(MAX_POLYGON_VERTS):
                if v >= n_verts or count >= MAX_CONTACTS_PER_BODY_EDGE:
                    break

                var local_x = shapes[shape_idx, 2 + v * 2]
                var local_y = shapes[shape_idx, 3 + v * 2]
                var world_x = body_x + local_x * cos_a - local_y * sin_a
                var world_y = body_y + local_x * sin_a + local_y * cos_a

                if world_x < x_min - Scalar[dtype](0.1) or world_x > x_max + Scalar[dtype](0.1):
                    continue

                var t = ((world_x - e_x0) * edge_dx + (world_y - e_y0) * edge_dy) / edge_len_sq
                if t < Scalar[dtype](0):
                    t = Scalar[dtype](0)
                if t > Scalar[dtype](1):
                    t = Scalar[dtype](1)

                var closest_x = e_x0 + t * edge_dx
                var closest_y = e_y0 + t * edge_dy

                var dist_x = world_x - closest_x
                var dist_y = world_y - closest_y
                var penetration = -(dist_x * e_nx + dist_y * e_ny)

                if penetration > Scalar[dtype](0):
                    var slot = slot_base + count
                    contacts[env, slot, CONTACT_BODY_A] = Scalar[dtype](body_idx)
                    contacts[env, slot, CONTACT_BODY_B] = Scalar[dtype](-1)
                    contacts[env, slot, CONTACT_POINT_X] = world_x
                    contacts[env, slot, CONTACT_POINT_Y] = world_y
                    contacts[env, slot, CONTACT_NORMAL_X] = e_nx
                    contacts[env, slot, CONTACT_NORMAL_Y] = e_ny
                    contacts[env, slot, CONTACT_DEPTH] = penetration
                    contacts[env, slot, CONTACT_NORMAL_IMPULSE] = Scalar[dtype](0)
                    contacts[env, slot, CONTACT_TANGENT_IMPULSE] = Scalar[dtype](0)
                    contact_flags[env, slot] = Scalar[dtype](1)
                    count += 1

        elif shape_type == SHAPE_CIRCLE:
            var radius = shapes[shape_idx, 1]
            var center_offset_x = shapes[shape_idx, 2]
            var center_offset_y = shapes[shape_idx, 3]

            var center_world_x = body_x + center_offset_x * cos_a - center_offset_y * sin_a
            var center_world_y = body_y + center_offset_x * sin_a + center_offset_y * cos_a

            var t = ((center_world_x - e_x0) * edge_dx + (center_world_y - e_y0) * edge_dy) / edge_len_sq
            if t < Scalar[dtype](0):
                t = Scalar[dtype](0)
            if t > Scalar[dtype](1):
                t = Scalar[dtype](1)

            var closest_x = e_x0 + t * edge_dx
            var closest_y = e_y0 + t * edge_dy

            var dist_x = center_world_x - closest_x
            var dist_y = center_world_y - closest_y
            var dist = sqrt(dist_x * dist_x + dist_y * dist_y)

            var penetration = radius - dist

            if penetration > Scalar[dtype](0):
                var contact_nx = rebind[Scalar[dtype]](e_nx)
                var contact_ny = rebind[Scalar[dtype]](e_ny)
                if dist > Scalar[dtype](1e-6):
                    contact_nx = rebind[Scalar[dtype]](dist_x) / rebind[Scalar[dtype]](dist)
                    contact_ny = rebind[Scalar[dtype]](dist_y) / rebind[Scalar[dtype]](dist)

                var slot = slot_base
                contacts[env, slot, CONTACT_BODY_A] = Scalar[dtype](body_idx)
                contacts[env, slot, CONTACT_BODY_B] = Scalar[dtype](-1)
                contacts[env, slot, CONTACT_POINT_X] = rebind[Scalar[dtype]](closest_x)
                contacts[env, slot, CONTACT_POINT_Y] = rebind[Scalar[dtype]](closest_y)
                contacts[env, slot, CONTACT_NORMAL_X] = contact_nx
                contacts[env, slot, CONTACT_NORMAL_Y] = contact_ny
                contacts[env, slot, CONTACT_DEPTH] = rebind[Scalar[dtype]](penetration)
                contacts[env, slot, CONTACT_NORMAL_IMPULSE] = Scalar[dtype](0)
                contacts[env, slot, CONTACT_TANGENT_IMPULSE] = Scalar[dtype](0)
                contact_flags[env, slot] = Scalar[dtype](1)

    # =========================================================================
    # Single-Body Method (for parallel collision detection over bodies only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn detect_single_body[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        env: Int,
        body_idx: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        n_edges: Int,
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * MAX_CONTACTS_PER_BODY, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        body_contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES), MutAnyOrigin
        ],
    ):
        """Detect collisions for a single body in a single environment.

        Each body writes to its own pre-allocated contact slots:
        - Body 0: slots [0, MAX_CONTACTS_PER_BODY)
        - Body 1: slots [MAX_CONTACTS_PER_BODY, 2*MAX_CONTACTS_PER_BODY)
        - etc.

        This allows parallel execution without atomics.
        """
        var count = 0
        var contact_base = body_idx * MAX_CONTACTS_PER_BODY

        var body_off = BODIES_OFFSET + body_idx * BODY_STATE_SIZE
        var body_x = state[env, body_off + IDX_X]
        var body_y = state[env, body_off + IDX_Y]
        var body_angle = state[env, body_off + IDX_ANGLE]
        var shape_idx = Int(state[env, body_off + IDX_SHAPE])
        var shape_type = Int(shapes[shape_idx, 0])

        var cos_a = cos(body_angle)
        var sin_a = sin(body_angle)

        if shape_type == SHAPE_POLYGON:
            var n_verts = Int(shapes[shape_idx, 1])

            for v in range(MAX_POLYGON_VERTS):
                if v >= n_verts:
                    break

                var local_x = shapes[shape_idx, 2 + v * 2]
                var local_y = shapes[shape_idx, 3 + v * 2]
                var world_x = body_x + local_x * cos_a - local_y * sin_a
                var world_y = body_y + local_x * sin_a + local_y * cos_a

                for edge_idx in range(MAX_EDGES):
                    if edge_idx >= n_edges:
                        break

                    var edge_off = EDGES_OFFSET + edge_idx * 6
                    var e_x0 = state[env, edge_off + 0]
                    var e_y0 = state[env, edge_off + 1]
                    var e_x1 = state[env, edge_off + 2]
                    var e_y1 = state[env, edge_off + 3]
                    var e_nx = state[env, edge_off + 4]
                    var e_ny = state[env, edge_off + 5]

                    var x_min = e_x0
                    var x_max = e_x1
                    if e_x0 > e_x1:
                        x_min = e_x1
                        x_max = e_x0

                    if world_x < x_min - Scalar[dtype](0.1) or world_x > x_max + Scalar[dtype](0.1):
                        continue

                    var edge_dx = e_x1 - e_x0
                    var edge_dy = e_y1 - e_y0
                    var edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy

                    if edge_len_sq < Scalar[dtype](1e-6):
                        continue

                    var t = ((world_x - e_x0) * edge_dx + (world_y - e_y0) * edge_dy) / edge_len_sq
                    if t < Scalar[dtype](0):
                        t = Scalar[dtype](0)
                    if t > Scalar[dtype](1):
                        t = Scalar[dtype](1)

                    var closest_x = e_x0 + t * edge_dx
                    var closest_y = e_y0 + t * edge_dy

                    var dist_x = world_x - closest_x
                    var dist_y = world_y - closest_y
                    var penetration = -(dist_x * e_nx + dist_y * e_ny)

                    if penetration > Scalar[dtype](0) and count < MAX_CONTACTS_PER_BODY:
                        var slot = contact_base + count
                        contacts[env, slot, CONTACT_BODY_A] = Scalar[dtype](body_idx)
                        contacts[env, slot, CONTACT_BODY_B] = Scalar[dtype](-1)
                        contacts[env, slot, CONTACT_POINT_X] = world_x
                        contacts[env, slot, CONTACT_POINT_Y] = world_y
                        contacts[env, slot, CONTACT_NORMAL_X] = e_nx
                        contacts[env, slot, CONTACT_NORMAL_Y] = e_ny
                        contacts[env, slot, CONTACT_DEPTH] = penetration
                        contacts[env, slot, CONTACT_NORMAL_IMPULSE] = Scalar[dtype](0)
                        contacts[env, slot, CONTACT_TANGENT_IMPULSE] = Scalar[dtype](0)
                        count += 1

        elif shape_type == SHAPE_CIRCLE:
            var radius = shapes[shape_idx, 1]
            var center_offset_x = shapes[shape_idx, 2]
            var center_offset_y = shapes[shape_idx, 3]

            var center_world_x = body_x + center_offset_x * cos_a - center_offset_y * sin_a
            var center_world_y = body_y + center_offset_x * sin_a + center_offset_y * cos_a

            for edge_idx in range(MAX_EDGES):
                if edge_idx >= n_edges:
                    break

                var edge_off = EDGES_OFFSET + edge_idx * 6
                var e_x0 = state[env, edge_off + 0]
                var e_y0 = state[env, edge_off + 1]
                var e_x1 = state[env, edge_off + 2]
                var e_y1 = state[env, edge_off + 3]
                var e_nx = state[env, edge_off + 4]
                var e_ny = state[env, edge_off + 5]

                var edge_dx = e_x1 - e_x0
                var edge_dy = e_y1 - e_y0
                var edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy

                if edge_len_sq < Scalar[dtype](1e-6):
                    continue

                var t = ((center_world_x - e_x0) * edge_dx + (center_world_y - e_y0) * edge_dy) / edge_len_sq
                if t < Scalar[dtype](0):
                    t = Scalar[dtype](0)
                if t > Scalar[dtype](1):
                    t = Scalar[dtype](1)

                var closest_x = e_x0 + t * edge_dx
                var closest_y = e_y0 + t * edge_dy

                var dist_x = center_world_x - closest_x
                var dist_y = center_world_y - closest_y
                var dist = sqrt(dist_x * dist_x + dist_y * dist_y)

                var penetration = radius - dist

                if penetration > Scalar[dtype](0) and count < MAX_CONTACTS_PER_BODY:
                    var contact_nx = rebind[Scalar[dtype]](e_nx)
                    var contact_ny = rebind[Scalar[dtype]](e_ny)
                    if dist > Scalar[dtype](1e-6):
                        contact_nx = rebind[Scalar[dtype]](dist_x) / rebind[Scalar[dtype]](dist)
                        contact_ny = rebind[Scalar[dtype]](dist_y) / rebind[Scalar[dtype]](dist)

                    var slot = contact_base + count
                    contacts[env, slot, CONTACT_BODY_A] = Scalar[dtype](body_idx)
                    contacts[env, slot, CONTACT_BODY_B] = Scalar[dtype](-1)
                    contacts[env, slot, CONTACT_POINT_X] = rebind[Scalar[dtype]](closest_x)
                    contacts[env, slot, CONTACT_POINT_Y] = rebind[Scalar[dtype]](closest_y)
                    contacts[env, slot, CONTACT_NORMAL_X] = contact_nx
                    contacts[env, slot, CONTACT_NORMAL_Y] = contact_ny
                    contacts[env, slot, CONTACT_DEPTH] = rebind[Scalar[dtype]](penetration)
                    contacts[env, slot, CONTACT_NORMAL_IMPULSE] = Scalar[dtype](0)
                    contacts[env, slot, CONTACT_TANGENT_IMPULSE] = Scalar[dtype](0)
                    count += 1

        body_contact_counts[env, body_idx] = Scalar[dtype](count)

    @always_inline
    @staticmethod
    fn detect_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        n_edges: Int,
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """Detect collisions for a single environment.

        This is the core logic, extracted to be callable from:
        - detect_kernel (standalone kernel)
        - PhysicsStepKernel (fused kernel)

        Returns the number of contacts detected.
        """
        var count = 0

        for body_idx in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body_idx * BODY_STATE_SIZE

            var body_x = state[env, body_off + IDX_X]
            var body_y = state[env, body_off + IDX_Y]
            var body_angle = state[env, body_off + IDX_ANGLE]
            var shape_idx = Int(state[env, body_off + IDX_SHAPE])
            var shape_type = Int(shapes[shape_idx, 0])

            var cos_a = cos(body_angle)
            var sin_a = sin(body_angle)

            if shape_type == SHAPE_POLYGON:
                var n_verts = Int(shapes[shape_idx, 1])

                for v in range(MAX_POLYGON_VERTS):
                    if v >= n_verts:
                        break

                    var local_x = shapes[shape_idx, 2 + v * 2]
                    var local_y = shapes[shape_idx, 3 + v * 2]
                    var world_x = body_x + local_x * cos_a - local_y * sin_a
                    var world_y = body_y + local_x * sin_a + local_y * cos_a

                    for edge_idx in range(MAX_EDGES):
                        if edge_idx >= n_edges:
                            break

                        var edge_off = EDGES_OFFSET + edge_idx * 6
                        var e_x0 = state[env, edge_off + 0]
                        var e_y0 = state[env, edge_off + 1]
                        var e_x1 = state[env, edge_off + 2]
                        var e_y1 = state[env, edge_off + 3]
                        var e_nx = state[env, edge_off + 4]
                        var e_ny = state[env, edge_off + 5]

                        var x_min = e_x0
                        var x_max = e_x1
                        if e_x0 > e_x1:
                            x_min = e_x1
                            x_max = e_x0

                        if world_x < x_min - Scalar[dtype](
                            0.1
                        ) or world_x > x_max + Scalar[dtype](0.1):
                            continue

                        var edge_dx = e_x1 - e_x0
                        var edge_dy = e_y1 - e_y0
                        var edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy

                        if edge_len_sq < Scalar[dtype](1e-6):
                            continue

                        var t = (
                            (world_x - e_x0) * edge_dx
                            + (world_y - e_y0) * edge_dy
                        ) / edge_len_sq
                        if t < Scalar[dtype](0):
                            t = Scalar[dtype](0)
                        if t > Scalar[dtype](1):
                            t = Scalar[dtype](1)

                        var closest_x = e_x0 + t * edge_dx
                        var closest_y = e_y0 + t * edge_dy

                        var dist_x = world_x - closest_x
                        var dist_y = world_y - closest_y
                        var penetration = -(dist_x * e_nx + dist_y * e_ny)

                        if (
                            penetration > Scalar[dtype](0)
                            and count < MAX_CONTACTS
                        ):
                            contacts[env, count, CONTACT_BODY_A] = Scalar[
                                dtype
                            ](body_idx)
                            contacts[env, count, CONTACT_BODY_B] = Scalar[
                                dtype
                            ](-1)
                            contacts[env, count, CONTACT_POINT_X] = world_x
                            contacts[env, count, CONTACT_POINT_Y] = world_y
                            contacts[env, count, CONTACT_NORMAL_X] = e_nx
                            contacts[env, count, CONTACT_NORMAL_Y] = e_ny
                            contacts[env, count, CONTACT_DEPTH] = penetration
                            contacts[
                                env, count, CONTACT_NORMAL_IMPULSE
                            ] = Scalar[dtype](0)
                            contacts[
                                env, count, CONTACT_TANGENT_IMPULSE
                            ] = Scalar[dtype](0)
                            count += 1

            elif shape_type == SHAPE_CIRCLE:
                var radius = shapes[shape_idx, 1]
                var center_offset_x = shapes[shape_idx, 2]
                var center_offset_y = shapes[shape_idx, 3]

                var center_world_x = (
                    body_x + center_offset_x * cos_a - center_offset_y * sin_a
                )
                var center_world_y = (
                    body_y + center_offset_x * sin_a + center_offset_y * cos_a
                )

                for edge_idx in range(MAX_EDGES):
                    if edge_idx >= n_edges:
                        break

                    var edge_off = EDGES_OFFSET + edge_idx * 6
                    var e_x0 = state[env, edge_off + 0]
                    var e_y0 = state[env, edge_off + 1]
                    var e_x1 = state[env, edge_off + 2]
                    var e_y1 = state[env, edge_off + 3]
                    var e_nx = state[env, edge_off + 4]
                    var e_ny = state[env, edge_off + 5]

                    var edge_dx = e_x1 - e_x0
                    var edge_dy = e_y1 - e_y0
                    var edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy

                    if edge_len_sq < Scalar[dtype](1e-6):
                        continue

                    var t = (
                        (center_world_x - e_x0) * edge_dx
                        + (center_world_y - e_y0) * edge_dy
                    ) / edge_len_sq
                    if t < Scalar[dtype](0):
                        t = Scalar[dtype](0)
                    if t > Scalar[dtype](1):
                        t = Scalar[dtype](1)

                    var closest_x = e_x0 + t * edge_dx
                    var closest_y = e_y0 + t * edge_dy

                    var dist_x = center_world_x - closest_x
                    var dist_y = center_world_y - closest_y
                    var dist = sqrt(dist_x * dist_x + dist_y * dist_y)

                    var penetration = radius - dist

                    if (
                        penetration > Scalar[dtype](0)
                        and count < MAX_CONTACTS
                    ):
                        var contact_nx = rebind[Scalar[dtype]](e_nx)
                        var contact_ny = rebind[Scalar[dtype]](e_ny)
                        if dist > Scalar[dtype](1e-6):
                            contact_nx = rebind[Scalar[dtype]](
                                dist_x
                            ) / rebind[Scalar[dtype]](dist)
                            contact_ny = rebind[Scalar[dtype]](
                                dist_y
                            ) / rebind[Scalar[dtype]](dist)

                        contacts[env, count, CONTACT_BODY_A] = Scalar[
                            dtype
                        ](body_idx)
                        contacts[env, count, CONTACT_BODY_B] = Scalar[
                            dtype
                        ](-1)
                        contacts[env, count, CONTACT_POINT_X] = rebind[
                            Scalar[dtype]
                        ](closest_x)
                        contacts[env, count, CONTACT_POINT_Y] = rebind[
                            Scalar[dtype]
                        ](closest_y)
                        contacts[env, count, CONTACT_NORMAL_X] = contact_nx
                        contacts[env, count, CONTACT_NORMAL_Y] = contact_ny
                        contacts[env, count, CONTACT_DEPTH] = rebind[
                            Scalar[dtype]
                        ](penetration)
                        contacts[
                            env, count, CONTACT_NORMAL_IMPULSE
                        ] = Scalar[dtype](0)
                        contacts[
                            env, count, CONTACT_TANGENT_IMPULSE
                        ] = Scalar[dtype](0)
                        count += 1

        contact_counts[env] = Scalar[dtype](count)

    @always_inline
    @staticmethod
    fn detect_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        edge_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """GPU kernel for edge terrain collision with 2D strided layout."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var n_edges = Int(edge_counts[env])
        EdgeTerrainCollision.detect_single_env[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS,
            MAX_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            EDGES_OFFSET,
        ](env, state, shapes, n_edges, contacts, contact_counts)

    @staticmethod
    fn detect_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        state_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch strided collision detection kernel on GPU."""
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
        ):
            EdgeTerrainCollision.detect_kernel[
                BATCH,
                NUM_BODIES,
                NUM_SHAPES,
                MAX_CONTACTS,
                MAX_EDGES,
                STATE_SIZE,
                BODIES_OFFSET,
                EDGES_OFFSET,
            ](state, shapes, edge_counts, contacts, contact_counts)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            shapes,
            edge_counts,
            contacts,
            contact_counts,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Parallel Collision Detection (parallelized over env × body)
    # =========================================================================

    @always_inline
    @staticmethod
    fn detect_parallel_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        edge_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * MAX_CONTACTS_PER_BODY, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        body_contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES), MutAnyOrigin
        ],
    ):
        """Parallel GPU kernel for collision detection using 2D grid.

        Grid dimensions: (BATCH, NUM_BODIES)
        - X dimension: environment index
        - Y dimension: body index

        Each thread processes one (env, body) pair.
        Each body writes to its own pre-allocated contact slots.
        """
        # 2D thread indexing: X = env, Y = body
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var body_idx = Int(block_dim.y * block_idx.y + thread_idx.y)

        if env >= BATCH or body_idx >= NUM_BODIES:
            return

        var n_edges = Int(edge_counts[env])

        EdgeTerrainCollision.detect_single_body[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS_PER_BODY,
            MAX_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            EDGES_OFFSET,
        ](env, body_idx, state, shapes, n_edges, contacts, body_contact_counts)

    @staticmethod
    fn detect_parallel_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        state_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut body_contact_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch parallel collision detection kernel.

        Parallelizes over (env × body) = BATCH × NUM_BODIES threads.
        Each body writes to its own pre-allocated contact slots.

        Contact layout: [BATCH, NUM_BODIES * MAX_CONTACTS_PER_BODY, CONTACT_DATA_SIZE]
        Body contact counts: [BATCH, NUM_BODIES]

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH, STATE_SIZE].
            shapes_buf: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            edge_counts_buf: Terrain edge counts [BATCH].
            contacts_buf: Contact output [BATCH, NUM_BODIES * MAX_CONTACTS_PER_BODY, CONTACT_DATA_SIZE].
            body_contact_counts_buf: Per-body contact counts [BATCH, NUM_BODIES].
        """
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * MAX_CONTACTS_PER_BODY, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var body_contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES), MutAnyOrigin
        ](body_contact_counts_buf.unsafe_ptr())

        # 2D grid: X covers envs, Y covers bodies
        # Block size: use smaller blocks since we're 2D
        comptime BLOCK_X = 32  # Threads per block in X (env dimension)
        comptime BLOCK_Y = NUM_BODIES  # Threads per block in Y (body dimension)
        comptime GRID_X = (BATCH + BLOCK_X - 1) // BLOCK_X
        comptime GRID_Y = 1  # All bodies fit in one block in Y

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NUM_BODIES * MAX_CONTACTS_PER_BODY, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            body_contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH, NUM_BODIES), MutAnyOrigin
            ],
        ):
            EdgeTerrainCollision.detect_parallel_kernel[
                BATCH,
                NUM_BODIES,
                NUM_SHAPES,
                MAX_CONTACTS_PER_BODY,
                MAX_EDGES,
                STATE_SIZE,
                BODIES_OFFSET,
                EDGES_OFFSET,
            ](state, shapes, edge_counts, contacts, body_contact_counts)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            shapes,
            edge_counts,
            contacts,
            body_contact_counts,
            grid_dim=(GRID_X, GRID_Y),
            block_dim=(BLOCK_X, BLOCK_Y),
        )

    # =========================================================================
    # Fully Parallel Collision Detection (parallelized over env × body × edge)
    # =========================================================================

    @always_inline
    @staticmethod
    fn detect_body_edge_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY_EDGE: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        edge_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_flags: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE), MutAnyOrigin
        ],
    ):
        """Fully parallel GPU kernel for collision detection using 3D indexing.

        Thread indexing:
        - X dimension: environment index
        - Y dimension: body index
        - Z dimension: edge index

        Each thread processes one (env, body, edge) triplet.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var body_idx = Int(block_dim.y * block_idx.y + thread_idx.y)
        var edge_idx = Int(block_dim.z * block_idx.z + thread_idx.z)

        if env >= BATCH or body_idx >= NUM_BODIES or edge_idx >= MAX_EDGES:
            return

        # Skip if this edge doesn't exist for this env
        var n_edges = Int(edge_counts[env])
        if edge_idx >= n_edges:
            # Mark slots as invalid
            var slot_base = (body_idx * MAX_EDGES + edge_idx) * MAX_CONTACTS_PER_BODY_EDGE
            for i in range(MAX_CONTACTS_PER_BODY_EDGE):
                contact_flags[env, slot_base + i] = Scalar[dtype](0)
            return

        EdgeTerrainCollision.detect_single_body_edge[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS_PER_BODY_EDGE,
            MAX_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            EDGES_OFFSET,
        ](env, body_idx, edge_idx, state, shapes, contacts, contact_flags)

    @staticmethod
    fn detect_body_edge_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY_EDGE: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        state_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_flags_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch fully parallel collision detection over (env × body × edge).

        Total threads: BATCH × NUM_BODIES × MAX_EDGES
        Each thread handles one (body, edge) pair for one environment.

        Contact layout: [BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE, CONTACT_DATA_SIZE]
        Contact flags:  [BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE] (1=valid, 0=invalid)

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH, STATE_SIZE].
            shapes_buf: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            edge_counts_buf: Terrain edge counts [BATCH].
            contacts_buf: Contact output.
            contact_flags_buf: Validity flags for each contact slot.
        """
        comptime TOTAL_CONTACTS = NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE

        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, TOTAL_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_flags = LayoutTensor[
            dtype, Layout.row_major(BATCH, TOTAL_CONTACTS), MutAnyOrigin
        ](contact_flags_buf.unsafe_ptr())

        # 3D grid: X=envs, Y=bodies, Z=edges
        # Choose block sizes to fit hardware limits (typically 1024 threads/block max)
        comptime BLOCK_X = 8   # Envs per block
        comptime BLOCK_Y = NUM_BODIES  # All bodies in one block dimension
        comptime BLOCK_Z = 8   # Edges per block (adjust based on MAX_EDGES)

        comptime GRID_X = (BATCH + BLOCK_X - 1) // BLOCK_X
        comptime GRID_Y = 1  # All bodies fit in block
        comptime GRID_Z = (MAX_EDGES + BLOCK_Z - 1) // BLOCK_Z

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, TOTAL_CONTACTS, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            contact_flags: LayoutTensor[
                dtype, Layout.row_major(BATCH, TOTAL_CONTACTS), MutAnyOrigin
            ],
        ):
            EdgeTerrainCollision.detect_body_edge_kernel[
                BATCH,
                NUM_BODIES,
                NUM_SHAPES,
                MAX_CONTACTS_PER_BODY_EDGE,
                MAX_EDGES,
                STATE_SIZE,
                BODIES_OFFSET,
                EDGES_OFFSET,
            ](state, shapes, edge_counts, contacts, contact_flags)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            shapes,
            edge_counts,
            contacts,
            contact_flags,
            grid_dim=(GRID_X, GRID_Y, GRID_Z),
            block_dim=(BLOCK_X, BLOCK_Y, BLOCK_Z),
        )
