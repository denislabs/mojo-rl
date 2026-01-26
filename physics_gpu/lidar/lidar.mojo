"""Lidar raycast module for BipedalWalker environment.

Implements ray-edge intersection testing for lidar sensor simulation.
Casts rays from the hull and detects terrain intersections.
"""

from math import sqrt, sin, cos, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from ..constants import dtype, TPB


struct Lidar:
    """Lidar raycast implementation for terrain detection.

    Casts rays from a source position and detects intersections with
    terrain edges. Used by BipedalWalker for the 10-dimensional lidar
    observation component.
    """

    @staticmethod
    @always_inline
    fn _ray_edge_intersection(
        ray_ox: Scalar[dtype],
        ray_oy: Scalar[dtype],
        ray_dx: Scalar[dtype],
        ray_dy: Scalar[dtype],
        edge_x0: Scalar[dtype],
        edge_y0: Scalar[dtype],
        edge_x1: Scalar[dtype],
        edge_y1: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Compute ray-edge intersection using parametric line intersection.

        Args:
            ray_ox, ray_oy: Ray origin.
            ray_dx, ray_dy: Ray direction (not normalized).
            edge_x0, edge_y0: Edge start point.
            edge_x1, edge_y1: Edge end point.

        Returns:
            Intersection parameter t in [0, 1] if hit, -1 otherwise.
            t=0 means hit at ray origin, t=1 means hit at ray end.
        """
        # Edge direction
        var ex = edge_x1 - edge_x0
        var ey = edge_y1 - edge_y0

        # Denominator: cross product of directions
        var denom = ray_dx * ey - ray_dy * ex

        # Check if parallel (denom ~= 0)
        if denom > Scalar[dtype](-1e-10) and denom < Scalar[dtype](1e-10):
            return Scalar[dtype](-1.0)

        # Vector from edge start to ray origin
        var dx = ray_ox - edge_x0
        var dy = ray_oy - edge_y0

        # Compute parameters
        var t = (ex * dy - ey * dx) / denom  # Parameter along ray
        var u = (ray_dx * dy - ray_dy * dx) / denom  # Parameter along edge

        # Check if intersection is valid (t in [0, 1], u in [0, 1])
        if t >= Scalar[dtype](0.0) and t <= Scalar[dtype](1.0) and u >= Scalar[dtype](0.0) and u <= Scalar[dtype](1.0):
            return t

        return Scalar[dtype](-1.0)

    @staticmethod
    fn raycast_single[
        STATE_SIZE: Int,
        EDGES_OFFSET: Int,
        EDGE_COUNT_OFFSET: Int,
        NUM_LIDAR: Int,
        MAX_EDGES: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(1, STATE_SIZE),
            MutAnyOrigin,
        ],
        obs: LayoutTensor[
            dtype,
            Layout.row_major(NUM_LIDAR),
            MutAnyOrigin,
        ],
        hull_x: Scalar[dtype],
        hull_y: Scalar[dtype],
        hull_angle: Scalar[dtype],
        lidar_range: Scalar[dtype],
    ):
        """Cast lidar rays for a single environment (CPU mode).

        Args:
            state: Environment state containing terrain edges.
            obs: Output buffer for lidar observations [NUM_LIDAR].
            hull_x, hull_y: Hull center position.
            hull_angle: Hull rotation angle.
            lidar_range: Maximum lidar range in world units.
        """
        var n_edges = Int(state[0, EDGE_COUNT_OFFSET])

        # Lidar rays span from 0 to ~1.5 radians (85 degrees)
        # Looking downward/forward from the hull
        for i in range(NUM_LIDAR):
            # Angle relative to hull: 0 to 1.5 radians
            var local_angle = Scalar[dtype](i) * Scalar[dtype](1.5) / Scalar[dtype](NUM_LIDAR - 1)
            var world_angle = hull_angle - local_angle - Scalar[dtype](pi / 2.0)

            # Ray direction (downward/forward)
            var ray_dx = cos(world_angle) * lidar_range
            var ray_dy = sin(world_angle) * lidar_range

            # Find minimum hit distance
            var min_t = Scalar[dtype](1.0)  # Default to max range

            for edge in range(n_edges):
                if edge >= MAX_EDGES:
                    break

                var edge_off = EDGES_OFFSET + edge * 6
                var edge_x0 = state[0, edge_off + 0]
                var edge_y0 = state[0, edge_off + 1]
                var edge_x1 = state[0, edge_off + 2]
                var edge_y1 = state[0, edge_off + 3]

                var t = Lidar._ray_edge_intersection(
                    hull_x, hull_y, ray_dx, ray_dy,
                    edge_x0, edge_y0, edge_x1, edge_y1,
                )

                if t >= Scalar[dtype](0.0) and t < min_t:
                    min_t = t

            # Store normalized hit fraction (0 = hit at hull, 1 = no hit within range)
            obs[i] = min_t

    @staticmethod
    @always_inline
    fn raycast_env_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        EDGES_OFFSET: Int,
        EDGE_COUNT_OFFSET: Int,
        LIDAR_START_IDX: Int,
        NUM_LIDAR: Int,
        MAX_EDGES: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        obs: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, OBS_DIM),
            MutAnyOrigin,
        ],
        hull_x: Scalar[dtype],
        hull_y: Scalar[dtype],
        hull_angle: Scalar[dtype],
        lidar_range: Scalar[dtype],
    ):
        """Cast lidar rays for a single environment in a batch (GPU inline version).

        This method is designed to be called from within a fused GPU kernel.

        Args:
            env: Environment index in the batch.
            state: Batched environment states.
            obs: Output observation buffer.
            hull_x, hull_y: Hull center position.
            hull_angle: Hull rotation angle.
            lidar_range: Maximum lidar range.
        """
        var n_edges = Int(state[env, EDGE_COUNT_OFFSET])

        for i in range(NUM_LIDAR):
            # Angle relative to hull: 0 to 1.5 radians
            var local_angle = Scalar[dtype](i) * Scalar[dtype](1.5) / Scalar[dtype](NUM_LIDAR - 1)
            var world_angle = hull_angle - local_angle - Scalar[dtype](pi / 2.0)

            # Ray direction
            var ray_dx = cos(world_angle) * lidar_range
            var ray_dy = sin(world_angle) * lidar_range

            # Find minimum hit distance
            var min_t = Scalar[dtype](1.0)

            for edge in range(MAX_EDGES):
                if edge >= n_edges:
                    break

                var edge_off = EDGES_OFFSET + edge * 6
                var edge_x0 = state[env, edge_off + 0]
                var edge_y0 = state[env, edge_off + 1]
                var edge_x1 = state[env, edge_off + 2]
                var edge_y1 = state[env, edge_off + 3]

                var t = Lidar._ray_edge_intersection(
                    hull_x, hull_y, ray_dx, ray_dy,
                    edge_x0, edge_y0, edge_x1, edge_y1,
                )

                if t >= Scalar[dtype](0.0) and t < min_t:
                    min_t = t

            # Store in observation buffer at lidar offset
            obs[env, LIDAR_START_IDX + i] = min_t

    @staticmethod
    fn _raycast_kernel[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
        EDGE_COUNT_OFFSET: Int,
        LIDAR_START_IDX: Int,
        NUM_LIDAR: Int,
        MAX_EDGES: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        obs: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, OBS_DIM),
            MutAnyOrigin,
        ],
        lidar_range: Scalar[dtype],
    ):
        """GPU kernel for batched lidar raycast."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        # Get hull position and angle (body 0 = hull)
        var hull_off = BODIES_OFFSET  # Hull is body 0
        var hull_x = state[env, hull_off + 0]  # IDX_X
        var hull_y = state[env, hull_off + 1]  # IDX_Y
        var hull_angle = state[env, hull_off + 2]  # IDX_ANGLE

        Lidar.raycast_env_gpu[
            BATCH, STATE_SIZE, OBS_DIM, EDGES_OFFSET, EDGE_COUNT_OFFSET,
            LIDAR_START_IDX, NUM_LIDAR, MAX_EDGES,
        ](env, state, obs, hull_x, hull_y, hull_angle, lidar_range)

    @staticmethod
    fn raycast_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        BODIES_OFFSET: Int,
        EDGES_OFFSET: Int,
        EDGE_COUNT_OFFSET: Int,
        LIDAR_START_IDX: Int,
        NUM_LIDAR: Int,
        MAX_EDGES: Int,
    ](
        ctx: DeviceContext,
        state_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        lidar_range: Scalar[dtype],
    ) raises:
        """GPU batched lidar raycast.

        Args:
            ctx: Device context.
            state_buf: Buffer containing environment states [BATCH, STATE_SIZE].
            obs_buf: Output observation buffer [BATCH, OBS_DIM].
            lidar_range: Maximum lidar range.
        """
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())

        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
            ],
            lidar_range: Scalar[dtype],
        ):
            Lidar._raycast_kernel[
                BATCH, STATE_SIZE, OBS_DIM, BODIES_OFFSET, EDGES_OFFSET,
                EDGE_COUNT_OFFSET, LIDAR_START_IDX, NUM_LIDAR, MAX_EDGES,
            ](state, obs, lidar_range)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            obs,
            lidar_range,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
