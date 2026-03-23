"""AlphaZero State — CPU and GPU state containers.

Much simpler than MuZero: one network (policy+value), circular replay
buffer storing (obs, mcts_policy, game_outcome) tuples.
"""

from std.memory import alloc, memset
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from .configs import AlphaZeroConfig


# ═══════════════════════════════════════════════════════════════════════════
# CPU State
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroCPUState[Config: AlphaZeroConfig](Movable):
    """CPU state: one prediction network + sliding window replay buffer.

    Replay stores (obs, mcts_policy, game_outcome) tuples.
    The buffer keeps only the last K iterations of self-play data,
    where K = Config.history_window. Old iterations are discarded
    to prevent stale data from dominating training.

    Internally uses a flat array with compaction: when a new iteration
    starts and we exceed K iterations, the oldest iteration's data is
    removed by shifting remaining data forward.
    """

    comptime OBS: Int = Self.Config.obs_dim
    comptime ACT: Int = Self.Config.action_dim
    comptime CAPACITY: Int = Self.Config.buffer_capacity
    comptime WINDOW: Int = Self.Config.history_window
    comptime PredModel = Self.Config.PredModel
    comptime OptType = Self.Config.OptType

    # Network
    var prediction: NetworkState[Self.PredModel, Self.OptType]

    # Flat replay buffer (active region: [0, buf_size))
    var buf_obs: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [CAP * OBS]
    var buf_policy: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [CAP * ACT]
    var buf_value: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [CAP]
    var buf_size: Int

    # Iteration tracking: iter_boundaries[i] = start index of iteration i
    # Number of stored iterations = num_iters
    # Iteration i spans [iter_boundaries[i], iter_boundaries[i+1])
    # Last iteration spans [iter_boundaries[num_iters-1], buf_size)
    var iter_boundaries: List[Int]
    var num_iters: Int

    def __init__(out self):
        self.prediction = NetworkState[Self.PredModel, Self.OptType]()
        self.prediction.initialize[Kaiming[]]()

        comptime OBS_SIZE = Self.CAPACITY * Self.OBS
        self.buf_obs = alloc[Scalar[dtype]](OBS_SIZE)
        memset(self.buf_obs, 0, OBS_SIZE)

        comptime POL_SIZE = Self.CAPACITY * Self.ACT
        self.buf_policy = alloc[Scalar[dtype]](POL_SIZE)
        memset(self.buf_policy, 0, POL_SIZE)

        self.buf_value = alloc[Scalar[dtype]](Self.CAPACITY)
        memset(self.buf_value, 0, Self.CAPACITY)

        self.buf_size = 0
        self.iter_boundaries = List[Int]()
        self.iter_boundaries.append(0)  # First iteration starts at 0
        self.num_iters = 1

    def __init__(out self, *, deinit take: Self):
        self.prediction = take.prediction^
        self.buf_obs = take.buf_obs
        self.buf_policy = take.buf_policy
        self.buf_value = take.buf_value
        self.buf_size = take.buf_size
        self.iter_boundaries = take.iter_boundaries^
        self.num_iters = take.num_iters

    def __del__(deinit self):
        self.buf_obs.free()
        self.buf_policy.free()
        self.buf_value.free()

    def add(
        mut self,
        obs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        policy: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        value: Scalar[dtype],
    ):
        """Store one training sample. Drops oldest iteration if at capacity."""
        # If buffer is full, force eviction of oldest iteration
        if self.buf_size >= Self.CAPACITY:
            if self.num_iters > 1:
                self._evict_oldest()
            else:
                # Single iteration fills entire buffer — overwrite from start
                self.buf_size = 0
                self.iter_boundaries[0] = 0

        var idx = self.buf_size
        for i in range(Self.OBS):
            self.buf_obs[idx * Self.OBS + i] = obs[i]
        for i in range(Self.ACT):
            self.buf_policy[idx * Self.ACT + i] = policy[i]
        self.buf_value[idx] = value
        self.buf_size += 1

    def start_new_iteration(mut self):
        """Mark the start of a new self-play iteration.

        If we already have history_window iterations, evict the oldest
        before starting a new one. This keeps the buffer fresh.
        """
        # Evict oldest iterations until we have room for one more
        while self.num_iters >= Self.WINDOW:
            self._evict_oldest()

        # Mark new iteration boundary at current write position
        self.iter_boundaries.append(self.buf_size)
        self.num_iters += 1

    def _evict_oldest(mut self):
        """Remove the oldest iteration by shifting data forward."""
        if self.num_iters <= 1:
            return

        # Oldest iteration spans [iter_boundaries[0], iter_boundaries[1])
        var drop_count = self.iter_boundaries[1] - self.iter_boundaries[0]
        if drop_count <= 0:
            # Empty iteration — just remove the boundary
            _ = self.iter_boundaries.pop(0)
            self.num_iters -= 1
            return

        var keep_count = self.buf_size - drop_count

        # Shift data left by drop_count
        for i in range(keep_count * Self.OBS):
            self.buf_obs[i] = self.buf_obs[drop_count * Self.OBS + i]
        for i in range(keep_count * Self.ACT):
            self.buf_policy[i] = self.buf_policy[drop_count * Self.ACT + i]
        for i in range(keep_count):
            self.buf_value[i] = self.buf_value[drop_count + i]

        # Update boundaries — shift all by -drop_count and remove first
        _ = self.iter_boundaries.pop(0)
        for i in range(len(self.iter_boundaries)):
            self.iter_boundaries[i] -= drop_count

        self.buf_size = keep_count
        self.num_iters -= 1

    def is_ready(self, batch_size: Int) -> Bool:
        return self.buf_size >= batch_size * 2


# ═══════════════════════════════════════════════════════════════════════════
# GPU State
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroGPUState[Config: AlphaZeroConfig](Movable):
    """GPU state: prediction network + training scratch buffers."""

    comptime OBS: Int = Self.Config.obs_dim
    comptime ACT: Int = Self.Config.action_dim
    comptime BATCH: Int = Self.Config.batch_size
    comptime PredModel = Self.Config.PredModel
    comptime OptType = Self.Config.OptType
    comptime PRED_OUT: Int = Self.Config.action_dim + 1  # policy + value

    # GPU network
    var prediction: GPUNetworkState[Self.PredModel, Self.OptType]

    # Training batch (uploaded from CPU)
    var batch_obs: DeviceBuffer[dtype]  # [BATCH * OBS]
    var batch_policy: DeviceBuffer[dtype]  # [BATCH * ACT]
    var batch_value: DeviceBuffer[dtype]  # [BATCH]

    # Forward/backward scratch
    var pred_out: DeviceBuffer[dtype]  # [BATCH * PRED_OUT]
    var pred_cache: DeviceBuffer[dtype]  # [BATCH * PredModel.CACHE_SIZE]
    var grad_out: DeviceBuffer[dtype]  # [BATCH * PRED_OUT]
    var grad_in: DeviceBuffer[dtype]  # [BATCH * OBS]
    var workspace: DeviceBuffer[dtype]  # [BATCH * WS]

    # Host transfer
    var obs_host: HostBuffer[dtype]  # [BATCH * OBS]
    var policy_host: HostBuffer[dtype]  # [BATCH * ACT]
    var value_host: HostBuffer[dtype]  # [BATCH]

    def __init__(out self, ctx: DeviceContext) raises:
        self.prediction = GPUNetworkState[Self.PredModel, Self.OptType](ctx)

        self.batch_obs = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)
        self.batch_policy = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACT
        )
        self.batch_value = ctx.enqueue_create_buffer[dtype](Self.BATCH)

        self.pred_out = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.PRED_OUT
        )

        comptime CACHE_SIZE = Self.PredModel.CACHE_SIZE
        self.pred_cache = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * CACHE_SIZE
        )

        self.grad_out = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.PRED_OUT
        )
        self.grad_in = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)

        comptime WS = Self.PredModel.WORKSPACE_SIZE_PER_SAMPLE
        self.workspace = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * WS if WS > 0 else 1
        )

        self.obs_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.OBS
        )
        self.policy_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.ACT
        )
        self.value_host = ctx.enqueue_create_host_buffer[dtype](Self.BATCH)

    def __init__(out self, *, deinit take: Self):
        self.prediction = take.prediction^
        self.batch_obs = take.batch_obs^
        self.batch_policy = take.batch_policy^
        self.batch_value = take.batch_value^
        self.pred_out = take.pred_out^
        self.pred_cache = take.pred_cache^
        self.grad_out = take.grad_out^
        self.grad_in = take.grad_in^
        self.workspace = take.workspace^
        self.obs_host = take.obs_host^
        self.policy_host = take.policy_host^
        self.value_host = take.value_host^

    def upload_from(
        mut self, cpu: AlphaZeroCPUState[Self.Config], ctx: DeviceContext
    ) raises:
        self.prediction.upload_from(cpu.prediction, ctx)

    def download_to(
        mut self, mut cpu: AlphaZeroCPUState[Self.Config], ctx: DeviceContext
    ) raises:
        self.prediction.download_to(cpu.prediction, ctx)
