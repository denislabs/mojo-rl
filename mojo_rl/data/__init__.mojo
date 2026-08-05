"""mojo_rl.data — the shared trajectory store.

One column-oriented format behind offline datasets, replay dumps,
demonstrations and BFM trajectory data. See `docs/DATA_PLATFORM_PLAN.md`.

    from mojo_rl.data import ColumnSpec, TrajectoryStore, TrajectoryStoreWriter

    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, 9))
    cols.append(ColumnSpec(String("action"), DType.float32, 6))
    var w = TrajectoryStoreWriter(path, cols^, env_id=String("walker-walk"))
    w.append[DType.float32](String("qpos"), qpos_ptr, n)
    w.append[DType.float32](String("action"), act_ptr, n)
    w.end_episode()
    w.close()

    var s = TrajectoryStore(path)
    var qpos = s.load_column[DType.float32](String("qpos"))
"""

from .column import (
    ColumnSpec, dtype_bytes, dtype_from_h5, dtype_from_name, dtype_name,
)
from .episode_index import EpisodeIndex
from .manifest import (
    MANIFEST_DATASET, Manifest, SCHEMA_VERSION, parse_column, parse_manifest,
)
from .resident import (
    IDX_DT, IndexBatch, ResidentColumn,
)
from .blocks import (
    StorePerSampleCpuStep, StoreUniformSampleCpuStep,
    StoreUniformSampleGpuStep,
)
from .replay import StoreReplay
from .replay_gpu import StoreReplayGpu
from .sampler import (
    PrioritizedSampler,
    SequenceWindowSampler,
    UniformDeviceSampler,
    UniformSampler,
)
from .store import (
    DEFAULT_CHUNK_ROWS,
    DEFAULT_MAX_RESIDENT_BYTES,
    EP_LEN_DATASET,
    EP_OFFSET_DATASET,
    TrajectoryStore,
    TrajectoryStoreWriter,
)
