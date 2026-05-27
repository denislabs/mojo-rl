"""OnPolicyState — shared per-step flow object for on-policy block trainers.

Analog to `trainer_block.mojo`'s `TrainerState` (off-policy). Carries:
  - the full rollout buffers (obs/act/olp/rew/val/done/term/adv/ret)
    sized ROLLOUT_LEN, written by record_step + gae_step, read by
    minibatch_gather_step
  - the per-step caches (action / log_prob / value) filled by act_step,
    drained by record_step
  - the per-step BATCH=1 scratches (ob1/ao1/v1/z) for actor + critic
    forward calls during action selection
  - the minibatch BATCH=MINIBATCH scratches (mb_obs/mb_act/mb_olp/mb_adv/
    mb_ret/mb_v/mb_gv/mb_gi) written by gather, consumed by actor/critic
    train steps
  - rollout cursor + ctx + Int32 indices for the Fisher-Yates shuffle

P.1 is CPU + N_ENVS=1. P.3 lifts the rollout layout to
[ROLLOUT_LEN, N_ENVS] and adds per-env episode bookkeeping.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext

from ..constants import DT
from ..core.scratch import Scratch


struct OnPolicyState[
    OBS: Int,
    ACT: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    # ── Rollout buffers (ROLLOUT_LEN-sized) ─────────────────────────
    var obs_buf:  Scratch["ros_obs",  Self.ROLLOUT_LEN * Self.OBS, True]
    var act_buf:  Scratch["ros_act",  Self.ROLLOUT_LEN * Self.ACT, True]
    var olp_buf:  Scratch["ros_olp",  Self.ROLLOUT_LEN, True]
    var rew_buf:  Scratch["ros_rew",  Self.ROLLOUT_LEN, True]
    var val_buf:  Scratch["ros_val",  Self.ROLLOUT_LEN, True]
    var done_buf: Scratch["ros_done", Self.ROLLOUT_LEN, True]
    var term_buf: Scratch["ros_term", Self.ROLLOUT_LEN, True]
    var adv_buf:  Scratch["ros_adv",  Self.ROLLOUT_LEN, True]
    var ret_buf:  Scratch["ros_ret",  Self.ROLLOUT_LEN, True]

    # Bootstrap obs (the obs after the last rollout step → critic
    # forward for V(s_T)).
    var bootstrap_obs: Scratch["bootstrap_obs", Self.OBS, True]

    # ── Per-step caches (filled by act_step, consumed by record_step) ─
    var cached_action:   Scratch["cached_action", Self.ACT, True]
    var cached_log_prob: Scalar[DT]
    var cached_value:    Scalar[DT]

    # ── Per-step BATCH=1 scratches ──────────────────────────────────
    var ob1: Scratch["ob1", Self.OBS, True]
    var ao1: Scratch["ao1", 2 * Self.ACT, True]
    var v1:  Scratch["v1",  1, True]
    var z:   Scratch["z",   Self.ACT, True]

    # ── Minibatch (BATCH=MINIBATCH) scratches ───────────────────────
    var mb_obs: Scratch["mb_obs", Self.MINIBATCH * Self.OBS, True]
    var mb_act: Scratch["mb_act", Self.MINIBATCH * Self.ACT, True]
    var mb_olp: Scratch["mb_olp", Self.MINIBATCH, True]
    var mb_adv: Scratch["mb_adv", Self.MINIBATCH, True]
    var mb_ret: Scratch["mb_ret", Self.MINIBATCH * 1, True]
    var mb_v:   Scratch["mb_v",   Self.MINIBATCH * 1, True]
    var mb_gv:  Scratch["mb_gv",  Self.MINIBATCH * 1, True]
    var mb_gi:  Scratch["mb_gi",  Self.MINIBATCH * Self.OBS, True]

    # Int32 shuffle/gather index array (Scratch is DT-only, so raw ptr).
    var indices: UnsafePointer[Int32, MutAnyOrigin]

    # Rollout cursor.
    var rollout_idx: Int

    # Step bookkeeping (mirrors TrainerState's ctx field).
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.obs_buf  = Scratch["ros_obs",  Self.ROLLOUT_LEN * Self.OBS, True]()
        self.act_buf  = Scratch["ros_act",  Self.ROLLOUT_LEN * Self.ACT, True]()
        self.olp_buf  = Scratch["ros_olp",  Self.ROLLOUT_LEN, True]()
        self.rew_buf  = Scratch["ros_rew",  Self.ROLLOUT_LEN, True]()
        self.val_buf  = Scratch["ros_val",  Self.ROLLOUT_LEN, True]()
        self.done_buf = Scratch["ros_done", Self.ROLLOUT_LEN, True]()
        self.term_buf = Scratch["ros_term", Self.ROLLOUT_LEN, True]()
        self.adv_buf  = Scratch["ros_adv",  Self.ROLLOUT_LEN, True]()
        self.ret_buf  = Scratch["ros_ret",  Self.ROLLOUT_LEN, True]()
        self.bootstrap_obs = Scratch["bootstrap_obs", Self.OBS, True]()
        self.cached_action = Scratch["cached_action", Self.ACT, True]()
        self.cached_log_prob = Scalar[DT](0.0)
        self.cached_value    = Scalar[DT](0.0)
        self.ob1 = Scratch["ob1", Self.OBS, True]()
        self.ao1 = Scratch["ao1", 2 * Self.ACT, True]()
        self.v1  = Scratch["v1",  1, True]()
        self.z   = Scratch["z",   Self.ACT, True]()
        self.mb_obs = Scratch["mb_obs", Self.MINIBATCH * Self.OBS, True]()
        self.mb_act = Scratch["mb_act", Self.MINIBATCH * Self.ACT, True]()
        self.mb_olp = Scratch["mb_olp", Self.MINIBATCH, True]()
        self.mb_adv = Scratch["mb_adv", Self.MINIBATCH, True]()
        self.mb_ret = Scratch["mb_ret", Self.MINIBATCH * 1, True]()
        self.mb_v   = Scratch["mb_v",   Self.MINIBATCH * 1, True]()
        self.mb_gv  = Scratch["mb_gv",  Self.MINIBATCH * 1, True]()
        self.mb_gi  = Scratch["mb_gi",  Self.MINIBATCH * Self.OBS, True]()
        self.indices = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=0,
        )
        self.rollout_idx = 0
        self.ctx = None

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU
        (asserted P.2 — P.1 is CPU-only, so this just raises if target
        is not 'cpu')."""
        comptime assert target == "cpu" or target == "gpu", (
            "OnPolicyState: target must be 'cpu' or 'gpu'"
        )
        comptime assert target == "cpu", (
            "OnPolicyState: P.1 is CPU-only (GPU lands in P.2)"
        )
        var s = Self()
        s.obs_buf.init_with[target](ctx)
        s.act_buf.init_with[target](ctx)
        s.olp_buf.init_with[target](ctx)
        s.rew_buf.init_with[target](ctx)
        s.val_buf.init_with[target](ctx)
        s.done_buf.init_with[target](ctx)
        s.term_buf.init_with[target](ctx)
        s.adv_buf.init_with[target](ctx)
        s.ret_buf.init_with[target](ctx)
        s.bootstrap_obs.init_with[target](ctx)
        s.cached_action.init_with[target](ctx)
        s.ob1.init_with[target](ctx)
        s.ao1.init_with[target](ctx)
        s.v1.init_with[target](ctx)
        s.z.init_with[target](ctx)
        s.mb_obs.init_with[target](ctx)
        s.mb_act.init_with[target](ctx)
        s.mb_olp.init_with[target](ctx)
        s.mb_adv.init_with[target](ctx)
        s.mb_ret.init_with[target](ctx)
        s.mb_v.init_with[target](ctx)
        s.mb_gv.init_with[target](ctx)
        s.mb_gi.init_with[target](ctx)
        s.indices = alloc[Int32](Self.ROLLOUT_LEN)
        s.ctx = ctx
        return s^
