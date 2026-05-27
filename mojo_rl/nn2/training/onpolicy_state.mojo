"""OnPolicyState — shared per-step flow object for on-policy block trainers.

Analog to `trainer_block.mojo`'s `TrainerState` (off-policy). Carries:
  - the full rollout buffers (obs/act/olp/rew/val/done/term/adv/ret)
    sized [ROLLOUT_LEN, N_ENVS], written by record_step + gae_step,
    read by minibatch_gather_step
  - the per-step caches (action / log_prob / value) sized N_ENVS,
    filled by act_step, drained by record_step
  - the per-step BATCH=N_ENVS scratches (ob1/ao1/v1/z) for batched
    actor + critic forward calls during action selection
  - the minibatch BATCH=MINIBATCH scratches (mb_obs/mb_act/mb_olp/mb_adv/
    mb_ret/mb_v/mb_gv/mb_gi) written by gather, consumed by actor/critic
    train steps. Minibatch is gathered from the [ROLLOUT_LEN, N_ENVS]
    flat pool (size ROLLOUT_LEN * N_ENVS), so MINIBATCH layout is
    unchanged from the N_ENVS=1 case.
  - rollout cursor (single Int — all envs advance synchronously) + ctx
  - Int32 indices for the Fisher-Yates shuffle, sized
    ROLLOUT_LEN * N_ENVS (flat index into the rollout pool)

N_ENVS defaults to 1 so existing single-env callers stay bit-identical
without code changes. P.3 enables N_ENVS > 1; the BatchedEnv driver
analog of off-policy's Tier-3 lives in `driver_onpolicy.mojo`.
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
    N_ENVS: Int = 1,
](Defaultable & Movable & ImplicitlyDestructible):
    # ── Rollout buffers (ROLLOUT_LEN × N_ENVS, T-major) ─────────────
    var obs_buf:  Scratch["ros_obs",  Self.ROLLOUT_LEN * Self.N_ENVS * Self.OBS, True]
    var act_buf:  Scratch["ros_act",  Self.ROLLOUT_LEN * Self.N_ENVS * Self.ACT, True]
    var olp_buf:  Scratch["ros_olp",  Self.ROLLOUT_LEN * Self.N_ENVS, True]
    var rew_buf:  Scratch["ros_rew",  Self.ROLLOUT_LEN * Self.N_ENVS, True]
    var val_buf:  Scratch["ros_val",  Self.ROLLOUT_LEN * Self.N_ENVS, True]
    var done_buf: Scratch["ros_done", Self.ROLLOUT_LEN * Self.N_ENVS, True]
    var term_buf: Scratch["ros_term", Self.ROLLOUT_LEN * Self.N_ENVS, True]
    var adv_buf:  Scratch["ros_adv",  Self.ROLLOUT_LEN * Self.N_ENVS, True]
    var ret_buf:  Scratch["ros_ret",  Self.ROLLOUT_LEN * Self.N_ENVS, True]

    # Bootstrap obs per env (the obs after the last rollout step →
    # critic forward for V(s_T)).
    var bootstrap_obs: Scratch["bootstrap_obs", Self.N_ENVS * Self.OBS, True]

    # ── Per-step caches per env (filled by act, drained by record) ──
    var cached_action: Scratch["cached_action", Self.N_ENVS * Self.ACT, True]
    var cached_log_prob: Scratch["cached_log_prob", Self.N_ENVS, True]
    var cached_value:    Scratch["cached_value",    Self.N_ENVS, True]

    # ── Per-step BATCH=N_ENVS scratches ─────────────────────────────
    var ob1: Scratch["ob1", Self.N_ENVS * Self.OBS, True]
    var ao1: Scratch["ao1", Self.N_ENVS * 2 * Self.ACT, True]
    var v1:  Scratch["v1",  Self.N_ENVS, True]
    var z:   Scratch["z",   Self.N_ENVS * Self.ACT, True]

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
        self.obs_buf  = Scratch["ros_obs",  Self.ROLLOUT_LEN * Self.N_ENVS * Self.OBS, True]()
        self.act_buf  = Scratch["ros_act",  Self.ROLLOUT_LEN * Self.N_ENVS * Self.ACT, True]()
        self.olp_buf  = Scratch["ros_olp",  Self.ROLLOUT_LEN * Self.N_ENVS, True]()
        self.rew_buf  = Scratch["ros_rew",  Self.ROLLOUT_LEN * Self.N_ENVS, True]()
        self.val_buf  = Scratch["ros_val",  Self.ROLLOUT_LEN * Self.N_ENVS, True]()
        self.done_buf = Scratch["ros_done", Self.ROLLOUT_LEN * Self.N_ENVS, True]()
        self.term_buf = Scratch["ros_term", Self.ROLLOUT_LEN * Self.N_ENVS, True]()
        self.adv_buf  = Scratch["ros_adv",  Self.ROLLOUT_LEN * Self.N_ENVS, True]()
        self.ret_buf  = Scratch["ros_ret",  Self.ROLLOUT_LEN * Self.N_ENVS, True]()
        self.bootstrap_obs = Scratch["bootstrap_obs", Self.N_ENVS * Self.OBS, True]()
        self.cached_action = Scratch["cached_action", Self.N_ENVS * Self.ACT, True]()
        self.cached_log_prob = Scratch["cached_log_prob", Self.N_ENVS, True]()
        self.cached_value    = Scratch["cached_value",    Self.N_ENVS, True]()
        self.ob1 = Scratch["ob1", Self.N_ENVS * Self.OBS, True]()
        self.ao1 = Scratch["ao1", Self.N_ENVS * 2 * Self.ACT, True]()
        self.v1  = Scratch["v1",  Self.N_ENVS, True]()
        self.z   = Scratch["z",   Self.N_ENVS * Self.ACT, True]()
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
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU.

        All Scratches are STAGING=True, so on GPU both the host mirror
        and device buffer are allocated. P.2 hybrid: rollout / per-step
        / minibatch scratches read and write on the host mirror; the
        minibatch is H2D-uploaded before actor/critic train steps."""
        comptime assert target == "cpu" or target == "gpu", (
            "OnPolicyState: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("OnPolicyState.make[target='gpu']: ctx required")
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
        s.cached_log_prob.init_with[target](ctx)
        s.cached_value.init_with[target](ctx)
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
        s.indices = alloc[Int32](Self.ROLLOUT_LEN * Self.N_ENVS)
        s.ctx = ctx
        return s^
