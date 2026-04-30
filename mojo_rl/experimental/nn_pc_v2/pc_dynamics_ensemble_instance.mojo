"""PCDynamicsEnsembleInstanceCPU — owning CPU wrapper for PCN dynamics ensemble.

The static-namespace `PCDynamicsEnsemble` requires the caller to allocate
all buffers. That works for a one-off test but not for an agent struct
that wants to declare `var dynamics: ...` and have the buffers carried
along. This wrapper owns the buffers (params, grads, Adam state, scratch)
and exposes the `MBPOAgent`-style methods:

- `train_model[CAP](buffer)`  — train all members on transitions sampled
                                 from a HeapReplayBuffer. Updates
                                 `elite_indices` from holdout MSE.
- `predict_single(obs, action, elite_idx, out_next, out_reward)`
- `elite_indices`             — public, indexable.

API matches `DynamicsEnsemble` (vanilla MBPO) closely enough that the
PCN-MBPO agent fork can be a near-mechanical type rename.

Design notes
------------
- `Owns` is fixed to one Adam optimizer at construction (default Adam).
  If we need multiple, we'll templatize later — agent only uses one.
- Params normalization (s, a) and target shape (delta_obs, reward / scale)
  are caller responsibilities, mirroring the today's CPU PCN-MBPO test
  pattern. Same convention as our existing test file.
"""

from layout import Layout, LayoutTensor
from std.math import sqrt
from std.memory import alloc, memset
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom

from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.checkpoint import (
    write_float_section_ptr,
    read_float_section_list,
)
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer

from .pc_dynamics import PCDynamics
from .pc_dynamics_ensemble import PCDynamicsEnsemble
from .pc_utils import clip_grad_norm


struct PCDynamicsEnsembleInstanceCPU[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    HIDDEN_DIM: Int = 200,
    NUM_ENSEMBLE: Int = 3,
    NUM_ELITES: Int = 2,
    DYN_BATCH: Int = 64,
    T_INFER: Int = 10,
    LR_X_FLOAT: Float64 = 0.01,
    DYN_LR: Float64 = 0.001,
    GRAD_CLIP_NORM: Float64 = 1.0,
    OBS_REWARD_SCALE: Float64 = 10.0,
    dtype: DType = DType.float32,
](Movable):
    """Owning CPU wrapper around `PCDynamicsEnsemble`.

    Drop-in replacement for vanilla MBPO's `DynamicsEnsemble` field on
    `MBPOCPUState`. Same method shapes (`train_model`, `predict_single`,
    `elite_indices`), different internal training procedure (PC weight
    rule + SGLD instead of Gaussian-NLL backprop).
    """

    comptime ENS = PCDynamicsEnsemble[
        Self.OBS_DIM,
        Self.ACTION_DIM,
        Self.HIDDEN_DIM,
        Self.NUM_ENSEMBLE,
        Self.NUM_ELITES,
        Self.dtype,
    ]
    comptime DYN = Self.ENS.DYN
    comptime OPT = Adam[LR=Self.DYN_LR]

    # Owned buffers — one big block per kind, member m at offset
    # `m * PER_MEMBER_PARAM_SIZE` (or appropriate for opt state).
    var params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var grads_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var opt_state_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var opt_global_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]

    # Per-member Adam step counters.
    var step_nums: List[Int]

    # Elite indices — top NUM_ELITES members by holdout MSE.
    var elite_indices: List[Int]

    # Shared SGLD scratch (DYN_BATCH-sized; reused across members).
    var lat_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var mu_eps_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var a_below_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var z_below_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var dx_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]

    # Dyn-input/target staging buffers (DYN_BATCH-sized).
    var s_a_batch: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var target_batch: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]

    # Eval/predict-1 scratch (BATCH=1).
    var p_a_aug: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var p_z: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var p_a_z: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var p_out: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var p_s_a: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]

    # Holdout scratch (DYN_BATCH-sized eval).
    var e_a_aug: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var e_z: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var e_a_z: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var e_out: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]

    # Per-instance RNG for batch sampling (separate from global random).
    var rng_state: UInt64

    # =========================================================================
    # Construction / destruction.
    # =========================================================================

    def __init__(out self, base_seed: UInt64 = UInt64(7)):
        """Allocate all buffers, init members with independent Xavier seeds."""
        self.params_buf = alloc[Scalar[Self.dtype]](Self.ENS.TOTAL_PARAM_SIZE)
        self.grads_buf = alloc[Scalar[Self.dtype]](Self.ENS.TOTAL_PARAM_SIZE)
        self.opt_state_buf = alloc[Scalar[Self.dtype]](
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM
        )
        self.opt_global_buf = alloc[Scalar[Self.dtype]](
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE
        )
        memset(self.params_buf, 0, Self.ENS.TOTAL_PARAM_SIZE)
        memset(self.grads_buf, 0, Self.ENS.TOTAL_PARAM_SIZE)
        memset(
            self.opt_state_buf, 0,
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM,
        )
        memset(
            self.opt_global_buf, 0,
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE,
        )
        Self.ENS.init_all(self.params_buf, base_seed=base_seed)

        self.step_nums = List[Int](capacity=Self.NUM_ENSEMBLE)
        for _ in range(Self.NUM_ENSEMBLE):
            self.step_nums.append(0)

        # Initially all members are elite.
        self.elite_indices = List[Int](capacity=Self.NUM_ELITES)
        for i in range(Self.NUM_ELITES):
            self.elite_indices.append(i)

        # SGLD scratch.
        self.lat_buf = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.SCRATCH_LAT
        )
        self.mu_eps_buf = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.SCRATCH_OUT
        )
        self.a_below_buf = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.SCRATCH_IN
        )
        self.z_below_buf = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.SCRATCH_IN
        )
        self.dx_buf = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.SCRATCH_LAT
        )
        memset(self.lat_buf, 0, Self.DYN_BATCH * Self.DYN.SCRATCH_LAT)
        memset(self.mu_eps_buf, 0, Self.DYN_BATCH * Self.DYN.SCRATCH_OUT)
        memset(self.a_below_buf, 0, Self.DYN_BATCH * Self.DYN.SCRATCH_IN)
        memset(self.z_below_buf, 0, Self.DYN_BATCH * Self.DYN.SCRATCH_IN)
        memset(self.dx_buf, 0, Self.DYN_BATCH * Self.DYN.SCRATCH_LAT)

        # Batch staging.
        self.s_a_batch = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.AUG_DIM
        )
        self.target_batch = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.READOUT
        )

        # Predict-1 scratch.
        self.p_a_aug = alloc[Scalar[Self.dtype]](Self.DYN.AUG_DIM)
        self.p_z = alloc[Scalar[Self.dtype]](Self.DYN.HIDDEN_DIM)
        self.p_a_z = alloc[Scalar[Self.dtype]](Self.DYN.HIDDEN_DIM)
        self.p_out = alloc[Scalar[Self.dtype]](Self.DYN.READOUT)
        self.p_s_a = alloc[Scalar[Self.dtype]](Self.DYN.AUG_DIM)

        # Holdout scratch.
        self.e_a_aug = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.AUG_DIM
        )
        self.e_z = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.HIDDEN_DIM
        )
        self.e_a_z = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.HIDDEN_DIM
        )
        self.e_out = alloc[Scalar[Self.dtype]](
            Self.DYN_BATCH * Self.DYN.READOUT
        )

        self.rng_state = base_seed + UInt64(1000)

    def __init__(out self, *, deinit take: Self):
        self.params_buf = take.params_buf
        self.grads_buf = take.grads_buf
        self.opt_state_buf = take.opt_state_buf
        self.opt_global_buf = take.opt_global_buf
        self.step_nums = take.step_nums^
        self.elite_indices = take.elite_indices^
        self.lat_buf = take.lat_buf
        self.mu_eps_buf = take.mu_eps_buf
        self.a_below_buf = take.a_below_buf
        self.z_below_buf = take.z_below_buf
        self.dx_buf = take.dx_buf
        self.s_a_batch = take.s_a_batch
        self.target_batch = take.target_batch
        self.p_a_aug = take.p_a_aug
        self.p_z = take.p_z
        self.p_a_z = take.p_a_z
        self.p_out = take.p_out
        self.p_s_a = take.p_s_a
        self.e_a_aug = take.e_a_aug
        self.e_z = take.e_z
        self.e_a_z = take.e_a_z
        self.e_out = take.e_out
        self.rng_state = take.rng_state

    fn __del__(deinit self):
        self.params_buf.free()
        self.grads_buf.free()
        self.opt_state_buf.free()
        self.opt_global_buf.free()
        self.lat_buf.free()
        self.mu_eps_buf.free()
        self.a_below_buf.free()
        self.z_below_buf.free()
        self.dx_buf.free()
        self.s_a_batch.free()
        self.target_batch.free()
        self.p_a_aug.free()
        self.p_z.free()
        self.p_a_z.free()
        self.p_out.free()
        self.p_s_a.free()
        self.e_a_aug.free()
        self.e_z.free()
        self.e_a_z.free()
        self.e_out.free()

    # =========================================================================
    # MBPO agent's contract: predict (obs, action) → (next_obs, reward) for
    # one elite member. No noise (PCN ensemble is deterministic; variance
    # comes from ensemble disagreement).
    # =========================================================================

    def predict_single(
        mut self,
        obs: List[Scalar[Self.dtype]],
        action: List[Scalar[Self.dtype]],
        elite_idx: Int,
        mut out_next_obs: List[Scalar[Self.dtype]],
        mut out_reward: List[Scalar[Self.dtype]],
    ):
        """Predict (next_obs, reward) using one elite ensemble member.

        Action and obs come in env-native scale; the dynamics's input is
        normalized (matches the today's CPU PCN-MBPO test convention):
        the caller's `action_scale` and the wrapper's static
        `OBS_REWARD_SCALE` handle scaling. Output `next_obs` is in
        env-native scale; `out_reward` is also un-scaled.

        For Pendulum the convention used elsewhere divides obs[2]/8.0 and
        action[0]/2.0 — the agent's `do_model_rollouts` should pass
        already-normalized inputs here. This method is shape-only; it
        does not impose any normalization itself, leaving that to caller.
        """
        var member_idx = self.elite_indices[elite_idx]
        # Stage [obs | action] into p_s_a.
        for d in range(Self.OBS_DIM):
            self.p_s_a[d] = obs[d]
        for d in range(Self.ACTION_DIM):
            self.p_s_a[Self.OBS_DIM + d] = action[d]
        # Predict via member m (BATCH=1 view).
        var p_s_a_t = LayoutTensor[
            Self.dtype, Layout.row_major(1, Self.DYN.AUG_DIM), MutAnyOrigin
        ](self.p_s_a)
        var p_out_t = LayoutTensor[
            Self.dtype, Layout.row_major(1, Self.DYN.READOUT), MutAnyOrigin
        ](self.p_out)
        Self.ENS.predict_member[1](
            member_idx, p_s_a_t, self.params_buf, p_out_t,
        )
        # Output: out[0:OBS_DIM] = predicted delta_obs (residual), out[OBS_DIM] = reward.
        # `predict_single`'s contract on vanilla MBPO returns absolute next_obs
        # (residual added to obs); mirror that here. Reward is passed through.
        out_next_obs.clear()
        for d in range(Self.OBS_DIM):
            var delta = Float64(self.p_out[d])
            out_next_obs.append(
                Scalar[Self.dtype](Float64(obs[d]) + delta)
            )
        out_reward.clear()
        out_reward.append(self.p_out[Self.OBS_DIM])

    # =========================================================================
    # MBPO agent's contract: train_model[CAP](real_buffer).
    # =========================================================================

    def train_model[
        buffer_capacity: Int,
    ](
        mut self,
        buffer: HeapReplayBuffer[
            buffer_capacity, Self.OBS_DIM, Self.ACTION_DIM, Self.dtype
        ],
        n_minibatches: Int = 30,
    ) raises:
        """Train all NUM_ENSEMBLE members on minibatches sampled from `buffer`.

        Mirrors `DynamicsEnsemble.train_model` shape so the agent's
        `train_dynamics(cpu_state)` call site doesn't need to change.

        Variances are not predicted (no NLL); we use MSE on
        (delta_obs, reward) targets and per-block PC weight gradients
        with SGLD inference for the latent z.

        After training, refreshes `elite_indices` from a fresh holdout
        batch's per-member MSE (top NUM_ELITES, lowest loss).
        """
        if buffer.size < Self.DYN_BATCH:
            return

        var rng = PhiloxRandom(seed=self.rng_state, offset=UInt64(0))

        # Per-member training.
        for m in range(Self.NUM_ENSEMBLE):
            for _ in range(n_minibatches):
                Self._build_batch_from_buffer[buffer_capacity](
                    rng, buffer,
                    self.s_a_batch, self.target_batch,
                )
                var s_a_t = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
                    MutAnyOrigin,
                ](self.s_a_batch)
                var target_t = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(Self.DYN_BATCH, Self.DYN.READOUT),
                    MutAnyOrigin,
                ](self.target_batch)
                var lat_t = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
                    MutAnyOrigin,
                ](self.lat_buf)
                var mu_eps_t = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_OUT),
                    MutAnyOrigin,
                ](self.mu_eps_buf)
                var a_below_t = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
                    MutAnyOrigin,
                ](self.a_below_buf)
                var z_below_t = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
                    MutAnyOrigin,
                ](self.z_below_buf)
                var dx_t = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
                    MutAnyOrigin,
                ](self.dx_buf)
                _ = Self.ENS.train_member[Self.DYN_BATCH, Self.OPT](
                    m, s_a_t, target_t,
                    self.params_buf, self.grads_buf,
                    self.opt_state_buf, self.opt_global_buf,
                    lat_t, mu_eps_t, a_below_t, z_below_t, dx_t,
                    self.step_nums[m],
                    T_infer=Self.T_INFER,
                    lr_x=Scalar[Self.dtype](Self.LR_X_FLOAT),
                    grad_clip_norm=Self.GRAD_CLIP_NORM,
                )

        # Refresh elite indices from a fresh holdout batch.
        Self._build_batch_from_buffer[buffer_capacity](
            rng, buffer, self.s_a_batch, self.target_batch
        )
        var s_a_h = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.s_a_batch)
        var target_h = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](self.target_batch)
        var e_a_aug_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.e_a_aug)
        var e_z_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ](self.e_z)
        var e_a_z_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ](self.e_a_z)
        var e_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](self.e_out)
        var holdout_losses = List[Float64](capacity=Self.NUM_ENSEMBLE)
        for m in range(Self.NUM_ENSEMBLE):
            var L = Self.ENS.eval_member_loss[Self.DYN_BATCH](
                m, s_a_h, target_h, self.params_buf, e_out_t,
            )
            holdout_losses.append(L)
        Self.ENS.select_elites(holdout_losses, self.elite_indices)

        # Persist RNG state for next call.
        self.rng_state += UInt64(n_minibatches * Self.NUM_ENSEMBLE)

    # =========================================================================
    # Internal: sample a (s_a, target) minibatch from the agent's real buffer.
    # `target` = (delta_obs, reward) — caller's MBPO loop predicts deltas.
    # =========================================================================

    @staticmethod
    fn _build_batch_from_buffer[
        buffer_capacity: Int,
    ](
        mut rng: PhiloxRandom,
        buffer: HeapReplayBuffer[
            buffer_capacity, Self.OBS_DIM, Self.ACTION_DIM, Self.dtype
        ],
        s_a_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        target_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
    ):
        """Sample DYN_BATCH random transitions from `buffer`. Build:

            s_a    ← [obs | action]                         (raw, env-native)
            target ← [next_obs - obs | reward]              (raw deltas)

        No normalization here — caller decides whether to normalize before
        the buffer (preferred) or post-hoc.
        """
        var n = buffer.size
        for b in range(Self.DYN_BATCH):
            var u = Float64(rng.step_uniform()[0])
            var idx = Int(u * Float64(n)) % n
            for d in range(Self.OBS_DIM):
                s_a_buf[b * Self.DYN.AUG_DIM + d] = buffer.obs[
                    idx * Self.OBS_DIM + d
                ]
            for d in range(Self.ACTION_DIM):
                s_a_buf[b * Self.DYN.AUG_DIM + Self.OBS_DIM + d] = (
                    buffer.actions[idx * Self.ACTION_DIM + d]
                )
            for d in range(Self.OBS_DIM):
                var s_d = Float64(buffer.obs[idx * Self.OBS_DIM + d])
                var sn_d = Float64(buffer.next_obs[idx * Self.OBS_DIM + d])
                target_buf[b * Self.DYN.READOUT + d] = Scalar[Self.dtype](
                    sn_d - s_d
                )
            target_buf[b * Self.DYN.READOUT + Self.OBS_DIM] = (
                buffer.rewards[idx]
            )

    # =========================================================================
    # Checkpoint surface — mirrors `NetworkState.write_sections` /
    # `read_sections`, but covers the whole ensemble in a single call (one
    # `params:` / `opt_state:` / `opt_global:` section blob keyed by `prefix`).
    # The fork's checkpoint code becomes a single line per ensemble instead
    # of the per-member loop in vanilla MBPO.
    # =========================================================================

    def write_sections(self, prefix: String) -> String:
        """Serialize ensemble (params + Adam state + step counters) as text sections.

        Sections written (each is `prefix + "<name>:"` followed by one
        float per line):

        - `params:`     — `TOTAL_PARAM_SIZE` floats covering all members
                          back-to-back at member-stride `PER_MEMBER_PARAM_SIZE`.
        - `opt_state:`  — `TOTAL_PARAM_SIZE * STATE_PER_PARAM` floats
                          (Adam m/v moments, per-member).
        - `opt_global:` — `NUM_ENSEMBLE * GLOBAL_STATE_SIZE` floats
                          (per-member step counter + lr_scale).
        - `step_nums:`  — `NUM_ENSEMBLE` integers (host-side mirrors of the
                          on-device step counters; encoded as floats).

        Elite indices are NOT written here — the agent persists them in its
        metadata block (matches vanilla MBPO).
        """
        var content = write_float_section_ptr(
            prefix + "params:", self.params_buf, Self.ENS.TOTAL_PARAM_SIZE
        )
        content += write_float_section_ptr(
            prefix + "opt_state:",
            self.opt_state_buf,
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM,
        )
        content += write_float_section_ptr(
            prefix + "opt_global:",
            self.opt_global_buf,
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE,
        )
        var steps = prefix + "step_nums:\n"
        for m in range(Self.NUM_ENSEMBLE):
            steps += String(self.step_nums[m]) + "\n"
        content += steps
        return content

    def read_sections(mut self, content: String, prefix: String) raises:
        """Restore ensemble from sections written by `write_sections`."""
        var loaded_params = read_float_section_list[Self.dtype](
            content, prefix + "params:", Self.ENS.TOTAL_PARAM_SIZE
        )
        for i in range(Self.ENS.TOTAL_PARAM_SIZE):
            (self.params_buf + i)[] = loaded_params[i]

        var loaded_opt = read_float_section_list[Self.dtype](
            content,
            prefix + "opt_state:",
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM,
        )
        for i in range(Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM):
            (self.opt_state_buf + i)[] = loaded_opt[i]

        var loaded_global = read_float_section_list[Self.dtype](
            content,
            prefix + "opt_global:",
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE,
        )
        for i in range(Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE):
            (self.opt_global_buf + i)[] = loaded_global[i]

        var loaded_steps = read_float_section_list[Self.dtype](
            content, prefix + "step_nums:", Self.NUM_ENSEMBLE
        )
        for m in range(Self.NUM_ENSEMBLE):
            self.step_nums[m] = Int(Float64(loaded_steps[m]))
