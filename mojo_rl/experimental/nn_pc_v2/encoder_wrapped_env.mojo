"""EncoderWrappedEnv — wrap a BoxContinuousActionEnv with a frozen PCEncoder.

The wrapper exposes the encoded latent `z_t` (HIDDEN-dim) as the observation
to off-the-shelf continuous-control RL algorithms (SAC, DDPG, TD3). Internally
it carries `prev_z` and `prev_action`; on each step it computes
`z_t = encoder(prev_z, prev_action, raw_obs)` and updates `prev_z := z_t`.

Designed for the Phase-2 PCN-as-representation experiment in
`docs/PCN_MBRL_PLAN.md`. The encoder is treated as frozen (no gradient
flows back from the agent through the wrapper).

All scratch and parameter buffers are caller-allocated — the wrapper holds
raw pointers and does not own/free them. This matches the testing-style
allocate-in-main pattern used elsewhere in `nn_pc_v2/`.

Layout assumption (matches PCN baseline tests):
    encoder input  = [prev_z (HIDDEN) | prev_action (ACTION_DIM) | raw_obs (OBS_DIM)]
    encoder output = z_t (HIDDEN)

ACTION_DIM and OBS_DIM are template parameters (not derived from BASE_ENV)
because trait methods on BASE_ENV that would return them aren't comptime.
Caller is responsible for matching them to the base env.
"""

from layout import Layout, LayoutTensor
from std.memory import alloc

from mojo_rl.core.env_traits import BoxContinuousActionEnv

from .pc_encoder import PCEncoder


struct EncoderWrappedEnv[
    BASE_ENV: BoxContinuousActionEnv,
    HIDDEN: Int,
    ENC_HIDDEN: Int,
    ACTION_DIM: Int,
    OBS_DIM: Int,
](BoxContinuousActionEnv):
    """Wraps a BoxContinuousActionEnv with a frozen PCEncoder.

    Replaces the base env's raw observation with `z_t = encoder(prev_z,
    prev_action, raw_obs)` at every reset/step. Drops in to any algorithm
    that only requires the BoxContinuousActionEnv interface (SAC, DDPG, TD3).
    """

    comptime ENC_INPUT_DIM: Int = Self.HIDDEN + Self.ACTION_DIM + Self.OBS_DIM
    comptime ENC_OUTPUT_DIM: Int = Self.HIDDEN
    comptime ENC = PCEncoder[Self.ENC_INPUT_DIM, Self.ENC_HIDDEN, Self.ENC_OUTPUT_DIM]

    # Trait aliases — passthrough from the base env.
    comptime dtype: DType = Self.BASE_ENV.dtype
    comptime StateType = Self.BASE_ENV.StateType
    comptime ActionType = Self.BASE_ENV.ActionType

    # Borrowed pointer to the caller's base env. Caller owns the lifecycle.
    var base_env: UnsafePointer[Self.BASE_ENV, origin=MutAnyOrigin]

    # Caller-allocated, wrapper-borrowed buffers.
    var enc_params: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var enc_input: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var enc_hpre: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var enc_hact: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var enc_output: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var prev_z: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var prev_action: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]

    # Optional input-scale divisors (caller-allocated). Wrapper divides
    # raw_obs[d] / obs_divisor[d] and action[d] / action_divisor[d] before
    # staging into the encoder input, so the wrapper matches the scales the
    # encoder was trained on. Pass all-1.0 buffers for no normalization.
    var obs_divisor: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var action_divisor: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        out self,
        *,
        base_env: UnsafePointer[Self.BASE_ENV, origin=MutAnyOrigin],
        enc_params: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        enc_input: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        enc_hpre: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        enc_hact: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        enc_output: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        prev_z: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        prev_action: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        obs_divisor: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        action_divisor: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
    ):
        """Construct the wrapper. Caller owns all the buffers and the base env."""
        self.base_env = base_env
        self.enc_params = enc_params
        self.enc_input = enc_input
        self.enc_hpre = enc_hpre
        self.enc_hact = enc_hact
        self.enc_output = enc_output
        self.prev_z = prev_z
        self.prev_action = prev_action
        self.obs_divisor = obs_divisor
        self.action_divisor = action_divisor
        # Caller is expected to zero prev_z and prev_action before the
        # first reset; reset_obs_list also zeros them on every reset.

    def __init__(out self, *, copy: Self):
        self.base_env = copy.base_env
        self.enc_params = copy.enc_params
        self.enc_input = copy.enc_input
        self.enc_hpre = copy.enc_hpre
        self.enc_hact = copy.enc_hact
        self.enc_output = copy.enc_output
        self.prev_z = copy.prev_z
        self.prev_action = copy.prev_action
        self.obs_divisor = copy.obs_divisor
        self.action_divisor = copy.action_divisor

    def __init__(out self, *, deinit take: Self):
        self.base_env = take.base_env
        self.enc_params = take.enc_params
        self.enc_input = take.enc_input
        self.enc_hpre = take.enc_hpre
        self.enc_hact = take.enc_hact
        self.enc_output = take.enc_output
        self.prev_z = take.prev_z
        self.prev_action = take.prev_action
        self.obs_divisor = take.obs_divisor
        self.action_divisor = take.action_divisor

    # No __del__ needed — base_env is borrowed (caller-owned), all other
    # fields are raw pointers (caller-owned). Wrapper owns nothing.

    # =========================================================================
    # Internal: encode the current observation given `prev_z`, `prev_action`,
    # and `raw_obs_ptr`. Updates `prev_z` in place.
    # =========================================================================

    def _encode(mut self):
        """Run encoder.forward at BATCH=1 over the staged enc_input.

        Caller must have already filled `self.enc_input` with
        `[prev_z, prev_action, raw_obs]`. Updates `prev_z` to the encoder
        output.
        """
        var enc_input_t = LayoutTensor[
            Self.dtype, Layout.row_major(1, Self.ENC_INPUT_DIM), MutAnyOrigin
        ](self.enc_input)
        var enc_hpre_t = LayoutTensor[
            Self.dtype, Layout.row_major(1, Self.ENC_HIDDEN), MutAnyOrigin
        ](self.enc_hpre)
        var enc_hact_t = LayoutTensor[
            Self.dtype, Layout.row_major(1, Self.ENC_HIDDEN), MutAnyOrigin
        ](self.enc_hact)
        var enc_output_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(1, Self.ENC_OUTPUT_DIM),
            MutAnyOrigin,
        ](self.enc_output)
        var enc_params_t = LayoutTensor[
            Self.dtype, Layout.row_major(Self.ENC.PARAM_SIZE), MutAnyOrigin
        ](self.enc_params)
        Self.ENC.forward[1, Self.dtype](
            enc_params_t, enc_input_t, enc_hpre_t, enc_hact_t, enc_output_t
        )
        for j in range(Self.HIDDEN):
            self.prev_z[j] = self.enc_output[j]

    def _stage_enc_input(mut self, raw_obs_f64: List[Float64]):
        """Build enc_input = [prev_z | prev_action | raw_obs / obs_div].

        Caller converts raw_obs to Float64 first; this avoids generic-dtype
        unification issues across BASE_ENV.dtype / Self.dtype that Mojo's
        type system doesn't fold even when they're aliased.
        """
        for j in range(Self.HIDDEN):
            self.enc_input[j] = self.prev_z[j]
        for d in range(Self.ACTION_DIM):
            self.enc_input[Self.HIDDEN + d] = self.prev_action[d]
        for d in range(Self.OBS_DIM):
            var div = Float64(self.obs_divisor[d])
            self.enc_input[Self.HIDDEN + Self.ACTION_DIM + d] = Scalar[
                Self.dtype
            ](raw_obs_f64[d] / div)

    def _latent_as_list[DTYPE_OUT: DType](self) -> List[Scalar[DTYPE_OUT]]:
        """Copy the current latent (`prev_z`) into a List of DTYPE_OUT."""
        var z = List[Scalar[DTYPE_OUT]](capacity=Self.HIDDEN)
        for j in range(Self.HIDDEN):
            z.append(Scalar[DTYPE_OUT](Float64(self.prev_z[j])))
        return z^

    def _zero_state(mut self):
        """Zero `prev_z` and `prev_action`."""
        for j in range(Self.HIDDEN):
            self.prev_z[j] = Scalar[Self.dtype](0.0)
        for d in range(Self.ACTION_DIM):
            self.prev_action[d] = Scalar[Self.dtype](0.0)

    # =========================================================================
    # Env trait methods
    # =========================================================================

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Trait method. Delegates to base env without encoding.

        SAC / DDPG / TD3 use `step_continuous_vec` instead; this exists
        only for trait conformance.
        """
        return self.base_env[].step(action, verbose)

    def reset(mut self) -> Self.StateType:
        """Trait method. Resets internal state and base env."""
        self._zero_state()
        return self.base_env[].reset()

    def get_state(self) -> Self.StateType:
        return self.base_env[].get_state()

    def close(mut self):
        self.base_env[].close()

    # =========================================================================
    # ContinuousStateEnv trait methods — observation is the encoded latent.
    # =========================================================================

    def obs_dim(self) -> Int:
        """Latent dimension (HIDDEN) is the observation dim seen by the agent.
        """
        return Self.HIDDEN

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return the most-recently-computed latent as a List."""
        var z = List[Scalar[Self.dtype]](capacity=Self.HIDDEN)
        for j in range(Self.HIDDEN):
            z.append(self.prev_z[j])
        return z^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset, encode the initial obs, return the resulting latent."""
        self._zero_state()
        var raw_obs = self.base_env[].reset_obs_list()
        var raw_f64 = List[Float64](capacity=Self.OBS_DIM)
        for d in range(Self.OBS_DIM):
            raw_f64.append(Float64(raw_obs[d]))
        self._stage_enc_input(raw_f64)
        self._encode()
        var z = List[Scalar[Self.dtype]](capacity=Self.HIDDEN)
        for j in range(Self.HIDDEN):
            z.append(self.prev_z[j])
        return z^

    # =========================================================================
    # ContinuousActionEnv trait methods — delegate to base.
    # =========================================================================

    def action_dim(self) -> Int:
        return self.base_env[].action_dim()

    def action_low(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Float64(self.base_env[].action_low()))

    def action_high(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Float64(self.base_env[].action_high()))

    # =========================================================================
    # BoxContinuousActionEnv trait methods — encode after every base step.
    # =========================================================================

    def step_continuous[
        DTYPE_SC: DType
    ](mut self, action: Scalar[DTYPE_SC]) -> Tuple[
        List[Scalar[DTYPE_SC]], Scalar[DTYPE_SC], Bool
    ]:
        """Take a 1-D continuous action; return (latent, reward, done)."""
        # Stash action at encoder-scale (raw / action_divisor).
        for d in range(Self.ACTION_DIM):
            self.prev_action[d] = Scalar[Self.dtype](0.0)
        var div0 = Float64(self.action_divisor[0])
        self.prev_action[0] = Scalar[Self.dtype](Float64(action) / div0)
        # Step base (raw action).
        var result = self.base_env[].step_continuous[DTYPE_SC](action)
        # Encode (raw_obs → Float64 → divide by obs_divisor inside _stage).
        var raw_f64 = List[Float64](capacity=Self.OBS_DIM)
        for d in range(Self.OBS_DIM):
            raw_f64.append(Float64(result[0][d]))
        self._stage_enc_input(raw_f64)
        self._encode()
        var z = self._latent_as_list[DTYPE_SC]()
        return (z^, result[1], result[2])

    def step_continuous_vec[
        DTYPE_VEC: DType
    ](
        mut self,
        action: List[Scalar[DTYPE_VEC]],
        verbose: Bool = False,
    ) -> Tuple[List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool]:
        """Take a multi-dim continuous action; return (latent, reward, done).

        Primary entry point used by SAC / DDPG / TD3 training loops.
        """
        # Stash action at encoder-scale (raw[d] / action_divisor[d]).
        for d in range(Self.ACTION_DIM):
            if d < len(action):
                var div = Float64(self.action_divisor[d])
                self.prev_action[d] = Scalar[Self.dtype](
                    Float64(action[d]) / div
                )
            else:
                self.prev_action[d] = Scalar[Self.dtype](0.0)
        # Step base (raw action).
        var result = self.base_env[].step_continuous_vec[DTYPE_VEC](
            action, verbose
        )
        # Encode (raw_obs → Float64 → divide by obs_divisor inside _stage).
        var raw_f64 = List[Float64](capacity=Self.OBS_DIM)
        for d in range(Self.OBS_DIM):
            raw_f64.append(Float64(result[0][d]))
        self._stage_enc_input(raw_f64)
        self._encode()
        var z = self._latent_as_list[DTYPE_VEC]()
        return (z^, result[1], result[2])
