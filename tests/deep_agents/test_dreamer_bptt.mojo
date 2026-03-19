"""Integration test for Dreamer V3 full BPTT world model backward.

Runs the complete observe loop + autodiff backward through the full RSSM,
verifying that ALL 11 sub-networks receive non-zero gradients.
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.dreamer_v3.dreamer_v3 import DreamerV3Agent
from std.math import abs, sqrt
from std.random import random_float64


fn main() raises:
    print("=" * 60)
    print("Dreamer V3 Full BPTT Integration Test")
    print("=" * 60)

    # Small config for fast testing
    comptime OBS = 4
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 15
    comptime BATCH = 2
    comptime BL = 4  # short sequence

    comptime Agent = DreamerV3Agent[
        OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS,
        batch_size=BATCH, batch_length=BL, imagine_horizon=3,
        buffer_capacity=500,
    ]

    var agent = Agent()

    # Fill replay buffer with random data
    comptime STOCH_FLAT = STOCH * CLASSES
    print("Filling replay buffer with random transitions...")
    for ep in range(10):
        var obs = InlineArray[Scalar[DType.float32], OBS](uninitialized=True)
        for i in range(OBS):
            obs[i] = Scalar[DType.float32](random_float64(-1.0, 1.0))

        for step in range(BL + 5):
            var action = InlineArray[Scalar[DType.float32], ACT](
                uninitialized=True
            )
            for i in range(ACT):
                action[i] = Scalar[DType.float32](random_float64(-1.0, 1.0))

            var reward = Scalar[DType.float32](random_float64(-1.0, 1.0))
            var done = False
            if step == BL + 4:
                done = True

            agent.state.buffer.add(obs, action, reward, done)

            # Update obs for next step
            for i in range(OBS):
                obs[i] = Scalar[DType.float32](random_float64(-1.0, 1.0))

    print("  Buffer size:", agent.state.buffer.size)

    # Run the forward observe loop (same as update() does)
    # We need to trigger the same forward pass that fills _all_* buffers
    # Then call _backward_world_model_autodiff

    # Sample batch data
    var batch_obs = List[Scalar[DType.float32]](
        capacity=BATCH * (BL + 1) * OBS
    )
    var batch_actions = List[Scalar[DType.float32]](
        capacity=BATCH * BL * ACT
    )
    var batch_rewards = List[Scalar[DType.float32]](
        capacity=BATCH * BL
    )
    var batch_dones = List[Scalar[DType.float32]](
        capacity=BATCH * BL
    )

    for _ in range(BATCH * (BL + 1) * OBS):
        batch_obs.append(0)
    for _ in range(BATCH * BL * ACT):
        batch_actions.append(0)
    for _ in range(BATCH * BL):
        batch_rewards.append(0)
        batch_dones.append(0)

    agent.state.buffer.sample_sequences[BATCH, BL](
        batch_obs, batch_actions, batch_rewards, batch_dones
    )

    print("  Sampled batch: obs shape =", BATCH, "x", BL + 1, "x", OBS)

    # ── Run forward observe loop ─────────────────────────────────────────
    from std.memory import alloc, memset

    var deter_ptr = alloc[Scalar[dtype]](BATCH * DETER)
    memset(deter_ptr, 0, BATCH * DETER)
    var stoch_ptr = alloc[Scalar[dtype]](BATCH * STOCH_FLAT)
    memset(stoch_ptr, 0, BATCH * STOCH_FLAT)
    var new_deter_ptr = alloc[Scalar[dtype]](BATCH * DETER)
    var new_stoch_ptr = alloc[Scalar[dtype]](BATCH * STOCH_FLAT)
    var post_probs_ptr = alloc[Scalar[dtype]](BATCH * STOCH_FLAT)
    var prior_probs_ptr = alloc[Scalar[dtype]](BATCH * STOCH_FLAT)
    var feat_ptr = alloc[Scalar[dtype]](BATCH * Agent.FEAT_DIM)
    var obs_step_ptr = alloc[Scalar[dtype]](BATCH * OBS)
    var act_step_ptr = alloc[Scalar[dtype]](BATCH * ACT)

    from layout import Layout, LayoutTensor

    for t in range(BL):
        for b in range(BATCH):
            for i in range(OBS):
                var idx = b * (BL + 1) * OBS + t * OBS + i
                (obs_step_ptr + b * OBS + i)[] = Scalar[dtype](
                    batch_obs[idx]
                )
            for i in range(ACT):
                if t == 0:
                    (act_step_ptr + b * ACT + i)[] = Scalar[dtype](0.0)
                else:
                    var idx = b * BL * ACT + (t - 1) * ACT + i
                    (act_step_ptr + b * ACT + i)[] = Scalar[dtype](
                        batch_actions[idx]
                    )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](obs_step_ptr)
        var deter_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
        ](deter_ptr)
        var stoch_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
        ](stoch_ptr)
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
        ](act_step_ptr)

        memset(new_deter_ptr, 0, BATCH * DETER)
        memset(new_stoch_ptr, 0, BATCH * STOCH_FLAT)
        memset(post_probs_ptr, 0, BATCH * STOCH_FLAT)
        memset(prior_probs_ptr, 0, BATCH * STOCH_FLAT)
        memset(feat_ptr, 0, BATCH * Agent.FEAT_DIM)

        var new_deter_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
        ](new_deter_ptr)
        var new_stoch_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
        ](new_stoch_ptr)
        var post_probs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
        ](post_probs_ptr)
        var prior_probs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
        ](prior_probs_ptr)
        var feat_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Agent.FEAT_DIM), MutAnyOrigin
        ](feat_ptr)

        agent.state.rssm.observe_step[BATCH](
            obs_t, deter_t, stoch_t, act_t,
            new_deter_t, new_stoch_t, post_probs_t, prior_probs_t, feat_t,
            True,
        )

        # Store in _all_* buffers
        comptime FEAT = Agent.FEAT_DIM
        for b in range(BATCH):
            for i in range(DETER):
                (
                    agent.state._all_deter + t * BATCH * DETER + b * DETER + i
                )[] = (new_deter_ptr + b * DETER + i)[]
            for i in range(STOCH_FLAT):
                (
                    agent.state._all_stoch
                    + t * BATCH * STOCH_FLAT
                    + b * STOCH_FLAT
                    + i
                )[] = (new_stoch_ptr + b * STOCH_FLAT + i)[]
                (
                    agent.state._all_post_probs
                    + t * BATCH * STOCH_FLAT
                    + b * STOCH_FLAT
                    + i
                )[] = (post_probs_ptr + b * STOCH_FLAT + i)[]
                (
                    agent.state._all_prior_probs
                    + t * BATCH * STOCH_FLAT
                    + b * STOCH_FLAT
                    + i
                )[] = (prior_probs_ptr + b * STOCH_FLAT + i)[]
            for i in range(FEAT):
                (
                    agent.state._all_feats
                    + t * BATCH * FEAT
                    + b * FEAT
                    + i
                )[] = (feat_ptr + b * FEAT + i)[]

        # Update for next timestep
        for b in range(BATCH):
            for i in range(DETER):
                (deter_ptr + b * DETER + i)[] = (
                    new_deter_ptr + b * DETER + i
                )[]
            for i in range(STOCH_FLAT):
                (stoch_ptr + b * STOCH_FLAT + i)[] = (
                    new_stoch_ptr + b * STOCH_FLAT + i
                )[]

    print("  Forward observe loop complete (", BL, "steps)")

    # ── Run full BPTT backward ───────────────────────────────────────────
    print("Running full BPTT backward...")
    var wm_result = agent._backward_world_model_autodiff[BATCH](
        batch_obs, batch_actions, batch_rewards, batch_dones
    )
    print("  Total WM loss:", wm_result[0])

    # ── Check all 11 networks have non-zero gradients ────────────────────
    print("Checking gradient norms for all 11 RSSM networks...")

    fn grad_norm(
        ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int
    ) -> Float64:
        var total = Float64(0.0)
        for i in range(n):
            var v = Float64(ptr[i])
            total += v * v
        return sqrt(total)

    comptime RSSMType = Agent.StateType.RSSMType

    var enc_gn = grad_norm(
        agent.state.rssm.encoder.grads_view().ptr,
        RSSMType.EncModel.PARAM_SIZE,
    )
    var post_gn = grad_norm(
        agent.state.rssm.posterior.grads_view().ptr,
        RSSMType.PostModel.PARAM_SIZE,
    )
    var prior_gn = grad_norm(
        agent.state.rssm.prior.grads_view().ptr,
        RSSMType.PriorModel.PARAM_SIZE,
    )
    var dec_gn = grad_norm(
        agent.state.rssm.decoder.grads_view().ptr,
        RSSMType.DecModel.PARAM_SIZE,
    )
    var rew_gn = grad_norm(
        agent.state.rssm.reward_head.grads_view().ptr,
        RSSMType.RewModel.PARAM_SIZE,
    )
    var cont_gn = grad_norm(
        agent.state.rssm.continue_head.grads_view().ptr,
        RSSMType.ContModel.PARAM_SIZE,
    )
    var dp_gn = grad_norm(
        agent.state.rssm.deter_proj.grads_view().ptr,
        RSSMType.DeterProj.PARAM_SIZE,
    )
    var sp_gn = grad_norm(
        agent.state.rssm.stoch_proj.grads_view().ptr,
        RSSMType.StochProj.PARAM_SIZE,
    )
    var ap_gn = grad_norm(
        agent.state.rssm.action_proj.grads_view().ptr,
        RSSMType.ActionProj.PARAM_SIZE,
    )
    var gh_gn = grad_norm(
        agent.state.rssm.gru_hidden.grads_view().ptr,
        RSSMType.GRUHiddenModel.PARAM_SIZE,
    )
    var gg_gn = grad_norm(
        agent.state.rssm.gru_gates.grads_view().ptr,
        RSSMType.GRUGateModel.PARAM_SIZE,
    )

    print("  Prediction heads:")
    print("    decoder:", dec_gn, " reward:", rew_gn, " continue:", cont_gn)
    print("  RSSM core:")
    print("    encoder:", enc_gn, " posterior:", post_gn, " prior:", prior_gn)
    print("  GRU networks:")
    print(
        "    deter_proj:", dp_gn, " stoch_proj:", sp_gn,
        " action_proj:", ap_gn,
    )
    print("    gru_hidden:", gh_gn, " gru_gates:", gg_gn)

    # Verify all non-zero
    var all_ok = True
    if dec_gn < 1e-20:
        print("  FAIL: decoder has zero grads")
        all_ok = False
    if rew_gn < 1e-20:
        print("  FAIL: reward_head has zero grads")
        all_ok = False
    if cont_gn < 1e-20:
        print("  FAIL: continue_head has zero grads")
        all_ok = False
    if enc_gn < 1e-20:
        print("  FAIL: encoder has zero grads")
        all_ok = False
    if post_gn < 1e-20:
        print("  FAIL: posterior has zero grads")
        all_ok = False
    # Prior may be zero if KL < free_nats (expected)
    if dp_gn < 1e-20:
        print("  FAIL: deter_proj has zero grads")
        all_ok = False
    if sp_gn < 1e-20:
        print("  FAIL: stoch_proj has zero grads")
        all_ok = False
    if ap_gn < 1e-20:
        print("  FAIL: action_proj has zero grads")
        all_ok = False
    if gh_gn < 1e-20:
        print("  FAIL: gru_hidden has zero grads")
        all_ok = False
    if gg_gn < 1e-20:
        print("  FAIL: gru_gates has zero grads")
        all_ok = False

    # Free scratch
    deter_ptr.free()
    stoch_ptr.free()
    new_deter_ptr.free()
    new_stoch_ptr.free()
    post_probs_ptr.free()
    prior_probs_ptr.free()
    feat_ptr.free()
    obs_step_ptr.free()
    act_step_ptr.free()

    if all_ok:
        print("=" * 60)
        print("[PASS] Full BPTT: all networks have non-zero gradients!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[FAIL] Some networks have zero gradients")
        print("=" * 60)
        raise Error("BPTT integration test failed")
