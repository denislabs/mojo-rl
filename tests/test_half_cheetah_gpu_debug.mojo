"""HalfCheetah GPU Debug Diagnostics.

Probes physics, action application, observations, and rewards step-by-step
to diagnose why training reward is stuck at -210.

Sections:
  1. Initial state after reset (are qpos/qvel reasonable?)
  2. 30 steps ZERO actions — free fall (does gravity work? xvel stay 0?)
  3. 30 steps ALL +1.0 actions (do forces move the cheetah forward?)
  4. Observation extraction sanity (obs[0]==qpos[1]? obs[8]==qvel[0]?)
  5. Reward breakdown over 100 steps (manual vs GPU formula)
  6. NaN/Inf check after extended run

Run with:
    pixi run -e apple mojo run tests/test_half_cheetah_gpu_debug.mojo
    pixi run -e nvidia mojo run tests/test_half_cheetah_gpu_debug.mojo
"""

from gpu.host import DeviceContext, DeviceBuffer
from deep_rl import dtype as gpu_dtype
from envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# Small batch for readable output
comptime N_ENVS: Int = 4
comptime OBS_DIM: Int = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM: Int = HalfCheetahConfig.ACTION_DIM  # 6
comptime NQ: Int = 9  # HalfCheetah joints
comptime NV: Int = 9
# qpos starts at offset 0, qvel at offset NQ
comptime QPOS_BASE: Int = 0
comptime QVEL_BASE: Int = NQ

comptime Env = HalfCheetah[gpu_dtype, TERMINATE_ON_UNHEALTHY=False]
comptime STATE_SIZE: Int = Env.STATE_SIZE
comptime TOTAL_STATE: Int = N_ENVS * STATE_SIZE
comptime TOTAL_OBS: Int = N_ENVS * OBS_DIM
comptime TOTAL_ACT: Int = N_ENVS * ACT_DIM


@always_inline
fn fmt7(v: Scalar[gpu_dtype]) -> String:
    var s = String(v)
    if len(s) > 8:
        return String(s[:8])
    return s


fn has_nan_or_inf(buf: UnsafePointer[Scalar[gpu_dtype]], n: Int) -> Bool:
    for i in range(n):
        var v = buf[i]
        if v != v:  # NaN check
            return True
        if v > Scalar[gpu_dtype](1e30) or v < Scalar[gpu_dtype](-1e30):
            return True
    return False


fn count_nan(buf: UnsafePointer[Scalar[gpu_dtype]], n: Int) -> Int:
    var count = 0
    for i in range(n):
        var v = buf[i]
        if v != v:
            count += 1
    return count


fn print_env_state(
    buf: UnsafePointer[Scalar[gpu_dtype]],
    env: Int,
    prefix: String = "",
):
    """Print qpos + qvel for one env."""
    var b = env * STATE_SIZE
    var rootx = buf[b + QPOS_BASE + 0]
    var rootz = buf[b + QPOS_BASE + 1]
    var rooty = buf[b + QPOS_BASE + 2]
    var xvel = buf[b + QVEL_BASE + 0]
    var zvel = buf[b + QVEL_BASE + 1]
    var avel = buf[b + QVEL_BASE + 2]
    print(
        prefix
        + "env"
        + String(env)
        + ":"
        + " rootx="
        + fmt7(rootx)
        + " rootz="
        + fmt7(rootz)
        + " rooty="
        + fmt7(rooty)
        + " xvel="
        + fmt7(xvel)
        + " zvel="
        + fmt7(zvel)
        + " avel="
        + fmt7(avel)
    )


fn main() raises:
    print("=" * 65)
    print("HalfCheetah GPU Debug Diagnostics")
    print("=" * 65)
    print(
        "  STATE_SIZE =",
        STATE_SIZE,
        " OBS_DIM =",
        OBS_DIM,
        " ACT_DIM =",
        ACT_DIM,
    )
    print("  N_ENVS     =", N_ENVS, " TOTAL_STATE =", TOTAL_STATE)
    print()

    with DeviceContext() as ctx:
        # GPU buffers
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](TOTAL_STATE)
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](TOTAL_OBS)
        var rew_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS)
        var done_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS)
        var act_buf = ctx.enqueue_create_buffer[gpu_dtype](TOTAL_ACT)

        # CPU-side arrays for readback
        var states_h = InlineArray[Scalar[gpu_dtype], TOTAL_STATE](
            uninitialized=True
        )
        var obs_h = InlineArray[Scalar[gpu_dtype], TOTAL_OBS](
            uninitialized=True
        )
        var rew_h = InlineArray[Scalar[gpu_dtype], N_ENVS](uninitialized=True)
        var act_h = InlineArray[Scalar[gpu_dtype], TOTAL_ACT](fill=0.0)

        # =====================================================================
        # SECTION 1: Initial state after reset
        # =====================================================================
        print("=== SECTION 1: State after reset ===")
        Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf)
        ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
        ctx.synchronize()

        for e in range(N_ENVS):
            print_env_state(states_h.unsafe_ptr(), e, "  ")

        var nan_count_reset = count_nan(states_h.unsafe_ptr(), TOTAL_STATE)
        print("  NaN count in reset state:", nan_count_reset)
        print(
            "  Expected: rootz≈0 (slide joint displacement, torso body at"
            " z=0.7)"
        )
        print("  Expected: rooty≈0 ± noise (small random angle)")
        print()

        # =====================================================================
        # SECTION 2: Free fall — ZERO actions for 30 steps
        # =====================================================================
        print("=== SECTION 2: ZERO actions (30 steps, free fall) ===")
        # act_h already zero-filled → copy to GPU
        ctx.enqueue_copy(act_buf, act_h.unsafe_ptr())
        Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, UInt64(10))
        ctx.synchronize()

        var prev_rootz = Float64(0.0)
        var prev_rootx = Float64(0.0)
        for step in range(30):
            Env.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACT_DIM](
                ctx,
                states_buf,
                act_buf,
                rew_buf,
                done_buf,
                obs_buf,
                UInt64(step),
            )
            if step == 0 or step % 10 == 9:
                ctx.enqueue_copy(rew_h.unsafe_ptr(), rew_buf)
                ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
                ctx.synchronize()
                print(
                    "  step="
                    + String(step + 1)
                    + " rew[0]="
                    + fmt7(rew_h[0])
                    + " rew[1]="
                    + fmt7(rew_h[1])
                    + " nan_in_state="
                    + String(count_nan(states_h.unsafe_ptr(), STATE_SIZE))
                )
                print_env_state(states_h.unsafe_ptr(), 0, "    ")
                print_env_state(states_h.unsafe_ptr(), 1, "    ")
                prev_rootz = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 1])
                prev_rootx = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 0])

        ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
        ctx.synchronize()
        var final_rootz = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 1])
        var final_rootx = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 0])
        var final_xvel = Float64(states_h[0 * STATE_SIZE + QVEL_BASE + 0])
        var final_zvel = Float64(states_h[0 * STATE_SIZE + QVEL_BASE + 1])

        print("  VERDICT:")
        if final_zvel < -0.1:
            print(
                "    [PASS] zvel="
                + String(final_zvel)[:7]
                + " < -0.1 → gravity is working"
            )
        else:
            print(
                "    [FAIL] zvel="
                + String(final_zvel)[:7]
                + " NOT falling → gravity broken!"
            )
        if final_xvel > -0.05 and final_xvel < 0.05:
            print(
                "    [PASS] xvel="
                + String(final_xvel)[:7]
                + " ≈ 0 → no spurious x motion"
            )
        else:
            print(
                "    [WARN] xvel="
                + String(final_xvel)[:7]
                + " non-zero with zero actions!"
            )
        print()

        # =====================================================================
        # SECTION 3: All +1.0 actions for 30 steps
        # =====================================================================
        print("=== SECTION 3: ALL +1.0 actions (30 steps) ===")
        for i in range(TOTAL_ACT):
            act_h[i] = Scalar[gpu_dtype](1.0)
        ctx.enqueue_copy(act_buf, act_h.unsafe_ptr())
        Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, UInt64(20))
        ctx.synchronize()

        for step in range(30):
            Env.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACT_DIM](
                ctx,
                states_buf,
                act_buf,
                rew_buf,
                done_buf,
                obs_buf,
                UInt64(step + 100),
            )
            if step == 0 or step % 10 == 9:
                ctx.enqueue_copy(rew_h.unsafe_ptr(), rew_buf)
                ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
                ctx.synchronize()
                print(
                    "  step="
                    + String(step + 1)
                    + " rew[0]="
                    + fmt7(rew_h[0])
                    + " rew[1]="
                    + fmt7(rew_h[1])
                    + " nan_in_state="
                    + String(count_nan(states_h.unsafe_ptr(), STATE_SIZE))
                )
                print_env_state(states_h.unsafe_ptr(), 0, "    ")

        ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
        ctx.synchronize()
        var s3_xvel = Float64(states_h[0 * STATE_SIZE + QVEL_BASE + 0])
        var s3_rootx = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 0])
        var s3_nan = count_nan(states_h.unsafe_ptr(), STATE_SIZE)

        print("  VERDICT:")
        if s3_nan > 0:
            print(
                "    [FAIL] NaN count="
                + String(s3_nan)
                + " → physics instability!"
            )
        elif s3_xvel > 0.1 or s3_rootx > 0.05:
            print(
                "    [PASS] xvel="
                + String(s3_xvel)[:7]
                + " rootx="
                + String(s3_rootx)[:7]
                + " → actions ARE moving cheetah"
            )
        elif s3_xvel < -0.1:
            print(
                "    [INFO] xvel="
                + String(s3_xvel)[:7]
                + " negative → cheetah moving BACKWARD with +1"
            )
        else:
            print(
                "    [WARN] xvel="
                + String(s3_xvel)[:7]
                + " rootx="
                + String(s3_rootx)[:7]
                + " → tiny motion, actions may not work!"
            )
        print()

        # =====================================================================
        # SECTION 4: Observation extraction sanity check
        # =====================================================================
        print("=== SECTION 4: Observation extraction sanity check ===")
        print(
            "  Expected layout: obs[0..7] = qpos[1..8], obs[8..16] = qvel[0..8]"
        )
        print("  (obs_qpos_skip=1 skips rootx from qpos)")
        print()

        # Reset and run 5 steps so state is non-trivial
        for i in range(TOTAL_ACT):
            act_h[i] = Scalar[gpu_dtype](0.5)
        ctx.enqueue_copy(act_buf, act_h.unsafe_ptr())
        Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, UInt64(30))
        ctx.synchronize()
        for step in range(5):
            Env.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACT_DIM](
                ctx,
                states_buf,
                act_buf,
                rew_buf,
                done_buf,
                obs_buf,
                UInt64(step + 200),
            )
        ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
        ctx.enqueue_copy(obs_h.unsafe_ptr(), obs_buf)
        ctx.synchronize()

        var obs_ok = True
        for e in range(N_ENVS):
            print("  Env " + String(e) + ":")
            # Check obs[0] == qpos[1] (rootz)
            var st_qpos1 = states_h[e * STATE_SIZE + QPOS_BASE + 1]
            var ob0 = obs_h[e * OBS_DIM + 0]
            var diff0 = Float64(ob0 - st_qpos1)
            if diff0 < 0.0:
                diff0 = -diff0
            var match0 = diff0 < 1e-5

            # Check obs[8] == qvel[0] (rootx_vel)
            var st_qvel0 = states_h[e * STATE_SIZE + QVEL_BASE + 0]
            var ob8 = obs_h[e * OBS_DIM + 8]
            var diff8 = Float64(ob8 - st_qvel0)
            if diff8 < 0.0:
                diff8 = -diff8
            var match8 = diff8 < 1e-5

            print(
                "    obs[0]="
                + fmt7(ob0)
                + "  qpos[1]="
                + fmt7(st_qpos1)
                + "  match="
                + String(match0)
                + ("  [PASS]" if match0 else "  [FAIL] <-- obs[0] != qpos[1]!")
            )
            print(
                "    obs[8]="
                + fmt7(ob8)
                + "  qvel[0]="
                + fmt7(st_qvel0)
                + "  match="
                + String(match8)
                + (
                    "  [PASS]" if match8 else "  [FAIL] <-- obs[8] != qvel[0]! (x_velocity missing from obs)"
                )
            )

            if not match0 or not match8:
                obs_ok = False

            # Print all 17 obs
            print("    All obs: [", end="")
            for i in range(OBS_DIM):
                print(fmt7(obs_h[e * OBS_DIM + i]), end=" ")
            print("]")
        print()

        if obs_ok:
            print(
                "  VERDICT: [PASS] Observation extraction matches expected"
                " layout"
            )
        else:
            print(
                "  VERDICT: [FAIL] Observation extraction has index mismatch!"
                " Policy sees wrong state!"
            )
        print()

        # =====================================================================
        # SECTION 5: Manual reward breakdown (100 steps with +1 actions)
        # =====================================================================
        print(
            "=== SECTION 5: Reward formula breakdown (100 steps, all +1"
            " actions) ==="
        )
        print(
            "  Formula: reward = x_velocity - 0.1*sum(clip(a)^2) - 0.5*|rooty|"
        )
        print("  With all actions=1.0: ctrl_cost = 0.1*6 = 0.6/step")
        print()

        for i in range(TOTAL_ACT):
            act_h[i] = Scalar[gpu_dtype](1.0)
        ctx.enqueue_copy(act_buf, act_h.unsafe_ptr())
        Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, UInt64(40))
        ctx.synchronize()

        # Read initial rootx (for first step's prev_x)
        ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
        ctx.synchronize()

        var total_gpu_rew = Float64(0.0)
        var total_xvel = Float64(0.0)
        var total_angle = Float64(0.0)
        var total_rootx = Float64(0.0)

        for step in range(100):
            # Save qpos[0] BEFORE step (= prev_x used in GPU reward kernel)
            var prev_rootx_ = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 0])

            Env.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACT_DIM](
                ctx,
                states_buf,
                act_buf,
                rew_buf,
                done_buf,
                obs_buf,
                UInt64(step + 300),
            )
            ctx.enqueue_copy(rew_h.unsafe_ptr(), rew_buf)
            ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
            ctx.synchronize()

            var x_after = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 0])
            var rooty = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 2])
            var xvel = (x_after - prev_rootx_) / (
                0.01 * 5.0
            )  # timestep * FRAME_SKIP
            var ctrl = 0.1 * 6.0  # 6 actions × 1.0^2 × 0.1
            var angle = rooty if rooty >= 0.0 else -rooty
            var angle_pen = 0.5 * angle
            var manual_r = xvel - ctrl - angle_pen
            var gpu_r = Float64(rew_h[0])

            total_gpu_rew += gpu_r
            total_xvel += xvel
            total_angle += angle
            total_rootx = x_after

            if step == 0 or step % 25 == 24:
                print(
                    "  step="
                    + String(step + 1)
                    + " gpu_r="
                    + String(gpu_r)[:8]
                    + " manual="
                    + String(manual_r)[:8]
                    + " xvel="
                    + String(xvel)[:7]
                    + " ctrl=-0.6"
                    + " angle_pen="
                    + String(angle_pen)[:6]
                    + " rootx="
                    + String(x_after)[:7]
                )

        print()
        print("  Cumulative over 100 steps:")
        print("    Sum GPU reward:   " + String(total_gpu_rew)[:9])
        print("    Mean GPU reward:  " + String(total_gpu_rew / 100.0)[:9])
        print("    Mean xvel:        " + String(total_xvel / 100.0)[:9])
        print("    Mean angle:       " + String(total_angle / 100.0)[:9])
        print("    Final rootx:      " + String(total_rootx)[:9])
        print("    Ctrl_cost/step:   -0.6 (constant with all +1 actions)")
        print()
        if total_xvel / 100.0 > 0.3:
            print(
                "  VERDICT: [PASS] Cheetah moves forward with max torques (avg"
                " xvel>0.3)"
            )
        elif total_xvel / 100.0 > 0.0:
            print("  VERDICT: [INFO] Cheetah barely moves forward (avg xvel>0)")
        else:
            print(
                "  VERDICT: [WARN] Cheetah moves BACKWARD with all +1 torques!"
            )
        print()

        # =====================================================================
        # SECTION 6: NaN/Inf check with varied actions over 500 steps
        # =====================================================================
        print(
            "=== SECTION 6: Stability — 500 steps with alternating ±0.5"
            " actions ==="
        )
        Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, UInt64(50))
        ctx.synchronize()

        var nan_steps = 0
        var max_seen_xvel = Float64(0.0)
        var max_seen_rootx = Float64(0.0)
        for step in range(500):
            # Alternate action sign every 10 steps
            var sign = Scalar[gpu_dtype](1.0) if (
                step // 10
            ) % 2 == 0 else Scalar[gpu_dtype](-1.0)
            for i in range(TOTAL_ACT):
                act_h[i] = sign * Scalar[gpu_dtype](0.5)
            ctx.enqueue_copy(act_buf, act_h.unsafe_ptr())

            Env.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACT_DIM](
                ctx,
                states_buf,
                act_buf,
                rew_buf,
                done_buf,
                obs_buf,
                UInt64(step + 400),
            )
            if step % 100 == 99:
                ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
                ctx.synchronize()
                var nc = count_nan(states_h.unsafe_ptr(), STATE_SIZE)
                if nc > 0:
                    nan_steps += 1
                var xv = Float64(states_h[0 * STATE_SIZE + QVEL_BASE + 0])
                var rx = Float64(states_h[0 * STATE_SIZE + QPOS_BASE + 0])
                if xv < 0.0:
                    xv = -xv
                if rx < 0.0:
                    rx = -rx
                if xv > max_seen_xvel:
                    max_seen_xvel = xv
                if rx > max_seen_rootx:
                    max_seen_rootx = rx
                print(
                    "  step="
                    + String(step + 1)
                    + " nan="
                    + String(nc)
                    + " rootx="
                    + fmt7(states_h[0 * STATE_SIZE + QPOS_BASE + 0])
                    + " rootz="
                    + fmt7(states_h[0 * STATE_SIZE + QPOS_BASE + 1])
                    + " xvel="
                    + fmt7(states_h[0 * STATE_SIZE + QVEL_BASE + 0])
                )

        ctx.enqueue_copy(states_h.unsafe_ptr(), states_buf)
        ctx.synchronize()
        var final_nan = count_nan(states_h.unsafe_ptr(), TOTAL_STATE)
        print("  VERDICT:")
        if final_nan > 0:
            print(
                "    [FAIL] NaN count="
                + String(final_nan)
                + " → physics explosion after sustained actions!"
            )
        else:
            print("    [PASS] No NaN/Inf after 500 steps")
        print()

        print("=" * 65)
        print("SUMMARY")
        print("=" * 65)
        print("Look for [FAIL] / [WARN] lines above to identify the bug.")
        print()
        print("Common issues:")
        print(
            "  SECTION 2 FAIL: gravity broken → integrator or state layout"
            " wrong"
        )
        print(
            "  SECTION 3 WARN: actions not moving cheetah →"
            " apply_actions_kernel bug"
        )
        print(
            "  SECTION 4 FAIL: obs index mismatch → policy never sees"
            " x_velocity!"
        )
        print(
            "  SECTION 5 xvel≈0: reward signal too weak → reward formula"
            " problem"
        )
        print("  SECTION 6 NaN: physics explodes → integrator instability")

    print()
    print(">>> Debug complete <<<")
