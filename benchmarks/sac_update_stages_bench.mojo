# +--------------------------------------------------------------------------+ #
# | One SAC update, stage by stage, on the device this runs on
# +--------------------------------------------------------------------------+ #
"""Where the time goes in `SACTrainer.train_step` at the SO-101 tower's shape.

    pixi run -e apple  mojo run -I . benchmarks/sac_update_stages_bench.mojo
    pixi run -e nvidia mojo run -I . benchmarks/sac_update_stages_bench.mojo

⚠⚠ WHY. The tower family trains on the laptop's Metal at 17.9 env-steps/s
(19 Sep) against 57 during the random warmup: the difference is ONE SAC
update per env-step (UTD 1), so an update costs ~38 ms. Three 256-wide MLPs
at batch 256 are ~0.7 GFLOP; at the GEMM rate `linear_matmul_bench` reports
that is a few ms, so either the matmuls are far below that rate here or the
update is launch-bound. A whole-update number cannot tell the two apart. This
times each block of `train_step` with a sync on both sides, prints the stage
sum against the whole step (the gap is the diagnostics tail and the launch
overlap the syncs remove), and the per-stage share.

THE SHAPE is `sac_family_driver`'s: OBS 49, ACT 6, HIDDEN 256, BATCH 256,
`SACActorNet` / `SACCriticNet` from `sac/config.mojo`. The replay is filled
with random transitions from the host — the arithmetic does not care what the
numbers are, and nothing here trains.

⚠ The block calls below are `train_step`'s body, copied. If `train_step`
gains a block, this bench does not see it: the whole-step row is the check,
because the stage sum is printed against it.
"""

from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac.config import SAC


comptime OBS = 49
comptime ACT = 6
comptime HIDDEN = 256
comptime BATCH = 256
comptime CAP = 8192
comptime N_FILL = 4096

comptime WARMUP = 5
comptime REPS = 40
comptime N_STAGES = 6


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed) * 6364136223846793005 + 1442695040888963407

    def unit(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 11) & 0x1FFFFFFFFFFFFF) / Float64(1 << 53)


def _fmt(x: Float64, digits: Int) -> String:
    var scale = 1.0
    for _ in range(digits):
        scale *= 10.0
    var r = Float64(Int(x * scale + (0.5 if x >= 0.0 else -0.5))) / scale
    return String(r)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out


def main() raises:
    comptime assert has_accelerator(), "this benchmark times GPU kernels"
    with DeviceContext() as ctx:
        print("=" * 84)
        print(
            "SAC update stages — OBS " + String(OBS) + " ACT " + String(ACT)
            + " HIDDEN " + String(HIDDEN) + " BATCH " + String(BATCH)
            + " — " + String(ctx.name())
        )
        print("=" * 84)

        var agent = SAC["gpu", OBS, ACT, BATCH, CAP, HIDDEN](
            ctx=ctx,
            target_entropy=Scalar[DT](-3.0),
            learning_starts=0,
        )

        # Fill the replay from the host with random transitions.
        var rng = Lcg(7)
        var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
        var nobs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
        var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
        for _ in range(N_FILL):
            for d in range(OBS):
                obs[d] = Scalar[DT](rng.unit() * 2.0 - 1.0)
                nobs[d] = Scalar[DT](rng.unit() * 2.0 - 1.0)
            for j in range(ACT):
                act[j] = Scalar[DT](rng.unit() * 2.0 - 1.0)
            agent.trainer.record(
                obs, act, Scalar[DT](rng.unit()), nobs,
                Scalar[DT](1.0 if rng.unit() < 0.01 else 0.0),
            )
        ctx.synchronize()

        # ── the whole step: host enqueue time, then the same with the wait ──
        # ⚠ THE SPLIT THAT DECIDES THE FIX. `enq` is the time `train_step`
        # takes to RETURN — every kernel enqueued, none waited for. If that
        # is most of the whole, the update is bound by the host's per-launch
        # dispatch cost and the lever is FEWER LAUNCHES; if it is small, the
        # GPU is the one taking the time and the lever is the kernels.
        var whole_best = 1.0e30
        var whole_sum = 0.0
        var enq_best = 1.0e30
        var enq_sum = 0.0
        for rep in range(WARMUP + REPS):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            if not agent.trainer.train_step(rep):
                raise Error("train_step did not step: replay too small?")
            var t_enq = Float64(perf_counter_ns() - t0) / 1e6
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if rep >= WARMUP:
                whole_sum += dt
                enq_sum += t_enq
                if dt < whole_best:
                    whole_best = dt
                if t_enq < enq_best:
                    enq_best = t_enq
        var whole_mean = whole_sum / Float64(REPS)
        var enq_mean = enq_sum / Float64(REPS)

        # ── the stages, synced on both sides ──
        var names: List[String] = [
            String("sample   (replay -> minibatch)"),
            String("target_y (actor fwd + 2 target critics)"),
            String("critics  (2x fwd + bwd + Adam)"),
            String("actor    (fwd + 2 critics fwd + bwd + Adam)"),
            String("alpha    (ScalarAdam on device)"),
            String("polyak   (2 target nets)"),
        ]
        var best = List[Float64](length=N_STAGES, fill=1.0e30)
        var sums = List[Float64](length=N_STAGES, fill=0.0)
        agent.trainer.state.ctx = agent.trainer.ctx
        for rep in range(WARMUP + REPS):
            agent.trainer.state.step_idx = rep
            agent.trainer.state.did_step = True
            var t = List[Float64](length=N_STAGES, fill=0.0)

            ctx.synchronize()
            var t0 = perf_counter_ns()
            agent.trainer.sample_blk.step(agent.trainer.state)
            ctx.synchronize()
            t[0] = Float64(perf_counter_ns() - t0) / 1e6
            if not agent.trainer.state.did_step:
                raise Error("sample block did not step")

            t0 = perf_counter_ns()
            agent.trainer.target_y_blk.step["gpu"](
                agent.trainer.state, agent.trainer.actor,
                agent.trainer.pair1.target_net, agent.trainer.pair2.target_net,
            )
            ctx.synchronize()
            t[1] = Float64(perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            agent.trainer.twin_critic_blk.step["gpu", ACCUMULATE=True](
                agent.trainer.state,
                agent.trainer.pair1.online, agent.trainer.critic1_opt,
                agent.trainer.pair2.online, agent.trainer.critic2_opt,
            )
            ctx.synchronize()
            t[2] = Float64(perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            var out = agent.trainer.actor_loss_blk.forward_backward["gpu"](
                agent.trainer.actor, agent.trainer.actor_opt,
                agent.trainer.pair1.online, agent.trainer.pair2.online,
                agent.trainer.state.mb_s, agent.trainer.state.alpha,
                agent.trainer.ctx,
            )
            ctx.synchronize()
            t[3] = Float64(perf_counter_ns() - t0) / 1e6
            _ = out

            t0 = perf_counter_ns()
            agent.trainer.alpha_opt.step_device(
                ctx, agent.trainer.actor_loss_blk.lp_mean_dev(),
                agent.trainer.alpha_blk.target_entropy,
            )
            ctx.synchronize()
            t[4] = Float64(perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            agent.trainer.polyak_blk.step["gpu"](
                agent.trainer.state, agent.trainer.pair1, agent.trainer.pair2
            )
            ctx.synchronize()
            t[5] = Float64(perf_counter_ns() - t0) / 1e6

            if rep >= WARMUP:
                for s in range(N_STAGES):
                    sums[s] += t[s]
                    if t[s] < best[s]:
                        best[s] = t[s]

        print("")
        print(
            "   " + _pad(String("stage"), 46) + _pad(String("best ms"), 10)
            + _pad(String("mean ms"), 10) + "share of stage sum"
        )
        var sum_best = 0.0
        var sum_mean = 0.0
        for s in range(N_STAGES):
            sum_best += best[s]
            sum_mean += sums[s] / Float64(REPS)
        for s in range(N_STAGES):
            var mean = sums[s] / Float64(REPS)
            print(
                "   " + _pad(names[s], 46) + _pad(_fmt(best[s], 3), 10)
                + _pad(_fmt(mean, 3), 10)
                + _fmt(100.0 * mean / sum_mean, 1) + "%"
            )
        print(
            "   " + _pad(String("stage sum"), 46) + _pad(_fmt(sum_best, 3), 10)
            + _fmt(sum_mean, 3)
        )
        print(
            "   " + _pad(String("whole train_step (no syncs inside)"), 46)
            + _pad(_fmt(whole_best, 3), 10) + _fmt(whole_mean, 3)
        )
        print(
            "   " + _pad(String("  of which host enqueue (train_step returns)"), 46)
            + _pad(_fmt(enq_best, 3), 10) + _fmt(enq_mean, 3)
            + "   " + _fmt(100.0 * enq_mean / whole_mean, 0) + "% of the whole"
        )
        print("")
        print(
            "   at UTD 1 and 32 lanes, one batched env-step carries 32 updates = "
            + _fmt(32.0 * whole_mean, 1) + " ms"
        )
