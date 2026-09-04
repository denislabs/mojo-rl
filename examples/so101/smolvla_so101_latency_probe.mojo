# +--------------------------------------------------------------------------+ #
# | How long does ONE SmolVLA query take? The number the control loop needs.
# +--------------------------------------------------------------------------+ #
"""Time one `select_action` on the published weights, phase by phase.

    pixi run -e apple mojo build -I . -o /tmp/svla_probe \\
        examples/so101/smolvla_so101_latency_probe.mojo
    /tmp/svla_probe

⚠⚠ **THIS IS A PREREQUISITE, NOT A CURIOSITY.** The ACT deploy program's whole
architecture — a 30 Hz action grid, a temporal ensemble, a chunk index that
means what it meant in training — is built on ONE measured number: 95 ms per
forward on this machine. Writing SmolVLA's control loop before measuring the
same number would be picking an architecture by guesswork.

The decisive comparison is between two durations:

  * **one chunk of motion** = `chunk_size / fps` = 50 / 30 = **1.667 s**. That is
    how much wall-clock the arm can execute from a single query.
  * **one query** = what this measures.

If a query is comfortably shorter than a chunk, the loop can be open-loop
chunk-to-chunk: query, execute 50 waypoints at 30 Hz, query again while the tail
is still playing. SmolVLA is DESIGNED for that — `n_action_steps` is 50, the
whole chunk, not ACT's handful. If a query is longer, the arm stalls between
chunks and the loop needs the forward off the control thread, which is a
different and much larger program.

⚠ **The per-phase figures do NOT sum to the end-to-end figure, and must not be
reported as if they did.** GPU work is enqueued asynchronously; timing a phase
means synchronising after it, and those extra synchronisations change the
pipeline being measured. End-to-end is the honest total. The breakdown is for
deciding WHAT to optimise, not for adding up.

⚠ Needs the published weights (~907 MB, cached) and instantiates every layer:
402,737,376 params x 4 bytes x 2 (host AND device) = **3.2 GB**. One policy.
"""

from std.math import sqrt
from std.time import perf_counter_ns
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.io.hf import hf_download_file, HF_MODEL
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.smolvla.policy import SmolVLAPolicy
from mojo_rl.deep_agents.smolvla.normalize import SmolVLAStats, normalize_state
from mojo_rl.deep_agents.smolvla.heads import SMOLVLA_CONNECTOR_IN
from mojo_rl.deep_agents.smolvla.observation import fill_camera_images
from mojo_rl.deep_agents.smolvla.tasks import TaskTokens
from mojo_rl.vision.resize_pad import SIGLIP_INPUT

comptime TARGET = "gpu"
"""⚠ ONE target per binary, deliberately. Building both would instantiate two
policies and cost 6.4 GB — the mistake that took a 16 GiB laptop down once
already. To measure the CPU path, change this and rebuild.

Worth measuring both: ACT found Metal SLOWER than CPU at batch 1, because a few
hundred tiny kernels each pay Metal's ~20 us command-buffer retirement floor
(`_the_metal_launch_floor_is_command_buffer_retirement`). SmolVLA's kernels are
far bigger — 1024-token attention in the vision tower — so that result may
invert here. It is a measurement either way, not an inference from ACT's."""

comptime REPO = String("lerobot/smolvla_base")
comptime TABLE = "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"
comptime N_CAM = 2
comptime N_LANG = 6
comptime CHUNK = 50
comptime STEPS = 10
comptime FPS = 30
comptime Pol = SmolVLAPolicy[N_CAM, N_LANG, CHUNK, STEPS, 1]
comptime CAM_W = 640
comptime CAM_H = 480
comptime RDIM = 6
comptime REPS = 5
comptime WARMUP = 2


def ms(ns: Int) -> Float64:
    return Float64(ns) / 1.0e6


def main() raises:
    print("=" * 70)
    print("SmolVLA query latency —", TARGET, ", batch 1,", N_CAM, "cameras")
    print("=" * 70)

    var tasks = TaskTokens(String(TABLE))
    var ids = tasks.for_index(0)
    print("  instruction:", tasks.texts[0], "->", len(ids), "tokens, P =",
          Pol.P)

    var path = hf_download_file(REPO, String("model.safetensors"), HF_MODEL)
    var d = DeviceContext()
    print("  building every layer (3.2 GB) …")
    var pol = Pol.make[TARGET](Optional(d))
    pol.load[TARGET](path, Optional(d))

    # The recording's own scale. Not the point of this probe, but a policy
    # without stats refuses to run.
    var m: List[Float32] = [16.64, -29.97, 31.07, 73.73, 41.12, 26.27]
    var sd: List[Float32] = [21.00, 54.38, 51.43, 17.93, 18.72, 9.21]
    for i in range(RDIM):
        pol.stats.state_mean.append(m[i])
        pol.stats.state_std.append(sd[i])
        pol.stats.action_mean.append(m[i])
        pol.stats.action_std.append(sd[i])

    # A plausible observation: two 640x480 frames through the real chain.
    var frames = List[List[UInt8]]()
    for c in range(N_CAM):
        var f = List[UInt8](unsafe_uninit_length=CAM_W * CAM_H * 3)
        for i in range(len(f)):
            f[i] = UInt8((i * 7 + c * 53) % 256)
        frames.append(f^)
    var widths: List[Int] = [CAM_W, CAM_W]
    var heights: List[Int] = [CAM_H, CAM_H]
    var scratch = List[Float32]()
    var images = Tensor()

    # ⚠ SYNCHRONISE BEFORE STOPPING THE CLOCK. Without this the timer measures
    # the enqueue and not the work — the first draft of this file reported
    # 17.6 ms here and 127.6 ms for the same call further down, and 17.6 was
    # fiction. This file's own header warns about exactly that; the warning did
    # not stop me writing it.
    var t_pre = perf_counter_ns()
    fill_camera_images[TARGET, N_CAM, SIGLIP_INPUT](
        frames, widths, heights, False, images, scratch, Optional(d)
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var preproc_ns = perf_counter_ns() - t_pre

    var pose: List[Float32] = [12.0, -40.0, 22.0, 70.0, 35.0, 20.0]
    comptime XN = CHUNK * 32
    var noise = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    comptime if TARGET != "cpu":
        noise.upload(d)
    var act = List[Float32]()

    # ⚠ Warm up first. The first call pays lazy buffer allocation and, on GPU,
    # kernel specialisation — real costs, but paid ONCE, so folding them into a
    # steady-state figure would overstate every subsequent query.
    for _ in range(WARMUP):
        pol.select_action[TARGET](
            images, ids, pose, noise, act, Optional(d)
        )
    comptime if TARGET != "cpu":
        d.synchronize()

    # ── end to end, the honest total ─────────────────────────────────────
    var best = 0
    var total = 0
    for r in range(REPS):
        var t0 = perf_counter_ns()
        pol.select_action[TARGET](
            images, ids, pose, noise, act, Optional(d)
        )
        comptime if TARGET != "cpu":
            d.synchronize()
        var dt = perf_counter_ns() - t0
        total += dt
        # ⚠ MIN, not mean: a baseline that drifts upward across a session is
        # this project's recorded failure mode, and the minimum is the figure
        # least polluted by whatever else the machine was doing.
        if best == 0 or dt < best:
            best = dt
        print("      rep", r, ":", ms(dt), "ms")

    print()
    print("  camera preprocessing (CPU, 2 frames):", ms(preproc_ns), "ms")
    print("  query, min of", REPS, ":", ms(best), "ms")
    print("  query, mean     :", ms(total // REPS), "ms")
    print("  queries/s (min) :", 1000.0 / ms(best))

    # ── the comparison the loop design turns on ──────────────────────────
    var chunk_ms = 1000.0 * Float64(CHUNK) / Float64(FPS)
    var query_ms = ms(best) + ms(preproc_ns)
    print()
    print("  one chunk of motion:", chunk_ms, "ms (", CHUNK, "steps at", FPS,
          "fps )")
    print("  one query + preproc:", query_ms, "ms")
    print("  duty cycle         :", query_ms / chunk_ms)
    if query_ms < chunk_ms:
        print()
        print("  => a query fits inside a chunk. Open-loop chunk-to-chunk is")
        print("     viable: execute 50 waypoints while the next query runs.")
        print("     Headroom:", chunk_ms - query_ms, "ms")
    else:
        print()
        print("  => A QUERY IS LONGER THAN THE MOTION IT PRODUCES. The arm")
        print("     would stall between chunks. The loop needs the forward off")
        print("     the control thread, or fewer denoising steps, or a")
        print("     smaller vision path — decide with the breakdown below,")
        print("     NOT by shortening the chunk.")

    # ── per phase, for deciding what to optimise ─────────────────────────
    # ⚠ These are measured with a synchronisation after each phase, so they do
    # NOT sum to the figure above. Read them as proportions.
    print()
    print("  per-phase (syncs added — does NOT sum to the total above):")

    var st = List[Float32]()
    normalize_state(pol.stats, pose, st, 32)
    pol.state_buf.ensure(32)
    for i in range(32):
        pol.state_buf.data[i] = Scalar[DT](st[i])
    comptime if TARGET != "cpu":
        pol.state_buf.upload_resident(d)

    var t_pfx = perf_counter_ns()
    pol.prefix.run[TARGET, Pol.VOCAB, SMOLVLA_CONNECTOR_IN, 32](
        pol.vision, pol.connector, pol.embed.weight.val, pol.state_proj,
        images, ids, pol.state_buf, pol.prefix_buf, Optional(d),
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var pfx_ns = perf_counter_ns() - t_pfx

    var t_fill = perf_counter_ns()
    pol.cache.reset()
    pol.prefill.run[TARGET](
        pol.tower, pol.cache, pol.prefix_buf, pol.prefill_out, Optional(d)
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var fill_ns = perf_counter_ns() - t_fill

    var t_den = perf_counter_ns()
    pol.sampler.sample[TARGET, Pol.P](
        pol.expert, pol.cache, pol.denoiser, pol.action_in, pol.time_mlp_in,
        pol.time_mlp_out, pol.action_out, noise, pol.chunk_buf, Optional(d),
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var den_ns = perf_counter_ns() - t_den

    var sum_ns = preproc_ns + pfx_ns + fill_ns + den_ns
    print("      preprocess 2 frames  :", ms(preproc_ns), "ms")
    print("      prefix (2x SigLIP +  :", ms(pfx_ns), "ms   <- 2 vision towers")
    print("        connector, gather, state_proj)")
    print("      prefill 16 layers    :", ms(fill_ns), "ms")
    print("      denoise", STEPS, "steps     :", ms(den_ns), "ms   (",
          ms(den_ns // STEPS), "ms/step )")
    print("      ---")
    print("      sum of phases        :", ms(sum_ns), "ms  vs end-to-end",
          ms(best), "ms")
    print()
    print("  share of the sum:")
    print("      preprocess :", 100.0 * Float64(preproc_ns) / Float64(sum_ns), "%")
    print("      prefix     :", 100.0 * Float64(pfx_ns) / Float64(sum_ns), "%")
    print("      prefill    :", 100.0 * Float64(fill_ns) / Float64(sum_ns), "%")
    print("      denoise    :", 100.0 * Float64(den_ns) / Float64(sum_ns), "%")

    print()
    print("Measured on this machine. Do not carry these numbers to another.")
