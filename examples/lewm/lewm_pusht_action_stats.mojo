"""LeWM (nn) — PushT offline ACTION statistics probe.

Closed-loop executability gate. The HF lewm-pusht dataset stores
PRE-NORMALIZED actions (the LeWM reference z-scores absolute agent target
positions in [0,512]); the original raw mean/std were computed from data
the repo doesn't ship, so the stored→env-action denormalization is unknown.
To drive the mojo PushTEnv (which takes raw [0,512] targets) from
WM-planned actions, we must recover that mapping — so MEASURE the stored
distribution: per-dim (x,y) min / max / mean / std / RMS over many clips.

Read the output:
  - mean≈0, std≈1                  → z-score; need the raw μ,σ (still blocked,
                                     but proprio cross-check may recover them)
  - range≈[0,1], mean≈0.5          → /512 normalization → env = stored·512
  - range≈[-1,1], mean≈0           → (stored+1)·256
  - small RMS (~0.2) + signed      → likely per-step DELTAS → env target =
                                     agent_pos + stored·scale (scale = recovered)

Action layout per step: frameskip(5) × action_dim(2) = 10, interleaved
[x0,y0,x1,y1,...]; we pool the 5 sub-steps per axis.

Run (NVIDIA box has the dataset):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_action_stats.mojo
"""

from std.math import sqrt
from std.memory import alloc
from std.random import seed as rng_seed

from mojo_rl.envs.pusht import PushTOfflineSampler


comptime B = 32
comptime T = 6
comptime FRAMESKIP = 5
comptime ACT_DIM = 2
comptime ACT = FRAMESKIP * ACT_DIM          # 10
comptime IMG = 224
comptime IMG_C = 3
comptime PIX_PER = IMG * IMG * IMG_C
comptime N_BATCH = 40                         # 32×6×5 ≈ 38k samples/dim total


def main() raises:
    print("=" * 70)
    print("LeWM nn — PushT offline action statistics")
    print("=" * 70)
    rng_seed(1)

    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var pix = alloc[Scalar[DType.uint8]](B * T * PIX_PER)
    var act = alloc[Scalar[DType.float32]](B * T * ACT)

    # per-axis accumulators (axis 0 = x, axis 1 = y)
    var mn = List[Float64](length=2, fill=1.0e30)
    var mx = List[Float64](length=2, fill=-1.0e30)
    var s1 = List[Float64](length=2, fill=0.0)   # Σ v
    var s2 = List[Float64](length=2, fill=0.0)   # Σ v²
    var cnt = List[Float64](length=2, fill=0.0)

    print("sampling", N_BATCH, "batches (B=", B, "T=", T, ") ...")
    for _ in range(N_BATCH):
        sampler.sample_batch_uint8(B, T, pix, act)
        for b in range(B):
            for t in range(T):
                var base = (b * T + t) * ACT
                for k in range(FRAMESKIP):
                    for d in range(ACT_DIM):
                        var v = Float64(act[base + k * ACT_DIM + d])
                        if v < mn[d]:
                            mn[d] = v
                        if v > mx[d]:
                            mx[d] = v
                        s1[d] += v
                        s2[d] += v * v
                        cnt[d] += 1.0

    print("-" * 70)
    var axis = List[String]()
    axis.append("x")
    axis.append("y")
    for d in range(2):
        var n = cnt[d]
        var mean = s1[d] / n
        var var_ = s2[d] / n - mean * mean
        var std = sqrt(var_) if var_ > 0.0 else 0.0
        var rms = sqrt(s2[d] / n)
        print(
            "  axis", axis[d], ": n=", Int(n),
            " min=", mn[d], " max=", mx[d],
        )
        print(
            "          mean=", mean, " std=", std, " rms=", rms,
        )
    print("-" * 70)
    print("(see header for how the shape maps to a denormalization)")

    pix.free()
    act.free()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
