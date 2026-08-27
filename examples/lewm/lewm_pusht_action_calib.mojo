"""LeWM (nn) — PushT action denormalization CALIBRATION.

Recovers the stored-action → env-action ([0,512] agent target) mapping
DEFINITIVELY from the dataset, instead of guessing. The rendered frames
carry the unique RoyalBlue agent dot, so we detect the agent's pixel
position per frame, convert to world coords (×512/224), and regress it
against the stored action. Two hypotheses, per axis, by least squares:

  ABSOLUTE:  agent_world[t+1] = a · action[t] + b      (action = norm target)
  DELTA:     agent_world[t+1]-agent_world[t] = a·action[t] + b   (norm move)

The hypothesis with R² ≈ 1 reveals the convention; (a, b) is the denorm
(env_target = a·action + b for ABSOLUTE; env_target = agent + a·action + b
for DELTA). Uses the LAST of the 5 frameskip sub-actions (the target the PD
controller converges toward by the next observation). Frames where the agent
is occluded (no blue centroid) are skipped.

Dataset-only (no WM/GPU). Run on the NVIDIA box (has the dataset):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_action_calib.mojo
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
comptime PIX_PER = IMG * IMG * IMG_C         # HWC
comptime N_BATCH = 30
comptime W2P = 512.0 / Float64(IMG)          # world units per pixel


struct _Reg(Copyable, Movable):
    """Streaming least-squares y = a·x + b + running R²."""
    var n: Float64
    var sx: Float64
    var sy: Float64
    var sxx: Float64
    var syy: Float64
    var sxy: Float64

    def __init__(out self):
        self.n = 0.0; self.sx = 0.0; self.sy = 0.0
        self.sxx = 0.0; self.syy = 0.0; self.sxy = 0.0

    def add(mut self, x: Float64, y: Float64):
        self.n += 1.0; self.sx += x; self.sy += y
        self.sxx += x * x; self.syy += y * y; self.sxy += x * y

    def report(self, name: String):
        if self.n < 2.0:
            print("   ", name, ": insufficient samples")
            return
        var mx = self.sx / self.n
        var my = self.sy / self.n
        var vx = self.sxx / self.n - mx * mx
        var vy = self.syy / self.n - my * my
        var cxy = self.sxy / self.n - mx * my
        var a = cxy / vx if vx > 1e-12 else 0.0
        var b = my - a * mx
        var r2 = (cxy * cxy) / (vx * vy) if (vx > 1e-12 and vy > 1e-12) else 0.0
        print("   ", name, ": a=", a, " b=", b, " R²=", r2, " (n=", Int(self.n), ")")


def _agent_centroid(
    pix: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    frame_off: Int,
    mut cx: Float64,
    mut cy: Float64,
) -> Bool:
    """Centroid of RoyalBlue (65,105,225) pixels in an HWC frame at
    `frame_off`. Returns False if the agent is occluded (too few px)."""
    var sx: Float64 = 0.0
    var sy: Float64 = 0.0
    var n: Int = 0
    for y in range(IMG):
        for x in range(IMG):
            var o = frame_off + (y * IMG + x) * 3
            var r = Int(pix[o])
            var g = Int(pix[o + 1])
            var bl = Int(pix[o + 2])
            if bl > 150 and bl > r + 40 and bl > g + 40:
                sx += Float64(x); sy += Float64(y); n += 1
    if n < 3:
        return False
    cx = (sx / Float64(n)) * W2P
    cy = (sy / Float64(n)) * W2P
    return True


def main() raises:
    print("=" * 70)
    print("LeWM nn — PushT action denormalization calibration")
    print("=" * 70)
    rng_seed(2)

    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var pix = alloc[Scalar[DType.uint8]](B * T * PIX_PER)
    var act = alloc[Scalar[DType.float32]](B * T * ACT)

    # regressions: absolute (agent[t+1] ~ act[t]) and delta per axis
    var abs_x = _Reg(); var abs_y = _Reg()
    var dly_x = _Reg(); var dly_y = _Reg()

    print("sampling", N_BATCH, "batches, detecting agent + regressing ...")
    for _ in range(N_BATCH):
        sampler.sample_batch_uint8(B, T, pix, act)
        for b in range(B):
            var prev_ok = False
            var prev_cx: Float64 = 0.0
            var prev_cy: Float64 = 0.0
            for t in range(T):
                var fo = ((b * T + t) * PIX_PER)
                var cx: Float64 = 0.0
                var cy: Float64 = 0.0
                var ok = _agent_centroid(pix, fo, cx, cy)
                # action at the PREVIOUS step drives motion into THIS frame;
                # use its last frameskip sub-action (indices 8,9).
                if ok and prev_ok and t >= 1:
                    var ao = (b * T + (t - 1)) * ACT
                    var ax = Float64(act[ao + 8])
                    var ay = Float64(act[ao + 9])
                    abs_x.add(ax, cx)
                    abs_y.add(ay, cy)
                    dly_x.add(ax, cx - prev_cx)
                    dly_y.add(ay, cy - prev_cy)
                prev_ok = ok; prev_cx = cx; prev_cy = cy

    print("-" * 70)
    print("ABSOLUTE  agent_world[t+1] = a·action[t] + b :")
    abs_x.report("x"); abs_y.report("y")
    print("DELTA     agent_world[t+1]-agent_world[t] = a·action[t] + b :")
    dly_x.report("x"); dly_y.report("y")
    print("-" * 70)
    print("High R² (~0.9+) identifies the convention; (a,b) is the denorm.")
    print("  ABSOLUTE → env_target = a·action + b")
    print("  DELTA    → env_target = agent_pos + a·action + b")

    pix.free(); act.free()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
