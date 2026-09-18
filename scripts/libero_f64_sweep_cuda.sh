#!/usr/bin/env bash
# The float64 elliptic leg-vs-leg sweep, from a FRESH CUDA BOX to a verdict.
#
#     bash scripts/libero_f64_sweep_cuda.sh
#
# Answers one question: is the systematic offset between `solve_newton_blocked`
# and `_newton_solve_env` on LIBERO a ROUNDING bias or a LOGIC difference? A
# rounding difference shrinks with the precision; a different computation does
# not. See `tools/tasks/solve_at_pose.mojo`'s header for the whole story.
#
# ⚠⚠ WHY THE BARE `solve_at_pose` COMMAND FAILS ON A NEW BOX. It reads a dumped
# state, and `build/` is gitignored (.gitignore:11) — so `build/diag/lr3_all.txt`
# does not survive a clone and must be REGENERATED here (step 3). The LIBERO
# meshes and textures are gitignored too (.gitignore:317, 364 of the 466 files
# under mojo_rl/tasks/libero); only the XML is tracked, so the asset pack has to
# be pulled (step 2) or the model will not parse.
#
# ⚠ THE TWO `sed`s ARE BOTH LOAD-BEARING and each is verified below:
#   * F64_GPU_SWEEP        — off, you get the "NOT DECIDED" notice, not a sweep.
#   * NEWTON_FORCE_PER_ENV — off, `solve_newton` DISPATCHES TO THE BLOCKED
#     KERNEL on NVIDIA (newton_solve.mojo:4396), so both arms would be the SAME
#     kernel and the sweep would be an identity. `_solve64_gpu` refuses to run
#     in that configuration rather than print a vacuous 0.0.
set -euo pipefail

FAMILY=libero_living_room_scene3
DUMP=build/diag/lr3_all.txt
NS=mojo_rl/physics3d/solver/newton_solve.mojo
SAP=tools/tasks/solve_at_pose.mojo

cd "$(dirname "$0")/.."
say() { printf '\n=== %s\n' "$*"; }

# ⚠ NOT `sed -i`. GNU wants `-i`, BSD/macOS wants `-i ''`, and the form that
# works on one silently eats an argument on the other. Write-and-move is the
# same on both.
sedi() {
    local expr=$1 file=$2 t
    t=$(mktemp)
    sed "$expr" "$file" > "$t" && mv "$t" "$file"
}

# A sed that matches nothing changes nothing and says nothing — which here
# would silently run the wrong experiment. Every edit is checked.
require_line() {
    local file=$1 pattern=$2
    grep -q -- "$pattern" "$file" \
        || { echo "FAILED: '$pattern' not found in $file — the file has moved" \
                  "under this script; fix the pattern rather than the file." >&2
             exit 1; }
}

say "1. dependencies"
command -v pixi >/dev/null || {
    echo "pixi is not on PATH. Install it and re-run:" >&2
    echo "  curl -fsSL https://pixi.sh/install.sh | bash" >&2
    exit 1
}
pixi install

say "2. the LIBERO asset pack (meshes + textures, gitignored)"
if [ -d mojo_rl/tasks/libero/assets/stable_hope_objects ]; then
    echo "already materialised, skipping"
else
    # `assets.kv` carries the URL and the sha256 that is the pack's identity.
    pixi run assets-pull libero
fi
[ -d mojo_rl/tasks/libero/assets/stable_hope_objects ] || {
    echo "the asset pack did not materialise — check the network and" \
         "mojo_rl/tasks/libero/assets.kv" >&2; exit 1; }

say "3. regenerate the dumped state ($DUMP)"
if [ -s "$DUMP" ]; then
    echo "already present, skipping (delete it to force a fresh dump)"
else
    mkdir -p "$(dirname "$DUMP")"
    # ⚠ `--cpu-lanes 0` MATTERS: it skips the float64 CPU rollout, which is the
    # slow part and is not what we are dumping. `--steps 2` because the state
    # under test is step 1, and `--dump-at-step -2` writes EVERY step for every
    # lane. The gate's own verdict is irrelevant here — the dump is written
    # during stepping — so a non-zero exit is tolerated on purpose.
    # A DIRECTORY, so the generated file keeps its `.mojo` extension and
    # nothing is left behind.
    tmpd=$(mktemp -d)
    trap 'rm -rf "$tmpd"' EXIT
    tmp="$tmpd/fam_$FAMILY.mojo"
    sed "s/^comptime FAMILY = .*/comptime FAMILY = \"$FAMILY\"/" \
        examples/tasks/libero_family_batched.mojo > "$tmp"
    grep -q "comptime FAMILY = \"$FAMILY\"" "$tmp" \
        || { echo "the FAMILY sed did not take" >&2; exit 1; }
    pixi run -e nvidia mojo run -I . "$tmp" \
        --steps 2 --cpu-lanes 0 --dump-at-step -2 --dump-state "$DUMP" || true
fi
# lanes 1 and 2 at step 1 are what the sweep replays.
for want in "QPOS lane 1 step 1 " "QVEL lane 1 step 1 " \
            "QPOS lane 2 step 1 " "QVEL lane 2 step 1 "; do
    grep -q "^$want" "$DUMP" \
        || { echo "$DUMP has no '$want' row — the dump did not complete" >&2
             exit 1; }
done
echo "dump OK ($(wc -l < "$DUMP") lines)"

say "4. arm the sweep (both flags, both verified)"
require_line "$SAP" '^comptime F64_GPU_SWEEP: Bool = '
require_line "$NS"  '^comptime NEWTON_FORCE_PER_ENV: Bool = '
sedi 's/^comptime F64_GPU_SWEEP: Bool = False$/comptime F64_GPU_SWEEP: Bool = True/' "$SAP"
sedi 's/^comptime NEWTON_FORCE_PER_ENV: Bool = False$/comptime NEWTON_FORCE_PER_ENV: Bool = True/' "$NS"
grep -q '^comptime F64_GPU_SWEEP: Bool = True$' "$SAP" \
    || { echo "F64_GPU_SWEEP is still False — the sweep would not run" >&2; exit 1; }
grep -q '^comptime NEWTON_FORCE_PER_ENV: Bool = True$' "$NS" \
    || { echo "NEWTON_FORCE_PER_ENV is still False — both arms would be the" \
              "BLOCKED kernel and the sweep would be vacuous" >&2; exit 1; }
echo "both flags set"

say "5. shrink the per-env kernel's PER-THREAD FRAME (or it OOMs at launch)"
# ⚠⚠ THIS IS NOT AN OPTIMISATION, IT IS WHAT MAKES THE FLOAT64 RUN LAUNCH AT
# ALL. `CUDA_ERROR_OUT_OF_MEMORY` *at kernel launch* is the local-memory
# RESERVATION, not a buffer allocation — the per-env leg (the one
# NEWTON_FORCE_PER_ENV turns on) keeps its big matrices in a per-thread frame,
# and every float in it doubles at float64.
#
#   `JT_PC` is the lever that exists for exactly this, added when the same frame
#   blew Metal's per-thread stack. False = one all-contact tangent cache of
#   MC*NT*NV; True = a per-contact buffer of NT*nv, refilled per contact.
#   Measured for this model (MC 56, NT 2, NV 45) at float64:
#         JT_PC=False  39.4 KB per thread
#         JT_PC=True    0.7 KB per thread
#   The default is `not has_nvidia_gpu_accelerator()` (newton_solve.mojo:4239),
#   so CUDA gets False and Metal gets True — which also means the Metal figures
#   this sweep is compared against were produced WITH JT_PC=True. Forcing it
#   here makes the two platforms agree rather than diverge.
#
#   `MC` cuts the same array linearly (56 -> 24 is 39.4 -> 16.9 KB) and the
#   device workspace with it. VERIFIED LOSSLESS on Metal at MC=24: identical
#   147/148 line-search evaluations and the same 5.7220458984375e-06.
#   ⚠ THE FLOOR IS THE CONTACT COUNT: this state has ncon 18, and an MC below
#   it TRUNCATES SILENTLY, so do not go under 20.
#   ⚠ MC is NOT a compile-time lever (54.80 s at 24 against 55.18 s at 48) —
#   change it for memory, never for build speed.
#
# The three NV*NV matrices (47.5 KB per thread at float64) are inherent to the
# per-env leg and no flag removes them. If it still OOMs after both edits, drop
# to one lane: `sed 's/^comptime BATCH = 2$/comptime BATCH = 1/'` on $SAP — the
# divergence follows the STATE and reproduces at BATCH 1 (see $SAP's header),
# so a single lane is still a valid comparison, just without lane 1 as control.
MC=${MC:-24}
[ "$MC" -ge 20 ] || { echo "MC=$MC is below the ncon 18 of this state and" \
                           "would truncate contacts silently" >&2; exit 1; }
require_line "$NS" '^        JT_PC = not has_nvidia_gpu_accelerator(),$'
require_line "$SAP" '^comptime MC = '
sedi 's/^        JT_PC = not has_nvidia_gpu_accelerator(),$/        JT_PC = True,  # forced: float64 per-thread frame (see libero_f64_sweep_cuda.sh)/' "$NS"
sedi "s/^comptime MC = .*/comptime MC = $MC/" "$SAP"
grep -q '^        JT_PC = True,' "$NS" \
    || { echo "JT_PC was not forced — the per-env float64 kernel will OOM at" \
              "launch" >&2; exit 1; }
grep -q "^comptime MC = $MC\$" "$SAP" \
    || { echo "MC was not set to $MC" >&2; exit 1; }
echo "JT_PC forced True, MC=$MC"

say "6. the sweep (expect a long compile: two elliptic Newton instantiations)"
echo "to undo EVERY edit this script made:"
echo "  git checkout -- $SAP $NS"
pixi run -e nvidia mojo run -I . "$SAP" "$DUMP" 1 1 2

# ── IF STEP 5 FAILS TO COMPILE OR LAUNCH ──────────────────────────────────
# * "DType.float64 is not supported for <fn> on NVIDIA GPU" — another float64
#   gap like the `cos` one that made flipping the global DTYPE impossible. The
#   named function is in the SOLVER's call graph; `pow` in the joint-limit
#   impedance path is the known candidate. Report the symbol.
# * CUDA_ERROR_OUT_OF_MEMORY *at launch* — step 5 exists for this; if it still
#   happens after JT_PC and MC, try `MC=20` and then BATCH=1 as step 5 says.
# * shared memory / "exceeds" from the BLOCKED kernel — that one keeps its
#   matrices in THREADGROUP memory instead (which is why it is the OOM-safe leg
#   at scale), roughly doubling from 32,852 B at float32 against CUDA's 48 KB
#   static limit. Lowering MC is the same fix.
