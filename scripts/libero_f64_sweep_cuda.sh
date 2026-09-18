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

say "5. the sweep (expect a long compile: two elliptic Newton instantiations)"
echo "to undo the two flags afterwards:"
echo "  git checkout -- $SAP $NS"
pixi run -e nvidia mojo run -I . "$SAP" "$DUMP" 1 1 2

# ── IF STEP 5 FAILS TO COMPILE OR LAUNCH ──────────────────────────────────
# * "DType.float64 is not supported for <fn> on NVIDIA GPU" — another float64
#   gap like the `cos` one that made flipping the global DTYPE impossible. The
#   named function is in the SOLVER's call graph; `pow` in the joint-limit
#   impedance path is the known candidate. Report the symbol.
# * shared memory / "exceeds" at launch — the blocked kernel roughly doubles
#   its threadgroup footprint at float64 (MC 56 needed 32,852 B at float32)
#   against CUDA's 48 KB static limit. Lower the cap; this state has ncon 18,
#   so 24 truncates nothing:
#       sed -i 's/^comptime MC = 56$/comptime MC = 24/' tools/tasks/solve_at_pose.mojo
#   (MC is NOT a compile-time lever — measured 54.80 s at 24 against 55.18 s at
#   48 — so change it only for the memory limit.)
