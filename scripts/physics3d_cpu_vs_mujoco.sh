#!/usr/bin/env bash
# CPU step, ours vs MuJoCo 3.10.0, interleaved rounds, one process per leg.
#
#   pixi run bash scripts/physics3d_cpu_vs_mujoco.sh
#   ROUNDS=5 STEPS=50000 MODEL_GROUPS="so101" pixi run bash scripts/physics3d_cpu_vs_mujoco.sh
#
# Builds one binary per model group (benchmarks/physics3d_cpu/bench_*.mojo),
# then for each round runs OURS then MuJoCo on every model, and folds the
# RESULT lines with benchmarks/physics3d_cpu_vs_mujoco_table.py.
#
# ⚠ INTERLEAVED, MIN REPORTED. Identical code has drifted 1.4-1.7x across a
# session on this machine (mojo_rl/physics3d/PERFORMANCE.md §8). Running all
# of ours and then all of MuJoCo would charge that drift to one side.
#
# ⚠ RUN FROM THE REPO ROOT, INSIDE pixi. Mesh assets resolve by repo-relative
# path -- from anywhere else the STLs fail to load, the engine warns, and the
# benchmark silently measures a model with no mesh collision. And the bare
# `mojo` on PATH outside pixi is a stale release that cannot build this tree.
set -uo pipefail

OUT=${OUT:-physics3d_cpu_vs_mujoco}
ROUNDS=${ROUNDS:-3}
WARMUP=${WARMUP:-2000}
STEPS=${STEPS:-20000}
MODEL_GROUPS=${MODEL_GROUPS:-"gym so101 contact"}
SKIP_BUILD=${SKIP_BUILD:-0}
mkdir -p "$OUT"

case "$(command -v mojo)" in
  *.pixi*) ;;
  *) echo "!! mojo is not pixi's -- run as: pixi run bash $0"; exit 1 ;;
esac
[ -f pixi.toml ] || { echo "!! run from the repo root"; exit 1; }

# group:name:xml[:warmup:steps:rounds]
#
# ⚠ THE PARK ROWS CARRY THEIR OWN HORIZON. Their props fall from z = 50 m and
# the first lands at step 1596 (MuJoCo, ctrl = 0.1); past that the scene is a
# contact scene, and at k >= 6 our side saturates MAX_CONTACTS while MuJoCo
# keeps counting. 100 + 700 + 700 = 1500 steps stays clear, and 8 in-process
# rounds (each a reset) give the short timed region a real minimum.
MODELS=(
  gym:walker2d:mojo_rl/envs/walker2d/assets/walker2d.xml
  gym:hopper:mojo_rl/envs/hopper/assets/hopper.xml
  gym:half_cheetah:mojo_rl/envs/half_cheetah/assets/half_cheetah.xml
  gym:ant:mojo_rl/envs/ant/assets/ant.xml
  gym:humanoid:mojo_rl/envs/humanoid/assets/humanoid.xml
  so101:so_arm101:mojo_rl/envs/robots/assets/so_arm101.xml
  so101:so_arm101_f64:mojo_rl/envs/robots/assets/so_arm101.xml
  so101:park_k0:mojo_rl/envs/robots/assets/so101_park_k0.xml:100:700:8
  so101:park_k3:mojo_rl/envs/robots/assets/so101_park_k3.xml:100:700:8
  so101:park_k6:mojo_rl/envs/robots/assets/so101_park_k6.xml:100:700:8
  so101:park_k9:mojo_rl/envs/robots/assets/so101_park_k9.xml:100:700:8
  contact:sawyer_reach:mojo_rl/envs/metaworld/assets/sawyer_reach.xml
  contact:dog_stand:mojo_rl/envs/dm_control/assets/dog_stand_walk.xml
  contact:humanoid_cmu:mojo_rl/envs/dm_control/assets/humanoid_cmu.xml
)

if [ "$SKIP_BUILD" = 0 ]; then
  for g in $MODEL_GROUPS; do
    echo "== build $g"
    t0=$(date +%s)
    mojo build -I . -I benchmarks "benchmarks/physics3d_cpu/bench_$g.mojo" -o "$OUT/bench_$g" \
      || { echo "!! build $g failed"; exit 1; }
    echo "   $(( $(date +%s) - t0 )) s"
  done
fi

RES="$OUT/results.txt"
: > "$RES"
for r in $(seq "$ROUNDS"); do
  for entry in "${MODELS[@]}"; do
    IFS=: read -r g name xml w st rd <<< "$entry"
    case " $MODEL_GROUPS " in *" $g "*) ;; *) continue ;; esac
    w=${w:-$WARMUP}; st=${st:-$STEPS}; rd=${rd:-1}
    echo "-- round $r  $name"
    "$OUT/bench_$g" "$name" "$w" "$st" "$rd" | grep '^RESULT' | tee -a "$RES"
    python benchmarks/physics3d_cpu_vs_mujoco.py "$name" "$xml" "$w" "$st" "$rd" \
      | grep '^RESULT' | tee -a "$RES"
  done
done

echo
python benchmarks/physics3d_cpu_vs_mujoco_table.py "$RES"
