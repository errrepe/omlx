#!/bin/bash
set -e
D=bench/results/coldtier_matrix/4m
mkdir -p $D
export OMLX_EXPERT_STREAMING_COLD_ROOT='$PWD/bench/results/cold_tier_4M/expert_cold'
COMMON='--model qwen-jang4m --budget 0 --decode 96 --prompt-len short --min-free-gb 12'
run_arm() {
  local name=$1; shift
  echo "=== 4m $name $(date +%H:%M:%S) ==="
  .venv/bin/python bench/bench_expert_streaming.py $COMMON "$@" --out $D/$name.json 2>&1 | grep -E "^decode" || true
}
run_arm warmup
for i in 1 2 3; do
  run_arm base_$i --cold-tier none
  run_arm cold3_$i --cold-tier 3
done
echo COLD_MATRIX_DONE
