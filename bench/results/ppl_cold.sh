#!/bin/bash
set -e
D=bench/results/ppl_cold
mkdir -p $D
export OMLX_EXPERT_STREAMING_COLD_ROOT='$PWD/bench/results/cold_tier_4M/expert_cold'
for i in 1 2; do
  echo "=== ppl cold 4M base $i $(date +%H:%M:%S) ==="
  .venv/bin/python bench/ppl_expert_streaming.py --streaming --model qwen-jang4m --cold-tier none --budget 0 --max-windows 24 --corpus bench/corpus/pg1342.txt --out $D/4m_base_$i.json 2>&1 | tail -n 2
  echo "=== ppl cold 4M tier3 $i $(date +%H:%M:%S) ==="
  .venv/bin/python bench/ppl_expert_streaming.py --streaming --model qwen-jang4m --cold-tier 3 --budget 0 --max-windows 24 --corpus bench/corpus/pg1342.txt --out $D/4m_tier3_$i.json 2>&1 | tail -n 2
done
echo COLD_PPL_DONE
