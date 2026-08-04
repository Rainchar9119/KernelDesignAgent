#!/bin/bash
export HOME=/root
NCU=/usr/local/cuda/bin/ncu
dur() {
  $NCU --metrics gpu__time_duration.sum --target-processes application-only \
    -k "regex:fused_norm_rope_flashmla" -s 20 -c 1 \
    python "$1" --num-tokens "$2" --mode "$3" 2>/dev/null \
    | awk '/gpu__time_duration.sum/{print $(NF)}' | tail -1
}
echo "N mode base_us cand_us ratio"
for MODE in decode extend; do
 for N in 32 64 128 256 512 1024 2048 4096 8192 16384; do
   b=$(dur harness/profile_baseline.py $N $MODE)
   c=$(dur harness/profile_cand.py $N $MODE)
   r=$(python3 -c "print(f'{$c/$b:.4f}')" 2>/dev/null || echo NA)
   echo "$N $MODE $b $c $r"
 done
done
