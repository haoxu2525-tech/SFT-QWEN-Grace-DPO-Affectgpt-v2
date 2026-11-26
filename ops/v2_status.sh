#!/bin/sh
# Operations script on the pod: state of the v2 runs (each setting on its own GPU).
S=/workspace/grace-ops/sched-v2
for X in C CK8 CK12; do
  W=/workspace/grace-work/v2/$X
  for r in 1 2; do
    st=$(ls -a $W/round$r 2>/dev/null | grep "^\.done-" | sed "s/.done-//" | tr "\n" " ")
    sc=$( [ -s $W/eval$r.scores.txt ] && tail -1 $W/eval$r.scores.txt | awk -F"&" "{print \$NF}" | tr -d " " || echo -)
    u=$(ls $W/eval$r.u*.done 2>/dev/null | wc -l)
    echo "v2-$X r$r stages: ${st:-none}| eval units $u/5 | score $sc"
  done
  echo "   reserved: $(ls $S/$X/gpu*.reserved 2>/dev/null | xargs -n1 basename 2>/dev/null | tr "\n" " ")busy: $(ls $S/$X/busy.* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr "\n" " ")"
  tail -1 $S/$X/train.log 2>/dev/null | sed "s/^/   /"; tail -1 $S/$X/eval.log 2>/dev/null | sed "s/^/   /"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h /workspace | tail -1
