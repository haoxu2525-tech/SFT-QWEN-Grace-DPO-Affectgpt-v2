#!/bin/sh
# Operations script on the pod: state of the v2 runs. Per setting and round: finished stages,
# held-out validation wheel F1 (all / ovlabel / description clips), benchmark units and score; the validation score of SFT epoch 30 (round 0).
S=/workspace/grace-ops/sched-v2
PY=/workspace/grace-env/bin/python
val() {   # validation.json -> "all 61.23 ovlabel 63.10 description 59.40"
  [ -s "$1" ] && $PY -c "import json,sys; s=json.load(open(sys.argv[1]))['scores']; print(' '.join('%s %.2f' % (k, v['ew_f1']) for k, v in s.items()))" "$1" || echo -
}
for W in /workspace/grace-work/v2/*/; do
  W=${W%/}; X=$(basename $W)
  [ -f $W/optimised.yaml ] || continue
  for r in 1 2; do
    [ -d $W/round$r ] || [ $r = 1 ] || continue
    st=$(ls -a $W/round$r 2>/dev/null | grep "^\.done-" | sed "s/.done-//" | tr "\n" " ")
    sc=$( [ -s $W/eval$r.scores.txt ] && tail -1 $W/eval$r.scores.txt | awk -F"&" "{print \$NF}" | tr -d " " || echo -)
    u=$(ls $W/eval$r.u*.done 2>/dev/null | wc -l)
    echo "v2-$X r$r stages: ${st:-none}| validation $(val $W/validation/round$r/validation.json) | eval units $u/5 | score $sc"
  done
done
for f in /workspace/grace-work/v2/*/validation/round0/validation.json; do
  [ -s "$f" ] && echo "SFT epoch 30 validation (round 0): $(val $f)"
done
for g in 0 1 2; do
  echo "gpu$g: $(tail -1 $S/lane$g.log 2>/dev/null | cut -c1-200)"
done
echo "reserved: $(ls $S/*/gpu*.reserved 2>/dev/null | awk -F/ '{print $(NF-1)"/"$NF}' | tr "\n" " ") busy: $(ls $S/*/busy.* 2>/dev/null | awk -F/ '{print $(NF-1)"/"$NF}' | tr "\n" " ")"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h /workspace | tail -1
