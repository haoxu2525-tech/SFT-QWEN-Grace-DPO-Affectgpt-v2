# Optimised training

GRACE-DPO with three method changes, chosen from the GRACE training curves and logs
(`vendor/grace/results/loss-curves`). Everything else is GRACE setting C (no anchors, 10% warm-up,
per-token log-probs, beta 2), which is the comparison point.

## What stays aligned with AffectGPT

| | |
|---|---|
| Training data | the same MERCaptionPlus clip IDs, original questions, prompts, media and annotations; every pair is checked against the original AffectGPT row before training (GRACE's `validate_against_original`, unchanged) |
| Clips per round | the same seed and order as GRACE, so round r uses the same 2,000 clips |
| Model path | baseline `PairData` (native AV prompt), `OmniModel`, `OmniTrainer`, official DPO `BasicTrainer` loop, RMSprop, lr 5e-6, batch 8 x 8 accumulation, max length 4096 without truncation, reference = verified SFT epoch 30 |
| Sampling, label extraction, claims, verification, export | GRACE's stages, called unchanged (`run_grace.py` functions) |
| Evaluation | `run_optimised.py eval` = GRACE `do_eval` = the baseline's unchanged v2 MER-UniBench `infer` + `evaluate` |

## What changes

**1. Claim-level (segment-weighted) DPO** (`segments.py`, `optimised_interface.py`).
Each response token gets weight 1 if it lies in a part where chosen and rejected differ (a word-level
diff), or inside a cue the grounding verifier marked as contradicted in the rejected answer. Every
other token gets `segment.shared_weight` (0.2, which is RLHF-V's DDPO with gamma 5). The log-prob of
an answer is the weighted per-token mean. GRACE's loss code then runs unchanged on these values,
and so does its chosen-NLL term, which now mostly pulls up the chosen answer's differing segments.
With `shared_weight: 1` and every pair weight at 1, this is exactly setting C (checked on CPU).

**2. Pairs from the within-clip ranking** (`ranking.py`).
The K answers of a clip are ranked by R = r_label' + 0.5 r_ground. Every pair with a reward gap of at
least 0.2 is a candidate, and the 2 largest-gap pairs per clip are kept. Previously one of the answers
had to be absolutely bad (r_label <= 0.2). As in GRACE, the chosen side never carries a contradicted
cue, and the ground truth joins the ranking only if no answer reaches 0.5. Each pair is weighted
1/(pairs of its clip), rescaled to a mean of 1.

**4. Length-penalised label reward** (`ranking.py`).
AffectGPT-R1's penalty: r_label' = r_label x n_gt / n_pred when the answer predicts more distinct
labels than the ground truth (`length_penalty: log` gives R1's P3, `none` turns it off). The label
counts come from GRACE's unchanged `score_labels.py` output, so nothing is re-extracted.

## Measured on setting C's round 1 (same 2,000 clips and answers, pairs stage only, no training)

| | C (GRACE) | ranking, no penalty | ranking + penalty (default) |
|---|---|---|---|
| pairs (cap 2 per clip) | 1,128 | 3,168 | 3,488 |
| ovlabel clips with a pair | ~20% | 661 / 984 (67%) | 798 / 984 (81%) |
| answers penalised | — | — | 2,348 / 8,000 (mean factor 0.89) |
| ovlabel pairs: chosen vs rejected label count | 3.62 vs 2.79 | 3.56 vs 3.43 | 2.97 vs 3.98 |
| ovlabel pairs where chosen has fewer labels | 23% | 38% | 64% |

A cap of 4 gives 5,408 / 6,128 pairs with the same clip coverage.

**Risk to watch:** with the penalty, the chosen answer has fewer labels than the rejected one in 64%
of ovlabel pairs. Standard DPO's label shrink (3.26 → 2.91 labels per answer) went with a lower
benchmark score. The penalty only acts above the ground-truth count, but check the label count of
the evaluated answers after round 1. `pairs-summary.json` reports these counts every round.

**Cost:** about 3x C's pairs per round, so training takes about 3x as long (C: ~2 h per round on one
A40, so about 6 h). Sampling, labels, claims and verification are unchanged.

## Files

| File | |
|---|---|
| `run_optimised.py` | driver: `round`, `eval`, `status`, `reuse` (GRACE's driver with `pairs` and `train` replaced) |
| `ranking.py` | changes 2 and 4: length-penalised reward, ranking pairs, pair weights |
| `segments.py` | change 1: diff / contradicted-cue spans → per-token weights |
| `optimised_interface.py` | `SegmentPairData`, `segment_forward`, `SegmentTrainer`, weighted reference and post-training passes |
| `train_optimised.py` | GRACE's `train_grace.py` path with `SegmentTrainer` |
| `optimised.yaml` | settings (C's values + `segment:` + the new `pairs:` keys) |
| `test_optimised_cpu.py` | CPU checks |

## Run (on the GRACE pod)

The GRACE code is pinned as `vendor/grace` (commit 29605ad). On the pod, the existing
`/workspace/grace` checkout is used instead: `export GRACE_ROOT=/workspace/grace`.

```bash
cd optimised-training
export GRACE_ROOT=/workspace/grace WANDB_MODE=offline
/workspace/grace-env/bin/python run_optimised.py reuse --config optimised.yaml --round 1 \
    --from /workspace/grace-work/abl/C/round1          # same round-1 answers as C
/workspace/grace-env/bin/python run_optimised.py round --config optimised.yaml --round 1
/workspace/grace-env/bin/python run_optimised.py eval  --config optimised.yaml --round 1
/workspace/grace-env/bin/python run_optimised.py round --config optimised.yaml --round 2
```

Outputs are the same as GRACE (`/workspace/grace-work/optimised/roundN/`), plus `pair_weight` and
`clip_pairs` in `pairs.jsonl`, `segment_*/chosen_weight_share` in the training stats, and
`optimised-run.json`.

## Verification status

`python test_optimised_cpu.py` passes (19 checks, set `OMNI_TOKENIZER` for the real-tokenizer
check):
- segment weights follow the official tokenisation, including the EOS token;
- `segment_forward` equals the official `concatenated_forward` sums when every weight is 1, and the
  hand-computed weighted sums otherwise;
- the loss equals setting C's GRACE loss when every weight is 1;
- pair weights scale each pair's loss;
- the post-training preference probability matches the loss;
- the ranking rules and length penalty behave as described;
- a full `build()` runs on a synthetic round.

The pairs stage was also run read-only on C's real round-1 files (numbers above).

Not run yet: the 7B GPU training path (data loading with real media, `SegmentTrainer` inside the
official loop, the export), and any benchmark score.
