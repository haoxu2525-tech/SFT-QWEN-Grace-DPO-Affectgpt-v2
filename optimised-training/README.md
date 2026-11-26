# Optimised training

GRACE-DPO with three method changes, chosen from the GRACE training curves and logs
(`vendor/grace/results/loss-curves`). Everything else is GRACE setting C (no anchors, 10% warm-up,
per-token log-probs, beta 2).

**The changes apply to ovlabel-question clips only.** The benchmark scores only label lists, and
GRACE's ablations moved with the ovlabel pairs. Round 1: +80 to +154 ovlabel pairs (CK8 / CK12 / CK16)
gave +1.2 to +1.9, while dropping 564 description pairs (D) changed the score by -0.28.
| Question type | Answers per clip | Pairs | Loss |
|---|---|---|---|
| ovlabel | 8 (`k_ovlabel`) | ranking (change 2), label-count penalty and rule (change 4), at most 2 per clip | segment-weighted (change 1, shared tokens 0.2) |
| description | 4 (`k`) | GRACE's own typed L / H / S pairing with setting C's values, raw label score, at most 2 per clip | setting C's: every token weighted 1 |

The round-1 comparison point is CK8: setting C with 8 answers per ovlabel clip, on the same answers.

## What stays aligned with AffectGPT

| | |
|---|---|
| Training data | the same MERCaptionPlus clip IDs, original questions, prompts, media and annotations; every pair is checked against the original AffectGPT row before training (GRACE's `validate_against_original`, unchanged) |
| Clips per round | the same seed and order as GRACE, so round r uses the same 2,000 clips |
| Model path | baseline `PairData` (native AV prompt), `OmniModel`, `OmniTrainer`, official DPO `BasicTrainer` loop, RMSprop, lr 5e-6, batch 8 x 8 accumulation, max length 4096 without truncation, reference = verified SFT epoch 30 |
| Sampling, label extraction, claims, verification, export | GRACE's stages, called unchanged (`run_grace.py` functions). The 4 extra ovlabel answers come from a second call of GRACE's unchanged candidates stage on the ovlabel clips (same policy, seeds offset by 500000), as the CK8 runs did |
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
The K answers of a clip are ranked by R = r_label' + 0.5 r_ground. Every pair with a reward gap
**strictly greater than 0.3** is a candidate; smaller gaps are mostly scoring noise. The 2 largest-gap
pairs per clip are kept. Previously one of the answers had to be absolutely bad (r_label <= 0.2).
As in GRACE, the chosen side never carries a contradicted cue, and the ground truth joins the ranking
only if no answer reaches 0.5. The margin is always the real gap (GRACE's 0.3 floor for hallucination
pairs is not used).
Each pair's loss is weighted by **GRACE's sigmoid(4 x gap)** and by **1/(pairs of its clip)**. The
clip weights are rescaled to a mean of 1, so a clip that gives two pairs counts as much as a clip that
gives one. The sigmoid is bounded: 0.77 at a gap of 0.3, 0.96 at 0.8 and 0.998 at 1.5. A pair with a
large gap, often the ground truth against a clearly wrong answer, therefore weighs about as much as
a pair just above the threshold. The threshold decides which pairs exist, and the sigmoid only
damps the ones near it. `pairs.margin_weighting: true` (with `loss.use_quality_weight: false`)
weights linearly in the gap instead: a 0.9 gap then counts 3x a 0.3 gap.

**4. Label-count penalty on the label reward, and a label-count rule for pairs** (`ranking.py`).
- **Penalty.** r_label' = r_label x min(n_pred, n_gt) / max(n_pred, n_gt). It uses the number of
  distinct labels in the answer and in the ground truth, from GRACE's unchanged `score_labels.py`
  output, so nothing is re-extracted. Over- and under-predicting are penalised alike.
  AffectGPT-R1's one-sided form (`length_penalty: linear`, R1's P2; `log` = P3) only penalises
  over-prediction. Added to F1's own precision term, it made the pairs teach fewer labels (table
  below), so it is no longer the default.
- **Rule** (`label_count_rule: closer_to_gt`). A pair is kept only if the chosen answer's label count
  is at least as close to the ground truth's as the rejected answer's:
  |n_chosen − n_gt| <= |n_rejected − n_gt|. A pair never rewards dropping below, or rising above,
  the ground-truth count.

What the label counts should look like: the chosen side at the ground-truth count, and the rejected
side off it in both directions, so no pair set teaches "fewer" or "more" on average.
`pairs-summary.json` → `label_counts_vs_gt` reports this every round.

## Measured on setting C's round 1 (same 2,000 clips and answers, pairs stage only, no training)

All columns use a gap > 0.3 and a cap of 2 pairs per clip. Counts are distinct labels in ovlabel pairs.

| | C (GRACE) | no penalty | R1 penalty (linear) | symmetric penalty | symmetric + count rule (default) |
|---|---|---|---|---|---|
| pairs | 1,128 | 2,312 | 2,848 | 3,200 | 3,152 |
| by type L / H / R | 564 / 561 / 3 S | 1,008 / 303 / 1,001 | 1,099 / 231 / 1,518 | 1,389 / 121 / 1,690 | 1,333 / 120 / 1,699 |
| ovlabel clips with a pair | ~20% | 471 / 984 (48%) | 663 / 984 (67%) | 765 / 984 (78%) | 761 / 984 (77%) |
| chosen / rejected / GT mean count | — | 3.68 / 3.43 / 3.73 | 2.94 / 4.08 / 3.37 | 3.66 / 3.45 / 3.70 | 3.65 / 3.45 / 3.70 |
| chosen below / above GT count | — | 33% / 30% | 33% / 8% | 12% / 9% | 11% / 9% |
| rejected below / above GT count | — | 46% / 33% | 29% / 61% | 52% / 42% | 52% / 43% |

- With R1's penalty, the chosen side sits 0.43 labels below the ground truth, and the rejected side
  is mostly an over-prediction. Training on that set teaches "fewer labels". Standard DPO's label
  shrink (3.26 → 2.91 labels per answer) went with a lower benchmark score.
- With the symmetric penalty, the chosen side matches the ground-truth count in about 80% of pairs.
  Rejected answers miss it in both directions.
- The count rule removes the few remaining pairs whose chosen side is further from the ground truth
  (48 pairs).
- There are fewer H pairs because an H rejected answer needs a label score >= 0.5 after the penalty.
  Also, C raised small hallucination gaps to 0.3, while here a hallucination pair needs a real gap
  above 0.3.
- Description pairs show the same pattern: with the default, chosen / rejected / GT average
  3.61 / 3.08 / 3.66.

Still check the label count of the evaluated answers after round 1.

The table above applies the new pairing to every clip at 4 answers. The default now applies it to
ovlabel clips only, at 8 answers. Measured on CK8's round 1 (8 answers per ovlabel clip, 4 per
description clip; pairs stage only, read-only):

| | CK8 (GRACE) | default |
|---|---|---|
| pairs | 1,208 | 2,752 |
| ovlabel pairs / clips with a pair | 274 / 274 of 984 (28%; one L pair at most per clip) | 1,817 (788 L, 1,029 R) / 921 of 984 (94%) |
| description pairs | 934 (562 H, 369 L, 3 S) | 935 (562 H, 370 L, 3 S): the same GRACE pairing |
| ovlabel median reward gap | — | 0.57 |
| pair weight range (mean 1) | — | 0.74–1.48 |

**Cost:** 4 more answers for each of the ~980 ovlabel clips cost about 1.25 h of sampling per round
with 2 sampler processes (measured for CK8). Training has about 2.3x C's pairs, so about 4.5–5 h on
one A40 (C: ~2 h). Claims and verification are unchanged (ovlabel answers have no claims).

## Held-out validation (`validation.py`)

Rounds and settings were so far chosen by their MER-UniBench score, the test benchmark itself.
`run_optimised.py validate --round r` gives a score from the training corpus instead:

- **Clips:** the last 500 clips of the seeded clip order. Rounds take their 2,000 clips from the front
  of that order (10 rounds leave the last 11,256 of 31,256 unused), so no round of any setting trains
  on them, and every setting (v1 and v2, same seed) is validated on the same 500.
- **Answers:** 4 per clip, sampled by GRACE's unchanged candidates stage from round r's exported
  checkpoint (`--round 0` = SFT epoch 30), with the clip's own AffectGPT question (ovlabel or
  description, as in training). No new prompt.
- **Score:** GRACE's unchanged label stage (AffectGPT extractor + per-sample emotion-wheel F1 against
  the clip's ground-truth labels), averaged per clip and then over clips, x100. Reported for all
  clips and per question type, with the standard error over clips, in
  `<work_dir>/validation/round<r>/validation.json`.

The command also accepts a GRACE v1 config (`grace.yaml`), so v1 exports are scored on the same
clips. Sampling is at temperature 1 and not seeded to repeat, so two runs of one checkpoint differ a
little; with 500 clips x 4 answers, differences of about one point are within noise.

## Switching each change on alone

Four lists of question types in `pairs:` switch the changes independently (`[]` = off):

| Key | Change | Off means |
|---|---|---|
| `ranking_types` | 2: pairs from the within-clip ranking | GRACE's own pairs (one answer absolutely bad) |
| `penalty_types` | 4: count-penalised reward and `label_count_rule` | raw wheel F1, no count rule |
| `segment_types` | 1: `segment.shared_weight` on the pair's tokens | every token weighted 1 |
| `clip_normalize_types` | 2: pair weight x 1/(pairs of its clip) | every pair weighted alike |

The last three default to `ranking_types`, which gives the full method. With all four `[]` the
pairs are exactly GRACE v1's (checked on CK8 round 1: the same 1,208 pairs). Pairs stage on CK8
round 1's answers, no training:

| Setting | Pairs | ovlabel pairs | Note |
|---|---|---|---|
| all off (= v1 CK8) | 1,208 | 274 | |
| ranking | 2,224 | 1,290 | |
| count penalty alone | 1,352 | 420 | on GRACE's pairs |
| segment weighting alone | 1,208 | 274 | weight 0.2 on the 274 ovlabel pairs only |
| clip normalisation alone | 1,208 | 274 | **no effect**: no ovlabel clip has two GRACE pairs |
| all four (= v2 CK8) | 2,752 | 1,818 | |

Because clip normalisation does nothing without ranking, and the other two touch only 274 pairs on
their own, the comparison adds one change at a time in this order: CK8 → CK8-R (ranking) → CK8-RP
(+ count penalty) → CK8-RPS (+ segment weighting) → v2 CK8 (+ clip normalisation).

## Files

| File | |
|---|---|
| `run_optimised.py` | driver: `round`, `eval`, `status`, `reuse`, `validate` (GRACE's driver with `pairs` and `train` replaced and an `extra` sampling stage added) |
| `validation.py` | held-out validation score of a round's export |
| `ranking.py` | changes 2 and 4: label-count penalty and rule, ranking pairs, pair weights |
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
    --from /workspace/grace-work/abl/CK8/round1        # same round-1 answers as CK8 (8 ovlabel / 4 description)
/workspace/grace-env/bin/python run_optimised.py round --config optimised.yaml --round 1
/workspace/grace-env/bin/python run_optimised.py eval  --config optimised.yaml --round 1
/workspace/grace-env/bin/python run_optimised.py round --config optimised.yaml --round 2
```

Outputs are the same as GRACE (`/workspace/grace-work/optimised/roundN/`), plus `candidates.k.jsonl`
(the first 4 answers) and `extra/` (the added ovlabel answers), `method`, `shared_weight`, `pair_weight`
and `clip_pairs` in `pairs.jsonl`, `segment_*/chosen_weight_share` in the training stats, and
`optimised-run.json`.

## Verification status

`python test_optimised_cpu.py` passes (33 checks, set `OMNI_TOKENIZER` for the real-tokenizer
check):
- each of the four changes can be switched on alone, and without the three new keys the output is
  the full method as before;
- the validation clips are the end of the round order, the score is the mean over clips, and
  `validate` calls GRACE's two stages with its own answer count and local sampling only;
- segment weights follow the official tokenisation, including the EOS token;
- `segment_forward` equals the official `concatenated_forward` sums when every weight is 1, and the
  hand-computed weighted sums otherwise;
- the loss equals setting C's GRACE loss when every weight is 1;
- pair weights scale each pair's loss;
- the post-training preference probability matches the loss;
- the ranking rules (gap strictly above 0.3, real margins, clip weights, optional linear gap weights), the label-count penalty
  (symmetric / linear / log) and the closer-to-GT count rule
  behave as described;
- a full `build()` runs on a synthetic round;
- description clips get GRACE's `select_pairs` result unchanged, every token weighted 1, and no
  count penalty, while ovlabel clips get ranking pairs;
- the extra stage samples only the ovlabel clips (with offset seeds), appends 4 answers to each,
  leaves description clips at 4, and gives the same result when rerun.

The pairs stage was also run read-only on the real round-1 files of C and CK8 (numbers above).

Not run yet: the 7B GPU training path (data loading with real media, `SegmentTrainer` inside the
official loop, the export), and any benchmark score.
