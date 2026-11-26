# GRACE-DPO v2 on SFT Qwen2.5-Omni-7B (AffectGPT)

The second version of GRACE-DPO. It changes the preference objective and pair construction, and
keeps the AffectGPT training data, the training path and the evaluation unchanged.

| Folder | |
|---|---|
| [`optimised-training/`](optimised-training/README.md) | the v2 method: claim-level (segment-weighted) DPO, pairs from the within-clip ranking, AffectGPT-R1 length-penalised label reward |
| `vendor/grace` | GRACE-DPO v1 ([`SFT-Qwen2.5Omni7B-GRACE-DPO-AffectGPT`](https://github.com/haoxu2525-tech/SFT-Qwen2.5Omni7B-GRACE-DPO-AffectGPT) @ `29605ad`), imported unchanged; it pins the Standard DPO baseline, the official DPO code, the SFT interface and AffectGPT |

Why these changes: the v1 training curves (`vendor/grace/results/loss-curves`) show the following.
- Within a round, reward accuracy only reaches 0.65–0.7 and the loss stays near ln 2.
- While the rejected reward falls, the chosen reward falls too: likelihood displacement on pairs
  that share most of their tokens.
- About 80% of ovlabel clips give no pair, because none of their K answers is absolutely bad.

Baseline results to compare with (9-dataset MER-UniBench mean): SFT epoch 30 72.81 / 73.12 / 73.10,
Standard DPO 71.78, GRACE setting C 74.15 / 76.12 / 73.21 (rounds 1–3).
