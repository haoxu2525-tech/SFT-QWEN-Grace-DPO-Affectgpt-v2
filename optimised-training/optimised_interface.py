"""Optimised GRACE training on the unchanged GRACE / Standard DPO interfaces.

Everything that reaches the model is unchanged: the baseline PairData tokenisation and native
AV prompt (via GRACE's GracePairData), OmniModel, OmniTrainer's media handling, the official
concatenated_inputs and BasicTrainer loop. What changes:
  - SegmentPairData attaches per-token segment weights (segments.py) and the pair weight
    (ranking.py) to every tokenised pair;
  - segment_forward is the official concatenated_forward with the per-token log-probs kept,
    returning weighted sums and weight totals instead of plain sums;
  - SegmentTrainer feeds the weighted per-token means to GRACE's unchanged grace_loss and
    multiplies each pair's loss by its pair weight.
With shared_weight = 1 and all pair weights 1 this is exactly setting C (GRACE loss with
length_normalize: true); checked in test_optimised_cpu.py.
"""
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
GRACE = Path(os.environ.get("GRACE_ROOT", HERE.parent / "vendor" / "grace")).resolve()
for path in (GRACE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import grace_interface as gi  # noqa: E402  (GRACE, unchanged; also puts the baseline on sys.path)
from grace_loss import WeightNormalizer, grace_loss  # noqa: E402
from segments import pair_spans, token_weights  # noqa: E402

base = gi.base


class SegmentPairData(gi.GracePairData):
    """GRACE's PairData unchanged; adds `segw_chosen` / `segw_rejected` (one weight per
    supervised response token) and `pair_weight` to each item."""
    def __init__(self, *args, shared_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_weight = float(shared_weight)

    def tokenize(self, prompt, chosen, rejected, truncation_mode, tokenizer, max_length, max_prompt_length):
        item = super().tokenize(prompt, chosen, rejected, truncation_mode, tokenizer,
                                max_length, max_prompt_length)
        row = self.meta[json.loads(prompt)["id"]]
        spans_c, spans_r = pair_spans(chosen, rejected, row.get("contradicted_cues"))
        for side, text, spans in (("chosen", chosen, spans_c), ("rejected", rejected, spans_r)):
            w = token_weights(tokenizer, text, spans, self.shared_weight)
            if len(w) != item["length_info"][f"{side}_after"]:
                raise ValueError(f"{row['id']}: segment weights do not align with the {side} tokens")
            if sum(w) <= 0:
                raise ValueError(f"{row['id']}: the {side} answer has no weighted token")
            item[f"segw_{side}"] = w
        item["pair_weight"] = float(row.get("pair_weight", 1.0))
        return item


def segment_forward(trainers, model, batch):
    """The official concatenated_forward (same concatenation, same fp32 log-softmax and
    gather as _get_batch_logps), keeping per-token values. Returns weighted log-prob sums and
    weight totals for chosen and rejected."""
    cat = trainers.concatenated_inputs(batch)
    with model.media_context(batch["media"] + batch["media"]):
        logits = model(cat["concatenated_input_ids"],
                       attention_mask=cat["concatenated_attention_mask"]).logits.to(torch.float32)
    labels = cat["concatenated_labels"][:, 1:].clone()
    logits = logits[:, :-1, :]
    mask = labels != -100
    labels[~mask] = 0
    per_token = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    weights = torch.zeros_like(per_token)
    rows = list(batch["segw_chosen"]) + list(batch["segw_rejected"])
    for i, w in enumerate(rows):
        positions = mask[i].nonzero(as_tuple=True)[0]
        if len(positions) != len(w):
            raise ValueError("Segment weights do not match the supervised tokens")
        weights[i, positions] = torch.tensor(w, dtype=weights.dtype, device=weights.device)
    wsum, wtot = (per_token * weights).sum(-1), weights.sum(-1)
    n = batch["chosen_input_ids"].shape[0]
    return wsum[:n], wsum[n:], wtot[:n], wtot[n:]


class PrecomputedSegmentReference(torch.nn.Module):
    """Reference weighted sums computed beforehand with the same weights and the same forward."""
    def __init__(self, table):
        super().__init__()
        self.table = table

    def lookup(self, ids, device):
        rows = [self.table[i] for i in ids]
        return tuple(torch.tensor([r[j] for r in rows], dtype=torch.float32, device=device) for j in range(4))


def segment_loss(pc, pr, rc, rr, wc, wr, margin, chosen_score, pair_weight, loss_cfg, normalizer=None):
    """GRACE's unchanged grace_loss on weighted per-token means, times the pair weight.
    The means are passed with token count 1 and length_normalize off, so grace_loss neither
    re-normalises nor changes them; its NLL term becomes the weighted mean NLL of the chosen
    answer (mostly its differing segments)."""
    cfg = dataclasses.replace(loss_cfg, length_normalize=False)
    pcm, prm, rcm, rrm = pc / wc, pr / wr, rc / wc, rr / wr
    ones = torch.ones_like(pcm)
    losses, stats = grace_loss(pcm, prm, rcm, rrm, margin, ones, cfg, normalizer=normalizer,
                               rejected_tokens=ones, chosen_score=chosen_score)
    return losses * pair_weight, stats, (pcm, prm)


def segment_trainer_class(trainers, loss_cfg, log_path=None):
    class SegmentTrainer(trainers.OmniTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.grace_norm = WeightNormalizer(loss_cfg.ema_decay)
            self.grace_log = Path(log_path) if log_path else None

        def get_batch_metrics(self, batch, loss_config, train=True):
            split = "train" if train else "eval"
            pc, pr, wc, wr = segment_forward(trainers, self.policy, batch)
            dev = pc.device
            if isinstance(self.reference_model, PrecomputedSegmentReference):
                rc, rr, _, _ = self.reference_model.lookup(batch["id"], dev)
            else:
                with torch.no_grad():
                    rc, rr, _, _ = segment_forward(trainers, self.reference_model, batch)
            margin = torch.tensor(batch["grace_margin"], dtype=torch.float32, device=dev)
            score = torch.tensor(batch["grace_chosen_score"], dtype=torch.float32, device=dev)
            pw = torch.tensor(batch["pair_weight"], dtype=torch.float32, device=dev)
            losses, stats, (pcm, prm) = segment_loss(pc, pr, rc, rr, wc, wr, margin, score, pw, loss_cfg,
                                                     normalizer=self.grace_norm if train else None)
            n_c = torch.tensor([len(w) for w in batch["segw_chosen"]], dtype=torch.float32, device=dev)
            metrics = {
                f"rewards_{split}/chosen": stats["reward_chosen"].cpu().tolist(),
                f"rewards_{split}/rejected": stats["reward_rejected"].cpu().tolist(),
                f"rewards_{split}/accuracies": (stats["reward_chosen"] > stats["reward_rejected"]).float().cpu().tolist(),
                f"rewards_{split}/margins": (stats["reward_chosen"] - stats["reward_rejected"]).cpu().tolist(),
                f"logps_{split}/chosen": pcm.detach().cpu().tolist(),
                f"logps_{split}/rejected": prm.detach().cpu().tolist(),
                f"loss/{split}": losses.detach().cpu().tolist(),
                f"grace_{split}/w_q": stats["w_q"].cpu().tolist(),
                f"grace_{split}/w_p": stats["w_p"].cpu().tolist(),
                f"grace_{split}/nll_active": stats["nll_active"].cpu().tolist(),
                f"segment_{split}/chosen_weight_share": (wc / n_c).detach().cpu().tolist(),
                f"segment_{split}/pair_weight": pw.cpu().tolist(),
            }
            if train and self.grace_log is not None:
                with self.grace_log.open("a", encoding="utf-8") as stream:
                    for i, pid in enumerate(batch["id"]):
                        stream.write(json.dumps({"id": pid, "type": batch["grace_type"][i],
                                                 "pref_prob": float(stats["pref_prob"][i]),
                                                 "w_q": float(stats["w_q"][i]), "w_p": float(stats["w_p"][i]),
                                                 "pair_weight": float(pw[i])}) + "\n")
            return losses.mean(), metrics

    return SegmentTrainer


@contextlib.contextmanager
def segment_interfaces(preference_datasets, trainers, data, loss_cfg, log_path=None):
    """Baseline registrations (unchanged) + SegmentTrainer, restored on exit."""
    with base.interfaces(preference_datasets, trainers, data):
        if hasattr(trainers, "SegmentTrainer"):
            raise RuntimeError("SegmentTrainer is already registered")
        trainers.SegmentTrainer = segment_trainer_class(trainers, loss_cfg, log_path)
        try:
            yield
        finally:
            del trainers.SegmentTrainer


def precompute_segment_logps(trainers, preference_datasets, model, tokenizer, cfg, out_path,
                             splits=("train", "test"), tag="reference"):
    """One no-grad pass over every pair of `splits` with segment_forward (GRACE's
    precompute_reference with weighted values): {id: [chosen wsum, rejected wsum,
    chosen weight total, rejected weight total]}."""
    table = {}
    model.eval()
    for split in splits:
        iterator = preference_datasets.get_batch_iterator(
            names=cfg.datasets, tokenizer=tokenizer, split=split, batch_size=1, shuffle=False,
            max_length=cfg.max_length, max_prompt_length=cfg.max_prompt_length, sft_mode=False,
            n_epochs=1, silent=True, cache_dir=cfg.local_dirs[0])
        for batch in iterator:
            device = next(model.parameters()).device
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                c, r, wc, wr = segment_forward(trainers, model, batch)
            table[batch["id"][0]] = [float(c[0]), float(r[0]), float(wc[0]), float(wr[0])]
            if len(table) % 500 == 0:
                print(f"[{tag}] {len(table)} pairs", flush=True)
    Path(out_path).write_text(json.dumps(table), encoding="utf-8")
    return table


def preference_probabilities(policy_table, reference_table, beta):
    """sigma(beta * h) per pair with the loss's weighted per-token means."""
    import math
    out = {}
    for pid, (pc, pr, wc, wr) in policy_table.items():
        rc, rr = reference_table[pid][0], reference_table[pid][1]
        x = beta * ((pc - rc) / wc - (pr - rr) / wr)
        out[pid] = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
    return out
