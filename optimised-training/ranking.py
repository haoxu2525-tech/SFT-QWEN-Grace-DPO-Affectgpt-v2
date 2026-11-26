"""Changes 2 and 4 - pairs from the within-clip ranking, scored with a length-penalised label reward.

Change 4 (reward). AffectGPT-R1's accuracy reward multiplies the emotion-wheel score by a
length penalty when the answer predicts more labels than the ground truth:
    r_label' = r_label * P,   P = 1                      if n_pred <= n_gt
                              P = n_gt / n_pred           (linear, R1's P2)
                              P = log n_gt / log n_pred   (log, R1's P3)
n_pred / n_gt are the distinct labels AffectGPT's extractor returned for the answer and the
ground truth (labels.jsonl, written by GRACE's unchanged score_labels.py), so nothing is
re-extracted. GRACE's r_label (per-sample wheel F1) is otherwise unchanged.

Change 2 (pairs). GRACE pairs a clip only when one of its K answers is absolutely bad
(r_label <= label_bad); ~80% of ovlabel clips then give nothing. Here every clip's answers
are ranked by R = r_label' + alpha * r_ground and every pair whose reward gap is strictly
greater than min_margin (0.3; smaller gaps are mostly scoring noise) is a candidate pair (the
all-pairs decomposition of the clip's ranking), so a clip contributes as soon as its answers
differ clearly. The margin is always the real reward gap (GRACE's h_min_margin floor is not
used: it would create pairs and weights from no gap). Kept from GRACE:
  - the chosen side never has a contradicted cue or low grounding (it must be "eligible");
  - if no answer reaches label_good, the original ground-truth answer joins the ranking as
    the top item (GRACE's GT fallback).
Each clip keeps its max_pairs_per_clip largest-gap pairs. Pair weight (mean 1 over the round):
    margin_weighting: pair weight proportional to the reward gap (a 0.9 gap counts 3x a 0.3 gap);
    clip_normalize:   times 1/(pairs of its clip), so a clip with many pairs does not count more.
"""
import json
import math
from pathlib import Path
import random


def length_penalty(n_pred, n_gt, mode):
    if mode in (None, "none") or not n_pred or not n_gt or n_pred <= n_gt:
        return 1.0
    if mode == "linear":
        return n_gt / n_pred
    if mode == "log":
        return math.log(n_gt) / math.log(n_pred)
    raise ValueError(f"Unknown length_penalty mode: {mode}")


def apply_length_penalty(clips, label_rows, mode):
    """Scale every candidate's r_label in place; the raw value is kept as r_label_raw."""
    lab = {r["uid"]: r for r in label_rows}
    for _, recs in clips.values():
        for rec in recs:
            row = lab[rec["uid"]]
            n_pred, n_gt = len(set(row.get("labels") or [])), len(set(row.get("gt_labels") or []))
            rec["r_label_raw"] = rec["r_label"]
            rec["length_penalty"] = length_penalty(n_pred, n_gt, mode)
            rec["r_label"] = rec["r_label"] * rec["length_penalty"]
    return clips


def rank_pairs(original, recs, cfg, reward, make_row):
    """All pairs of one clip from its ranking, largest reward gap first."""
    items = []
    for rec in recs:
        k = rec["uid"].rsplit(":", 1)[1]
        eligible = rec["n_contradicted"] == 0 and (rec["r_ground"] is None or rec["r_ground"] >= cfg["ground_good"])
        items.append(dict(rec, key=k, R=reward(rec, cfg), eligible=eligible, source="policy"))
    if not any(it["eligible"] and it["r_label"] >= cfg["label_good"] for it in items):
        gt_n = next((r["n_gt_labels"] for r in recs if r.get("n_gt_labels") is not None), None)
        items.append({"key": "gt", "uid": "gt", "response": original["ground_truth_answer"],
                      "R": 1.0 + cfg["alpha"] * cfg["gt_ground_assumed"], "r_label": 1.0,
                      "n_contradicted": 0, "contradicted_cues": [], "eligible": True,
                      "source": "gt", "n_labels": gt_n})
    found = []
    for a in items:
        if not a["eligible"]:
            continue
        for b in items:
            if b is a or b["response"].strip() == a["response"].strip():
                continue
            margin = a["R"] - b["R"]
            if b["r_label"] <= cfg["label_bad"]:
                ptype = "L"
            elif b["r_label"] >= cfg["label_good"] and b["n_contradicted"] >= 1:
                ptype = "H"
            else:
                ptype = "R"   # ranked: neither absolutely bad nor hallucinated, just worse
            if margin <= cfg["min_margin"] + 1e-9:   # strictly greater; 1e-9 absorbs float error
                continue
            found.append((margin, a, b, ptype))
    found.sort(key=lambda x: (-x[0], x[1]["key"], x[2]["key"]))
    found = found[: cfg["max_pairs_per_clip"]]
    rows = []
    for margin, a, b, ptype in found:
        row = make_row(original, ptype, a["response"], b["response"], float(margin), {
            "chosen_source": a["source"], "chosen_uid": a["uid"], "rejected_uid": b["uid"],
            "r_chosen": a["R"], "r_rejected": b["R"],
            "chosen_n_labels": a.get("n_labels"), "rejected_n_labels": b.get("n_labels"),
            "contradicted_cues": b.get("contradicted_cues", []), "clip_pairs": len(found)})
        row["id"] = f"{original['id']}#{a['key']}>{b['key']}"
        rows.append(row)
    return rows


def assign_pair_weights(rows, clip_normalize, margin_weighting=True):
    """pair_weight = margin (if margin_weighting) x 1/(pairs of the clip) (if clip_normalize),
    rescaled to mean 1 over the round's rows. Replayed rows keep their own margin and clip share."""
    raw = []
    for r in rows:
        w = float(r["margin"]) if margin_weighting else 1.0
        if clip_normalize:
            w /= max(1, int(r.get("clip_pairs") or 1))
        raw.append(w)
    if min(raw) <= 0:
        raise ValueError("Pair weights must be positive")
    mean = sum(raw) / len(raw)
    for r, w in zip(rows, raw):
        r["pair_weight"] = w / mean
    return rows


def build(round_dir, cfg, batch_size, seed, replay_rows=(), bp=None):
    """Round pairs: ranking pairs + replay, complete batches only (no anchors)."""
    if bp is None:
        import build_pairs as bp
    if cfg.get("anchor_ratio", 0) not in (0, 0.0, None):
        raise ValueError("The optimised method runs without anchor pairs (pairs.anchor_ratio: 0)")
    rng = random.Random(seed)
    round_dir = Path(round_dir)
    labels = bp.read_jsonl(round_dir / "labels.jsonl")
    clips = bp.scored_clips(bp.read_jsonl(round_dir / "candidates.jsonl"), labels,
                            bp.read_jsonl(round_dir / "claims.jsonl"), bp.read_jsonl(cfg["verify_cache"]))
    apply_length_penalty(clips, labels, cfg.get("length_penalty", "linear"))
    pairs = []
    for clip in sorted(clips):
        original, recs = clips[clip]
        pairs.extend(rank_pairs(original, recs, cfg, bp.reward, bp.make_row))
    used = {p["id"] for p in pairs}
    replay = []
    for r in list(replay_rows)[: int(cfg["replay_max_fraction"] * max(1, len(pairs)))]:
        r = dict(r, id=r["id"] + "@replay", replayed=True)
        if r["id"] not in used:
            replay.append(r)
            used.add(r["id"])
    allp = pairs + replay
    rng.shuffle(allp)
    keep = len(allp) - len(allp) % batch_size
    allp, dropped = allp[:keep], allp[keep:]
    if not allp:
        raise RuntimeError("Fewer pairs than one training batch")
    assign_pair_weights(allp, cfg.get("clip_normalize", True), cfg.get("margin_weighting", True))
    bp.write_jsonl(round_dir / "pairs.jsonl", allp)
    bp.write_jsonl(round_dir / "pairs.dropped.jsonl", dropped)
    pens = [rec["length_penalty"] for _, recs in clips.values() for rec in recs]
    per_clip = {}
    for p in pairs:
        per_clip[p["source_id"]] = per_clip.get(p["source_id"], 0) + 1
    ov = [c for c, (o, _) in clips.items() if o.get("label_type") == "ovlabel"]
    summary = {"clips": len(clips), "pairs": len(allp),
               "by_type": {t: sum(p["type"] == t for p in allp) for t in "LHR"},
               "dropped_incomplete_batch": len(dropped), "replayed": len(replay),
               "clips_with_pairs": len(per_clip),
               "ovlabel_clips_with_pairs": sum(c in per_clip for c in ov), "ovlabel_clips": len(ov),
               "chosen_from_policy": sum(p.get("chosen_source") == "policy" for p in pairs),
               "chosen_from_gt": sum(p.get("chosen_source") == "gt" for p in pairs),
               "length_penalty": {"mode": cfg.get("length_penalty", "linear"),
                                  "penalised_answers": sum(x < 1 for x in pens), "answers": len(pens),
                                  "mean_factor": sum(pens) / len(pens) if pens else None},
               "label_counts": bp.label_count_summary(pairs + replay)}
    (round_dir / "pairs-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
