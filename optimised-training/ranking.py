"""Changes 2 and 4 - pairs from the within-clip ranking, scored with a length-penalised label reward.

Change 4 (reward). The emotion-wheel score is multiplied by a label-count penalty:
    r_label' = r_label * P,   P = min(n_pred, n_gt) / max(n_pred, n_gt)   (symmetric, the default)
AffectGPT-R1's own forms penalise only over-prediction (P = 1 if n_pred <= n_gt, else
n_gt / n_pred = linear, R1's P2, or log n_gt / log n_pred = log, R1's P3). On setting C's
round 1 that one-sided penalty, added to F1's own precision term, made the chosen side of
ovlabel pairs average 2.94 labels against 3.37 for the ground truth: the pairs taught fewer
labels, the Standard DPO shrink. The symmetric form keeps chosen at the ground-truth count
(3.66 vs 3.70) while still penalising over-prediction.
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
Label-count rule (label_count_rule). A preference must not teach "predict fewer labels" (Standard
DPO shrank answers from 3.26 to 2.91 labels and lost accuracy) nor "predict more". Counts only
mean something relative to the ground truth, so with `closer_to_gt` a pair is kept only if the
chosen answer's distinct label count is at least as close to the ground truth's as the rejected
answer's: |n_chosen - n_gt| <= |n_rejected - n_gt|. A rejected answer that over-predicts (the
length penalty's target) still makes a pair; a chosen answer that drops below the ground-truth
count while the rejected one matches it does not. Pairs with an unknown count are kept.
`none` turns the rule off.
Each clip keeps its max_pairs_per_clip largest-gap pairs. Pair weight (mean 1 over the round):
    margin_weighting: pair weight proportional to the reward gap (off by default: the gap is
                      weighted by GRACE's bounded sigmoid(eta * gap) in the loss instead);
    clip_normalize:   times 1/(pairs of its clip), so a clip with many pairs does not count more.
"""
import json
import math
from pathlib import Path
import random


def length_penalty(n_pred, n_gt, mode):
    if mode in (None, "none") or not n_pred or not n_gt:
        return 1.0
    if mode == "symmetric":   # both directions: min/max of the two counts
        return min(n_pred, n_gt) / max(n_pred, n_gt)
    if n_pred <= n_gt:
        return 1.0
    if mode == "linear":
        return n_gt / n_pred
    if mode == "log":
        return math.log(n_gt) / math.log(n_pred)
    raise ValueError(f"Unknown length_penalty mode: {mode}")


def apply_length_penalty(clips, label_rows, mode, label_types=None):
    """Scale the r_label of every candidate of the given question types in place (all types if
    None); the raw value is kept as r_label_raw. Label counts are recorded for every candidate."""
    lab = {r["uid"]: r for r in label_rows}
    for original, recs in clips.values():
        if label_types is not None and original.get("label_type") not in label_types:
            mode_here = "none"
        else:
            mode_here = mode
        for rec in recs:
            row = lab[rec["uid"]]
            n_pred, n_gt = len(set(row.get("labels") or [])), len(set(row.get("gt_labels") or []))
            rec["n_pred"], rec["n_gt"] = n_pred, n_gt   # distinct counts (penalty and count rule)
            rec["r_label_raw"] = rec["r_label"]
            rec["length_penalty"] = length_penalty(n_pred, n_gt, mode_here)
            rec["r_label"] = rec["r_label"] * rec["length_penalty"]
    return clips


def count_ok(a, b, rule):
    """label_count_rule: is the chosen answer's label count at least as close to the ground truth's?"""
    if rule in (None, "none"):
        return True
    if rule != "closer_to_gt":
        raise ValueError(f"Unknown label_count_rule: {rule}")
    n_gt = a.get("n_gt") or b.get("n_gt")
    if not n_gt or not a.get("n_pred") or not b.get("n_pred"):
        return True
    return abs(a["n_pred"] - n_gt) <= abs(b["n_pred"] - n_gt)


def rank_pairs(original, recs, cfg, reward, make_row):
    """All pairs of one clip from its ranking, largest reward gap first."""
    items = []
    for rec in recs:
        k = rec["uid"].rsplit(":", 1)[1]
        eligible = rec["n_contradicted"] == 0 and (rec["r_ground"] is None or rec["r_ground"] >= cfg["ground_good"])
        items.append(dict(rec, key=k, R=reward(rec, cfg), eligible=eligible, source="policy"))
    if not any(it["eligible"] and it["r_label"] >= cfg["label_good"] for it in items):
        gt_n = next((r["n_gt_labels"] for r in recs if r.get("n_gt_labels") is not None), None)
        gt_d = next((r["n_gt"] for r in recs if r.get("n_gt")), None)
        items.append({"key": "gt", "uid": "gt", "response": original["ground_truth_answer"],
                      "R": 1.0 + cfg["alpha"] * cfg["gt_ground_assumed"], "r_label": 1.0,
                      "n_contradicted": 0, "contradicted_cues": [], "eligible": True,
                      "source": "gt", "n_labels": gt_n, "n_pred": gt_d, "n_gt": gt_d})
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
            if not count_ok(a, b, cfg.get("label_count_rule", "none")):
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
            "chosen_n_distinct": a.get("n_pred"), "rejected_n_distinct": b.get("n_pred"), "gt_n_distinct": a.get("n_gt") or b.get("n_gt"),
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


def count_vs_gt(pairs):
    """Distinct label counts of chosen / rejected / ground truth, per question type (logging)."""
    out = {}
    for key in ("all", "ovlabel", "description"):
        rows = [p for p in pairs if p.get("chosen_n_distinct") and p.get("rejected_n_distinct") and p.get("gt_n_distinct")
                and key in ("all", p.get("label_type"))]
        if not rows:
            continue
        n = len(rows)
        c, r, g = ([p[f] for p in rows] for f in ("chosen_n_distinct", "rejected_n_distinct", "gt_n_distinct"))
        out[key] = {"pairs": n, "chosen_mean": sum(c) / n, "rejected_mean": sum(r) / n, "gt_mean": sum(g) / n,
                    "chosen_fewer": sum(x < y for x, y in zip(c, r)) / n,
                    "chosen_more": sum(x > y for x, y in zip(c, r)) / n,
                    "chosen_below_gt": sum(x < z for x, z in zip(c, g)) / n,
                    "rejected_below_gt": sum(y < z for y, z in zip(r, g)) / n,
                    "chosen_above_gt": sum(x > z for x, z in zip(c, g)) / n,
                    "rejected_above_gt": sum(y > z for y, z in zip(r, g)) / n}
    return out


GRACE_DEFAULTS = {"min_margin": 0.2, "h_min_margin": 0.3, "use_spurious": True,
                  "type_priority": ["H", "L", "S"], "max_pairs_per_clip": 2}


def grace_pairs(original, recs, cfg, bp):
    """GRACE's unchanged pairing (setting C) for one clip: typed L/H/S pairs against an
    absolutely bad answer, GRACE's thresholds, raw r_label (no count penalty). Each row is
    trained like setting C: every response token weighted 1, clip share 1."""
    gcfg = {**cfg, **GRACE_DEFAULTS, **(cfg.get("grace") or {})}
    rows = bp.select_pairs(original, [dict(r, r_label=r.get("r_label_raw", r["r_label"])) for r in recs], gcfg)
    for row in rows:
        row.update(method="grace", shared_weight=1.0, clip_pairs=1)
    return rows


def build(round_dir, cfg, batch_size, seed, replay_rows=(), bp=None, shared_weight=None):
    """Round pairs, complete batches only (no anchors). Clips whose question type is in
    cfg["ranking_types"] (default: all) get ranking pairs; the others GRACE's own pairs.
    `shared_weight` is written into every ranking row (the trainer's default otherwise)."""
    if bp is None:
        import build_pairs as bp
    if cfg.get("anchor_ratio", 0) not in (0, 0.0, None):
        raise ValueError("The optimised method runs without anchor pairs (pairs.anchor_ratio: 0)")
    rng = random.Random(seed)
    round_dir = Path(round_dir)
    labels = bp.read_jsonl(round_dir / "labels.jsonl")
    clips = bp.scored_clips(bp.read_jsonl(round_dir / "candidates.jsonl"), labels,
                            bp.read_jsonl(round_dir / "claims.jsonl"), bp.read_jsonl(cfg["verify_cache"]))
    ranking_types = cfg.get("ranking_types")
    apply_length_penalty(clips, labels, cfg.get("length_penalty", "linear"), ranking_types)
    pairs = []
    for clip in sorted(clips):
        original, recs = clips[clip]
        if ranking_types is None or original.get("label_type") in ranking_types:
            rows = rank_pairs(original, recs, cfg, bp.reward, bp.make_row)
            for row in rows:
                row["method"] = "ranking"
                if shared_weight is not None:
                    row["shared_weight"] = float(shared_weight)
        else:
            rows = grace_pairs(original, recs, cfg, bp)
        pairs.extend(rows)
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
    pens = [rec["length_penalty"] for o, recs in clips.values() for rec in recs
            if ranking_types is None or o.get("label_type") in ranking_types]
    per_clip = {}
    for p in pairs:
        per_clip[p["source_id"]] = per_clip.get(p["source_id"], 0) + 1
    ov = [c for c, (o, _) in clips.items() if o.get("label_type") == "ovlabel"]
    summary = {"clips": len(clips), "pairs": len(allp),
               "by_type": {t: sum(p["type"] == t for p in allp) for t in "LHSR"},
               "by_question_type": {q: sum(p["label_type"] == q for p in allp) for q in ("ovlabel", "description")},
               "by_method": {m: sum(p.get("method") == m for p in allp) for m in ("ranking", "grace")},
               "dropped_incomplete_batch": len(dropped), "replayed": len(replay),
               "clips_with_pairs": len(per_clip),
               "ovlabel_clips_with_pairs": sum(c in per_clip for c in ov), "ovlabel_clips": len(ov),
               "chosen_from_policy": sum(p.get("chosen_source") == "policy" for p in pairs),
               "chosen_from_gt": sum(p.get("chosen_source") == "gt" for p in pairs),
               "length_penalty": {"mode": cfg.get("length_penalty", "linear"),
                                  "penalised_answers": sum(x < 1 for x in pens), "answers": len(pens),
                                  "mean_factor": sum(pens) / len(pens) if pens else None},
               "label_count_rule": cfg.get("label_count_rule", "none"),
               "label_counts": bp.label_count_summary(pairs + replay),
               "label_counts_vs_gt": count_vs_gt(pairs)}
    (round_dir / "pairs-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
