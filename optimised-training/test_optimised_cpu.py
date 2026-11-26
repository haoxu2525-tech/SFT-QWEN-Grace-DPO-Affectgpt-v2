"""CPU checks of the optimised method (torch + transformers, no 7B model, no GPU).

    python test_optimised_cpu.py
    OMNI_TOKENIZER=/path/to/Qwen2.5-Omni-7B python test_optimised_cpu.py   # also the real tokenizer checks

Checked:
  - segments: word diff of label / description answers, contradicted-cue spans;
  - token weights follow the official tokenisation, one EOS weight, differing words weighted 1;
  - segment_forward equals the official concatenated_forward + _get_batch_logps sums when every
    weight is 1, and the weighted sums otherwise (official functions loaded from vendor/dpo);
  - segment_loss with shared_weight 1 and pair weight 1 equals GRACE's loss of setting C
    (length_normalize: true); pair weights scale each pair's loss;
  - the post-training preference probability equals the loss's sigma(beta*h);
  - length penalty (linear / log / none);
  - ranking pairs: relative pairs without an absolutely bad answer, GT fallback, eligibility,
    H margin floor, cap, clip-normalised weights, full build() on a synthetic round.
"""
import ast
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest

import torch

import optimised_interface as oi
import ranking
import segments
import build_pairs as bp
from grace_loss import GraceLossConfig, grace_loss


def official_functions():
    """_get_batch_logps / concatenated_inputs / pad_to_length from the official DPO source,
    executed on their own (the full module needs wandb/omegaconf, which the CPU check lacks)."""
    dpo = oi.base.DPO
    import typing
    ns = {"torch": torch, "Dict": typing.Dict, "Union": typing.Union, "List": typing.List}
    for file, names in ((dpo / "utils.py", {"pad_to_length"}),
                        (dpo / "trainers.py", {"_get_batch_logps", "concatenated_inputs"})):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if len(body) != len(names):
            raise RuntimeError(f"official functions missing in {file}")
        exec(compile(ast.Module(body=body, type_ignores=[]), str(file), "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in ("pad_to_length", "_get_batch_logps", "concatenated_inputs")})


class TinyLM(torch.nn.Module):
    """Stand-in model with OmniModel's call shape (media_context + forward(ids, attention_mask))."""
    def __init__(self, vocab=50):
        super().__init__()
        torch.manual_seed(0)
        self.emb, self.out, self.media = torch.nn.Embedding(vocab, 16), torch.nn.Linear(16, vocab), None

    def media_context(self, media):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            self.media = media
            yield
            self.media = None
        return ctx()

    def forward(self, input_ids, attention_mask):
        assert self.media is not None and len(self.media) == input_ids.shape[0]
        return types.SimpleNamespace(logits=self.out(torch.tanh(self.emb(input_ids))))


def fake_batch(off, n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    batch = {"media": [{}] * n, "segw_chosen": [], "segw_rejected": []}
    for side in ("chosen", "rejected"):
        ids, labels = [], []
        for _ in range(n):
            p, r = int(torch.randint(3, 6, (1,), generator=g)), int(torch.randint(2, 7, (1,), generator=g))
            seq = torch.randint(0, 50, (p + r,), generator=g).tolist()
            ids.append(seq)
            labels.append([-100] * p + seq[p:])
            batch[f"segw_{side}"].append([float(torch.rand(1, generator=g)) for _ in range(r)])
        L = max(map(len, ids))
        batch[f"{side}_input_ids"] = torch.tensor([s + [0] * (L - len(s)) for s in ids])
        batch[f"{side}_attention_mask"] = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids])
        batch[f"{side}_labels"] = torch.tensor([s + [-100] * (L - len(s)) for s in labels])
    return batch


class Segments(unittest.TestCase):
    def test_label_answers_differ_only_in_labels(self):
        a = "The character's emotional state is happy, excited."
        b = "The character's emotional state is sad, angry."
        sa, sb = segments.diff_spans(a, b)
        self.assertEqual([a[s:e] for s, e in sa], ["happy,", "excited."])
        self.assertEqual([b[s:e] for s, e in sb], ["sad,", "angry."])

    def test_shared_label_and_case_punctuation(self):
        a = "Emotional state: Happy, relaxed"
        b = "emotional state: happy. tense"
        sa, sb = segments.diff_spans(a, b)
        self.assertEqual([a[s:e] for s, e in sa], ["relaxed"])
        self.assertEqual([b[s:e] for s, e in sb], ["tense"])

    def test_contradicted_cue_spans(self):
        text = "He is frowning and Frowning again while smiling."
        spans = segments.cue_spans(text, ["frowning", ""])
        self.assertEqual([text[s:e].lower() for s, e in spans], ["frowning", "frowning"])
        sc, sr = segments.pair_spans("He smiles.", text, ["smiling"])
        self.assertIn((text.index("smiling"), text.index("smiling") + 7), sr)

    def test_token_weights_with_offsets(self):
        class Tok:  # whitespace tokenizer with offsets, enough to test the mapping
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                spans = [(m.start(), m.end()) for m in segments.WORD.finditer(text)]
                out = {"input_ids": [hash(text[s:e]) % 1000 for s, e in spans]}
                if return_offsets_mapping:
                    out["offset_mapping"] = spans
                return out
        text = "state is happy now"
        w = segments.token_weights(Tok(), text, [(9, 14)], 0.2)
        self.assertEqual(w, [0.2, 0.2, 1.0, 0.2, 0.2])   # last = EOS
        with self.assertRaises(ValueError):
            segments.token_weights(Tok(), text, [], 1.5)

    @unittest.skipUnless(os.environ.get("OMNI_TOKENIZER"), "set OMNI_TOKENIZER to a Qwen2.5-Omni tokenizer folder")
    def test_real_tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ["OMNI_TOKENIZER"])
        a = "The character's emotional state is happy, excited, relieved."
        b = "The character's emotional state is anxious, excited, nervous."
        sa, sb = segments.pair_spans(a, b)
        for text, spans in ((a, sa), (b, sb)):
            w = segments.token_weights(tok, text, spans, 0.2)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            self.assertEqual(len(w), len(ids) + 1)
            heavy = tok.decode([i for i, x in zip(ids, w) if x == 1.0])
            light = tok.decode([i for i, x in zip(ids, w) if x < 1.0])
            self.assertNotIn("excited", heavy)
            self.assertIn("excited", light)
            self.assertIn("emotional state", light)
        heavy_a = tok.decode([i for i, x in zip(tok(a, add_special_tokens=False)["input_ids"],
                                                 segments.token_weights(tok, a, sa, 0.2)) if x == 1.0])
        self.assertIn("happy", heavy_a)
        self.assertIn("relieved", heavy_a)


class Forward(unittest.TestCase):
    def setUp(self):
        self.off = official_functions()
        self.trainers = types.SimpleNamespace(concatenated_inputs=self.off.concatenated_inputs)
        self.model = TinyLM()

    def official_sums(self, batch):
        cat = self.off.concatenated_inputs(batch)
        with self.model.media_context(batch["media"] + batch["media"]):
            logits = self.model(cat["concatenated_input_ids"],
                                attention_mask=cat["concatenated_attention_mask"]).logits.to(torch.float32)
        lp = self.off._get_batch_logps(logits, cat["concatenated_labels"], average_log_prob=False)
        n = batch["chosen_input_ids"].shape[0]
        return lp[:n], lp[n:]

    def test_unit_weights_equal_official(self):
        batch = fake_batch(self.off)
        for side in ("chosen", "rejected"):
            batch[f"segw_{side}"] = [[1.0] * len(w) for w in batch[f"segw_{side}"]]
        c, r, wc, wr = oi.segment_forward(self.trainers, self.model, batch)
        oc, orr = self.official_sums(batch)
        torch.testing.assert_close(c, oc)
        torch.testing.assert_close(r, orr)
        self.assertEqual(wc.tolist(), [float(len(w)) for w in batch["segw_chosen"]])

    def test_weighted_sums(self):
        batch = fake_batch(self.off, seed=3)
        c, r, wc, wr = oi.segment_forward(self.trainers, self.model, batch)
        cat = self.off.concatenated_inputs(batch)
        with self.model.media_context(batch["media"] * 2):
            logits = self.model(cat["concatenated_input_ids"], cat["concatenated_attention_mask"]).logits.float()
        labels = cat["concatenated_labels"]
        expect = []
        for i, w in enumerate(batch["segw_chosen"] + batch["segw_rejected"]):
            pos = [t for t in range(1, labels.shape[1]) if labels[i, t] != -100]
            lps = [torch.log_softmax(logits[i, t - 1], -1)[labels[i, t]] for t in pos]
            expect.append(sum(float(x) * wt for x, wt in zip(lps, w)))
        torch.testing.assert_close(torch.cat([c, r]), torch.tensor(expect), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(wc, torch.tensor([sum(w) for w in batch["segw_chosen"]]))

    def test_misaligned_weights_fail(self):
        batch = fake_batch(self.off)
        batch["segw_chosen"][0] = batch["segw_chosen"][0] + [1.0]
        with self.assertRaises(ValueError):
            oi.segment_forward(self.trainers, self.model, batch)


class Loss(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.n = 6
        self.pc, self.pr = -torch.rand(self.n, generator=g) * 40, -torch.rand(self.n, generator=g) * 40
        self.rc, self.rr = self.pc + torch.randn(self.n, generator=g), self.pr + torch.randn(self.n, generator=g)
        self.tc = torch.randint(5, 40, (self.n,), generator=g)
        self.tr = torch.randint(5, 40, (self.n,), generator=g)
        self.margin = torch.rand(self.n, generator=g)
        self.score = torch.rand(self.n, generator=g) + 0.5
        self.cfg = GraceLossConfig(beta=2.0, length_normalize=True, nll_weight=0.05, nll_min_chosen_score=0.8)

    def test_unit_weights_equal_setting_c(self):
        ref, ref_stats = grace_loss(self.pc, self.pr, self.rc, self.rr, self.margin, self.tc, self.cfg,
                                    rejected_tokens=self.tr, chosen_score=self.score)
        got, stats, _ = oi.segment_loss(self.pc, self.pr, self.rc, self.rr, self.tc.float(), self.tr.float(),
                                        self.margin, self.score, torch.ones(self.n), self.cfg)
        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(stats["pref_prob"], ref_stats["pref_prob"])

    def test_pair_weight_scales_each_pair(self):
        pw = torch.linspace(0.5, 2.0, self.n)
        one, _, _ = oi.segment_loss(self.pc, self.pr, self.rc, self.rr, self.tc.float(), self.tr.float(),
                                    self.margin, self.score, torch.ones(self.n), self.cfg)
        got, _, _ = oi.segment_loss(self.pc, self.pr, self.rc, self.rr, self.tc.float(), self.tr.float(),
                                    self.margin, self.score, pw, self.cfg)
        torch.testing.assert_close(got, one * pw)

    def test_gradient_only_through_policy(self):
        pc, pr = self.pc.clone().requires_grad_(), self.pr.clone().requires_grad_()
        losses, _, _ = oi.segment_loss(pc, pr, self.rc, self.rr, self.tc.float() * 0.3, self.tr.float() * 0.3,
                                       self.margin, self.score, torch.ones(self.n), self.cfg)
        losses.sum().backward()
        self.assertTrue(torch.isfinite(pc.grad).all() and pr.grad.abs().sum() > 0)

    def test_post_training_probability_matches_loss(self):
        wc, wr = self.tc.float() * 0.4, self.tr.float() * 0.7
        _, stats, _ = oi.segment_loss(self.pc, self.pr, self.rc, self.rr, wc, wr, self.margin, self.score,
                                      torch.ones(self.n), self.cfg)
        pol = {str(i): [float(self.pc[i]), float(self.pr[i]), float(wc[i]), float(wr[i])] for i in range(self.n)}
        ref = {str(i): [float(self.rc[i]), float(self.rr[i]), float(wc[i]), float(wr[i])] for i in range(self.n)}
        pref = oi.preference_probabilities(pol, ref, self.cfg.beta)
        for i in range(self.n):
            self.assertAlmostEqual(pref[str(i)], float(stats["pref_prob"][i]), places=5)


def rec(uid, r_label, n_contradicted=0, r_ground=None, response=None, n_labels=3, cues=()):
    return {"uid": uid, "response": response or f"answer {uid}", "r_label": r_label, "hit": r_label > 0,
            "n_labels": n_labels, "n_gt_labels": 3, "r_ground": r_ground, "n_contradicted": n_contradicted,
            "contradicted_cues": list(cues), "n_claims": 0, "n_affective": 0, "n_affective_supported": 0}


ORIGINAL = {k: f"<{k}>" for k in bp.ORIGINAL_FIELDS}
ORIGINAL.update(id="clip1", ground_truth_answer="GT answer", label_type="ovlabel")
CFG = {"label_good": 0.5, "label_bad": 0.2, "ground_good": 0.6, "ground_default": 0.5, "gt_ground_assumed": 1.0,
       "alpha": 0.5, "min_margin": 0.3, "max_pairs_per_clip": 4,
       "clip_normalize": True, "margin_weighting": True, "length_penalty": "linear", "anchor_ratio": 0, "replay_max_fraction": 0.1}


class Ranking(unittest.TestCase):
    def pairs(self, recs, **over):
        return ranking.rank_pairs(ORIGINAL, recs, dict(CFG, **over), bp.reward, bp.make_row)

    def test_length_penalty(self):
        self.assertEqual(ranking.length_penalty(4, 2, "linear"), 0.5)
        self.assertEqual(ranking.length_penalty(2, 4, "linear"), 1.0)
        self.assertAlmostEqual(ranking.length_penalty(4, 2, "log"), math.log(2) / math.log(4))
        self.assertEqual(ranking.length_penalty(9, 1, "none"), 1.0)
        self.assertEqual(ranking.length_penalty(3, 0, "linear"), 1.0)
        self.assertEqual(ranking.length_penalty(2, 4, "symmetric"), 0.5)
        self.assertEqual(ranking.length_penalty(6, 3, "symmetric"), 0.5)
        self.assertEqual(ranking.length_penalty(3, 3, "symmetric"), 1.0)

    def test_label_count_rule(self):
        gt3 = lambda n: {"n_pred": n, "n_gt": 3}
        self.assertTrue(ranking.count_ok(gt3(3), gt3(5), "closer_to_gt"))    # rejected over-predicts
        self.assertTrue(ranking.count_ok(gt3(4), gt3(1), "closer_to_gt"))
        self.assertTrue(ranking.count_ok(gt3(2), gt3(4), "closer_to_gt"))    # equally far: kept
        self.assertFalse(ranking.count_ok(gt3(1), gt3(3), "closer_to_gt"))   # would teach fewer labels
        self.assertFalse(ranking.count_ok(gt3(6), gt3(3), "closer_to_gt"))   # would teach more labels
        self.assertTrue(ranking.count_ok(gt3(1), {"n_pred": None, "n_gt": 3}, "closer_to_gt"))
        self.assertTrue(ranking.count_ok(gt3(1), gt3(3), "none"))
        with self.assertRaises(ValueError):
            ranking.count_ok(gt3(1), gt3(3), "fewer")

    def test_count_rule_in_ranking(self):
        recs = [dict(rec("clip1:0", 0.9), n_pred=1, n_gt=3), dict(rec("clip1:1", 0.3), n_pred=3, n_gt=3),
                dict(rec("clip1:2", 0.2), n_pred=5, n_gt=3)]
        kept = {(r["chosen_uid"], r["rejected_uid"]) for r in self.pairs(recs, label_count_rule="closer_to_gt")}
        self.assertEqual(kept, {("clip1:0", "clip1:2")})   # 1 label vs 3: dropped; 1 vs 5: equally far, kept
        self.assertEqual(len(self.pairs(recs, label_count_rule="none")), 2)

    def test_relative_pair_without_bad_answer(self):
        # GRACE: no answer <= 0.2, so no pair. Ranking: 0.9 vs 0.55 is a pair (gap 0.35 > 0.3);
        # 0.9 vs 0.75 and 0.75 vs 0.55 are not.
        rows = self.pairs([rec("clip1:0", 0.9), rec("clip1:1", 0.55), rec("clip1:2", 0.75)])
        self.assertEqual([(r["chosen_uid"], r["rejected_uid"], r["type"]) for r in rows],
                         [("clip1:0", "clip1:1", "R")])
        self.assertEqual(rows[0]["id"], "clip1#0>1")
        self.assertEqual(bp.select_pairs(ORIGINAL, [rec("clip1:0", 0.9), rec("clip1:1", 0.55)],
                                         dict(CFG, type_priority=["H", "L", "S"], use_spurious=True,
                                              max_pairs_per_clip=2, h_min_margin=0.3)), [])

    def test_gap_must_exceed_threshold(self):
        self.assertEqual(self.pairs([rec("clip1:0", 0.8), rec("clip1:1", 0.5)]), [])   # gap exactly 0.3
        self.assertEqual(len(self.pairs([rec("clip1:0", 0.81), rec("clip1:1", 0.5)])), 1)

    def test_gt_fallback_and_types(self):
        rows = self.pairs([rec("clip1:0", 0.4), rec("clip1:1", 0.1)])
        got = {(r["chosen_uid"], r["rejected_uid"], r["type"]) for r in rows}
        # 0.4 vs 0.1 has a gap of exactly 0.3, so only the GT pairs remain
        self.assertEqual(got, {("gt", "clip1:0", "R"), ("gt", "clip1:1", "L")})
        self.assertEqual(rows[0]["chosen"], "GT answer")

    def test_hallucinated_answer_never_chosen_and_real_margin(self):
        rows = self.pairs([rec("clip1:0", 0.7, r_ground=1.0), rec("clip1:1", 0.6, n_contradicted=1,
                                                                   r_ground=0.0, cues=["smiling"])])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["chosen_uid"], rows[0]["type"]), ("clip1:0", "H"))
        self.assertAlmostEqual(rows[0]["margin"], 0.6)   # the real gap, no floor
        # a small real gap is not raised to a pair
        self.assertEqual(self.pairs([rec("clip1:0", 0.7, r_ground=1.0),
                                     rec("clip1:1", 0.8, n_contradicted=1, r_ground=0.5)]), [])
        self.assertEqual(rows[0]["contradicted_cues"], ["smiling"])

    def test_cap_and_identical_text(self):
        recs = [rec(f"clip1:{k}", s) for k, s in enumerate([1.0, 0.75, 0.5, 0.25, 0.0])]
        recs.append(rec("clip1:5", 0.0, response="answer clip1:0"))
        rows = self.pairs(recs, max_pairs_per_clip=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["margin"], 1.0)
        self.assertTrue(all(r["clip_pairs"] == 3 for r in rows))
        self.assertFalse(any(r["chosen"] == r["rejected"] for r in rows))

    def test_pair_weights(self):
        rows = [{"clip_pairs": 1, "margin": 0.4}, {"clip_pairs": 1, "margin": 1.2},
                {"clip_pairs": 2, "margin": 0.4}, {"clip_pairs": 2, "margin": 0.8}]
        ranking.assign_pair_weights(rows, False, True)       # proportional to the gap
        self.assertAlmostEqual(sum(r["pair_weight"] for r in rows) / len(rows), 1.0)
        self.assertAlmostEqual(rows[1]["pair_weight"], 3 * rows[0]["pair_weight"])
        self.assertAlmostEqual(rows[3]["pair_weight"], 2 * rows[2]["pair_weight"])
        ranking.assign_pair_weights(rows, True, True)        # gap x clip share
        self.assertAlmostEqual(rows[0]["pair_weight"], 2 * rows[2]["pair_weight"])
        self.assertAlmostEqual(sum(r["pair_weight"] for r in rows) / len(rows), 1.0)
        ranking.assign_pair_weights(rows, True, False)       # clip share only
        self.assertAlmostEqual(rows[0]["pair_weight"], 2 * rows[3]["pair_weight"])
        ranking.assign_pair_weights(rows, False, False)
        self.assertEqual({r["pair_weight"] for r in rows}, {1.0})

    def test_default_config_weights_gap_once(self):
        import yaml
        cfg = yaml.safe_load((Path(__file__).parent / "optimised.yaml").read_text(encoding="utf-8"))
        self.assertTrue(cfg["loss"]["use_quality_weight"])
        self.assertFalse(cfg["pairs"]["margin_weighting"])
        rows = [{"margin": 0.4, "clip_pairs": 2}, {"margin": 0.9, "clip_pairs": 2}, {"margin": 1.2, "clip_pairs": 1}]
        ranking.assign_pair_weights(rows, True, margin_weighting=False)
        self.assertEqual(rows[0]["pair_weight"], rows[1]["pair_weight"])        # gap not in the pair weight
        self.assertAlmostEqual(rows[2]["pair_weight"], 2 * rows[0]["pair_weight"])
        w_q = [1 / (1 + math.exp(-4 * g)) for g in (0.3, 0.8, 1.5)]               # GRACE quality weight, eta 4
        self.assertEqual([round(x, 2) for x in w_q], [0.77, 0.96, 1.0])

    def test_build_synthetic_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cands, labels = [], []
            for c in range(6):
                row = dict(ORIGINAL, id=f"c{c}", candidates=[f"ans {c} {k}" for k in range(4)])
                cands.append(row)
                for k in range(4):
                    n_pred = 6 if k == 0 else 2
                    labels.append({"uid": f"c{c}:{k}", "r_label": [0.9, 0.7, 0.65, 0.2][k], "hit": True,
                                   "labels": [f"l{i}" for i in range(n_pred)], "gt_labels": ["a", "b", "c"]})
            bp.write_jsonl(d / "candidates.jsonl", cands)
            bp.write_jsonl(d / "labels.jsonl", labels)
            bp.write_jsonl(d / "claims.jsonl", [])
            cfg = dict(CFG, verify_cache=str(d / "verify.jsonl"))
            summary = ranking.build(d, cfg, batch_size=4, seed=1, bp=bp)
            pairs = bp.read_jsonl(d / "pairs.jsonl")
            self.assertEqual(len(pairs) % 4, 0)
            self.assertEqual(summary["length_penalty"]["penalised_answers"], 6)
            # answer 0 (r_label 0.9, 6 labels vs 3) is penalised to 0.45, so answers 1 / 2 rank first
            self.assertTrue(all(p["chosen_uid"].endswith(":1") or p["chosen_uid"].endswith(":2") for p in pairs))
            self.assertAlmostEqual(sum(p["pair_weight"] for p in pairs) / len(pairs), 1.0)
            with self.assertRaises(ValueError):
                ranking.build(d, dict(cfg, anchor_ratio=0.2), batch_size=4, seed=1, bp=bp)


class PerTypeMethod(unittest.TestCase):
    """ovlabel clips: ranking pairs, count penalty, segment weights; description clips: GRACE's
    own pairing (setting C), raw r_label, every token weighted 1; ovlabel clips get k_ovlabel answers."""

    def round_dir(self, d):
        cands, labels, claims = [], [], []
        for c in range(4):
            ltype = "ovlabel" if c % 2 == 0 else "description"
            n = 8 if ltype == "ovlabel" else 4
            cands.append(dict(ORIGINAL, id=f"c{c}", label_type=ltype, candidates=[f"ans {c} {k}" for k in range(n)]))
            for k in range(n):
                score = [0.9, 0.55, 0.5, 0.1, 0.85, 0.5, 0.45, 0.3][k]
                labels.append({"uid": f"c{c}:{k}", "r_label": score, "hit": True,
                               "labels": ["a", "b", "c", "d", "e", "f"] if k == 0 else ["a", "b", "c"],
                               "gt_labels": ["a", "b", "c"]})
                claims.append({"uid": f"c{c}:{k}", "id": f"c{c}", "k": k, "claims": []})
        bp.write_jsonl(d / "candidates.jsonl", cands)
        bp.write_jsonl(d / "labels.jsonl", labels)
        bp.write_jsonl(d / "claims.jsonl", claims)

    def test_split_by_question_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.round_dir(d)
            cfg = dict(CFG, verify_cache=str(d / "verify.jsonl"), ranking_types=["ovlabel"],
                       length_penalty="symmetric", label_count_rule="closer_to_gt", margin_weighting=False)
            summary = ranking.build(d, cfg, batch_size=1, seed=1, bp=bp, shared_weight=0.2)
            pairs = bp.read_jsonl(d / "pairs.jsonl")
            ov = [p for p in pairs if p["label_type"] == "ovlabel"]
            de = [p for p in pairs if p["label_type"] == "description"]
            self.assertTrue(ov and de)
            self.assertEqual({(p["method"], p["shared_weight"]) for p in ov}, {("ranking", 0.2)})
            self.assertEqual({(p["method"], p["shared_weight"], p["clip_pairs"]) for p in de}, {("grace", 1.0, 1)})
            # GRACE pairing on description: answer 0 (0.9, raw - not penalised) vs the only bad answer (0.1)
            self.assertEqual({(p["type"], p["chosen"], p["rejected"]) for p in de},
                             {("L", "ans 1 0", "ans 1 3"), ("L", "ans 3 0", "ans 3 3")})
            # ranking on ovlabel: answer 0 is penalised (6 labels vs 3 -> 0.45), so it is never chosen over 0.85
            self.assertFalse(any(p["chosen_uid"].endswith(":0") for p in ov))
            self.assertEqual(summary["by_method"], {"ranking": len(ov), "grace": len(de)})
            self.assertEqual(summary["length_penalty"]["answers"], 16)   # ovlabel answers only
            self.assertEqual(summary["length_penalty"]["penalised_answers"], 2)

    def test_grace_pairs_equal_build_pairs(self):
        recs = [rec("clip1:0", 0.9), rec("clip1:1", 0.1), rec("clip1:2", 0.8, n_contradicted=1, cues=["smiling"])]
        gcfg = dict(CFG, **ranking.GRACE_DEFAULTS)
        expected = bp.select_pairs(ORIGINAL, recs, gcfg)
        got = ranking.grace_pairs(ORIGINAL, recs, CFG, bp)
        self.assertEqual([{k: v for k, v in r.items() if k not in ("method", "shared_weight", "clip_pairs")} for r in got],
                         expected)
        self.assertEqual([r["type"] for r in got], ["H", "L"])

    def test_extra_sampling_merge(self):
        import run_optimised as ro
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = [dict(ORIGINAL, id="o1", label_type="ovlabel", candidates=["a", "b", "c", "d"], end_tokens_stripped=1),
                    dict(ORIGINAL, id="d1", label_type="description", candidates=["w", "x", "y", "z"])]
            bp.write_jsonl(d / "candidates.jsonl", rows)
            seen = {}

            def fake_candidates(cfg, r, sub):
                seen.update(k=cfg["k"], seed=cfg["seed"], ids=json.loads((sub / "ids.json").read_text()))
                bp.write_jsonl(sub / "candidates.jsonl", [dict(rows[0], candidates=["e", "f", "g", "h"],
                                                                end_tokens_stripped=2)])
            old = ro.rg.stage_candidates
            ro.rg.stage_candidates = fake_candidates
            try:
                ro.stage_extra({"k": 4, "k_ovlabel": 8, "seed": 0}, 1, d)
                ro.stage_extra({"k": 4, "k_ovlabel": 8, "seed": 0}, 1, d)   # rerun: same result
            finally:
                ro.rg.stage_candidates = old
            out = {r["id"]: r for r in bp.read_jsonl(d / "candidates.jsonl")}
            self.assertEqual(seen, {"k": 4, "seed": 500000, "ids": ["o1"]})
            self.assertEqual(out["o1"]["candidates"], list("abcdefgh"))
            self.assertEqual(out["o1"]["end_tokens_stripped"], 3)
            self.assertEqual(out["d1"]["candidates"], list("wxyz"))
            self.assertEqual(len(bp.read_jsonl(d / "candidates.k.jsonl")[0]["candidates"]), 4)

    def test_row_shared_weight(self):
        data = object.__new__(oi.SegmentPairData)
        data.shared_weight = 0.2
        self.assertEqual(data.row_shared_weight({}), 0.2)
        self.assertEqual(data.row_shared_weight({"shared_weight": 1.0}), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
