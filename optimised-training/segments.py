"""Change 1 - claim-level (segment-weighted) preference: which response tokens carry the pair's difference.

Chosen and rejected answers sampled from the same policy share most of their words; a
summed or plain-mean log-prob is dominated by those shared words. Following RLHF-V's dense
DPO (DDPO), every response token gets a weight:

    1.0            the token lies in a segment where the two answers differ
                   (word-level diff of chosen vs rejected), or inside a cue of the rejected
                   answer that the grounding verifier marked as contradicted
    shared_weight  every other token, including the end-of-sequence token

and an answer's log-probability becomes the weighted per-token mean
    lp(y) = sum_t w_t log p(y_t) / sum_t w_t.
shared_weight = 0.2 is DDPO's gamma = 5 (changed segments weighted 5x the unchanged ones,
normalised by |y_u| + gamma |y_c|); shared_weight = 1 is the plain per-token mean of setting C.

Pure Python plus the tokenizer; no model code.
"""
import difflib
import re

WORD = re.compile(r"\S+")


def _key(word):
    """Words compare case-insensitively and without attached punctuation ("happy," == "Happy")."""
    stripped = re.sub(r"[^\w']", "", word.lower())
    return stripped or word


def _words(text):
    return [(m.start(), m.end(), _key(m.group())) for m in WORD.finditer(text)]


def diff_spans(a, b):
    """Character spans of the words of `a` and of `b` that are not part of the longest
    matching word sequence between them (difflib, no junk heuristic)."""
    wa, wb = _words(a), _words(b)
    sm = difflib.SequenceMatcher(None, [w[2] for w in wa], [w[2] for w in wb], autojunk=False)
    shared_a, shared_b = set(), set()
    for block in sm.get_matching_blocks():
        shared_a.update(range(block.a, block.a + block.size))
        shared_b.update(range(block.b, block.b + block.size))
    return ([(s, e) for i, (s, e, _) in enumerate(wa) if i not in shared_a],
            [(s, e) for i, (s, e, _) in enumerate(wb) if i not in shared_b])


def cue_spans(text, cues):
    """Every case-insensitive occurrence of each cue phrase in `text`."""
    spans, low = [], text.lower()
    for cue in cues or ():
        cue = str(cue).strip().lower()
        if not cue:
            continue
        start = low.find(cue)
        while start >= 0:
            spans.append((start, start + len(cue)))
            start = low.find(cue, start + 1)
    return spans


def pair_spans(chosen, rejected, contradicted_cues=()):
    """(chosen spans, rejected spans) that get full weight."""
    sc, sr = diff_spans(chosen, rejected)
    return sc, sr + cue_spans(rejected, contradicted_cues)


def token_weights(tokenizer, text, spans, shared_weight, eos=True):
    """One weight per response token, in the order of the official DPO tokenisation
    (tokenizer(text, add_special_tokens=False) followed by one EOS token)."""
    if not 0 <= shared_weight <= 1:
        raise ValueError("shared_weight must be in [0, 1]")
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    plain = tokenizer(text, add_special_tokens=False)["input_ids"]
    if list(enc["input_ids"]) != list(plain):
        raise ValueError("Offset tokenisation differs from the training tokenisation")
    weights = []
    for start, end in enc["offset_mapping"]:
        hit = any(start < e and s < end for s, e in spans)
        weights.append(1.0 if hit else float(shared_weight))
    if eos:
        weights.append(float(shared_weight))
    return weights
