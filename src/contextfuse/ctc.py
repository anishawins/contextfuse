"""
CTC: Connectionist Temporal Classification. Alphabet, decoding, error rates.

THE PROBLEM CTC SOLVES:
the network emits 24 frame predictions. The target is a word of, say, 5
characters. Nobody has told you WHICH frames produced which character - that
alignment is unknown, and labelling it by hand for 9 million images is not a
plan. CTC trains without alignments by summing the probability of EVERY
alignment that collapses to the right word.

THE COLLAPSING RULE, which is the whole idea:
add one extra symbol, the BLANK (written '-'). To read a frame sequence:

    1. collapse runs of repeated symbols
    2. then delete the blanks

    c-aa-t     ->  c-a-t   ->  cat
    ccaaat     ->  cat     ->  cat
    cc-aa-tt   ->  c-a-t   ->  cat

The blank is what makes double letters possible. Without it, "hello" could
never be emitted, because the two l's would collapse into one. With it, the
network emits  h-e-l-l-o  with a blank BETWEEN the l's, and the collapse rule
keeps them apart. That is the entire reason the blank exists, and it is the
follow-up question interviewers ask after "what is CTC".

THE LOSS:
    P(word | image) = SUM over all alignments that collapse to `word`
                      of  PROD over frames of  P(symbol_t | frame_t)

computed efficiently by dynamic programming (the forward-backward algorithm)
in O(T * L) rather than enumerating exponentially many alignments.
PyTorch's nn.CTCLoss implements it; this module supplies the alphabet and
the decoding side, which nn.CTCLoss does not.

CONSTRAINT WORTH KNOWING: the network cannot emit a word longer than it has
frames, and needs a spare frame between repeated characters. With T = 26 that
is not binding for MJSynth, whose longest labels are ~23 characters.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

# index 0 is reserved for the CTC blank - nn.CTCLoss defaults to blank=0
DIGITS = "0123456789"
LOWER = "abcdefghijklmnopqrstuvwxyz"


class Alphabet:
    """Maps characters to class indices. Index 0 is always the blank."""

    def __init__(self, chars: str = DIGITS + LOWER, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        self.chars = chars
        self.itos = ["<blank>"] + list(chars)
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.itos)

    def normalise(self, text: str) -> str:
        t = text if self.case_sensitive else text.lower()
        return "".join(c for c in t if c in self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in self.normalise(text)]

    def decode_indices(self, idx) -> str:
        return "".join(self.itos[i] for i in idx if i != 0)


# ----------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------
def greedy_decode(logits: torch.Tensor, alphabet: Alphabet) -> list[str]:
    """
    Best path decoding: take argmax at each frame, collapse repeats, drop blanks.

    logits: [T, B, C]

    Fast and usually close to optimal, but it is NOT the most probable WORD -
    it is the most probable ALIGNMENT. Several different alignments can collapse
    to the same word, and their probabilities add up; a word can therefore be
    more likely overall than the single best path suggests. Beam search below
    accounts for that.
    """
    best = logits.argmax(dim=2).cpu().numpy()         # [T, B]
    out = []
    for b in range(best.shape[1]):
        prev, chars = -1, []
        for t in range(best.shape[0]):
            k = int(best[t, b])
            if k != prev and k != 0:                   # collapse, then drop blanks
                chars.append(alphabet.itos[k])
            prev = k
        out.append("".join(chars))
    return out


def beam_decode(logits: torch.Tensor, alphabet: Alphabet,
                beam_width: int = 10) -> list[str]:
    """
    Prefix beam search: sums the probability of all alignments per prefix.

    Each beam entry tracks TWO probabilities for a prefix - one where the last
    emitted frame was a blank (p_blank) and one where it was a real character
    (p_nonblank). That split is required to apply the collapse rule correctly:
    emitting 'l' again extends "hel" to "hell" only if the previous frame was a
    blank; otherwise it merges into the existing 'l'.
    """
    logp = torch.log_softmax(logits, dim=2).cpu().numpy()   # [T, B, C]
    T, B, C = logp.shape
    results = []

    for b in range(B):
        # prefix -> (log p ending in blank, log p ending in a character)
        beams = {(): (0.0, -np.inf)}
        for t in range(T):
            nxt: dict[tuple, list[float]] = defaultdict(lambda: [-np.inf, -np.inf])
            top = np.argsort(-logp[t, b])[:beam_width]
            for prefix, (pb, pnb) in beams.items():
                total = np.logaddexp(pb, pnb)
                for c in top:
                    p = float(logp[t, b, c])
                    if c == 0:                                   # blank
                        e = nxt[prefix]
                        e[0] = np.logaddexp(e[0], total + p)
                    elif prefix and c == prefix[-1]:
                        # same char as last: merges unless a blank intervened
                        same = nxt[prefix]
                        same[1] = np.logaddexp(same[1], pnb + p)  # merge
                        ext = nxt[prefix + (int(c),)]
                        ext[1] = np.logaddexp(ext[1], pb + p)     # blank between
                    else:
                        ext = nxt[prefix + (int(c),)]
                        ext[1] = np.logaddexp(ext[1], total + p)
            beams = dict(sorted(nxt.items(),
                                key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1])
                                )[:beam_width])
            beams = {k: (v[0], v[1]) for k, v in beams.items()}
        best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]
        results.append("".join(alphabet.itos[i] for i in best))
    return results


# ----------------------------------------------------------------------
# Error rates
# ----------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance - insertions, deletions and substitutions."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (ca != cb)))   # substitution
        prev = cur
    return prev[-1]


def cer(preds: list[str], targets: list[str]) -> float:
    """
    Character Error Rate = total edit distance / total target characters.

    Aggregated over the corpus, NOT averaged per word. Per-word averaging lets
    one badly-read three-letter word count as much as a correctly-read
    twenty-letter one, which is the wrong weighting and inflates the number.
    """
    dist = sum(edit_distance(p, t) for p, t in zip(preds, targets))
    total = sum(len(t) for t in targets)
    return dist / total if total else float("nan")


def wer(preds: list[str], targets: list[str]) -> float:
    """
    Word Error Rate on cropped-word data = fraction of words not read exactly.

    On this task a "word" is the whole image, so WER is 1 - exact-match accuracy.
    It is unforgiving by design: one wrong character fails the whole word.
    """
    if not targets:
        return float("nan")
    return sum(p != t for p, t in zip(preds, targets)) / len(targets)
