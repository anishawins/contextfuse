"""
Query-adaptive fusion weighting.

THE PROBLEM a fixed alpha cannot solve, observed on a real query:

    "vellore"                      -> rare proper noun, appears verbatim
                                      inside images. Lexical search is
                                      almost perfect. Wants alpha ~0.9.

    "logical reasoning question"   -> describes what a picture LOOKS like.
                                      The target image's OCR text is
                                      "Which option comes next in the
                                      sequence?" - not one query word in it.
                                      Lexical search actively misleads,
                                      because unrelated screenshots DO
                                      contain the word "question".
                                      Wants alpha ~0.2.

A single constant serves one and sabotages the other. So estimate, per query,
how much trustworthy lexical evidence actually exists, and set alpha from it.

THE SIGNAL: inverse document frequency, which BM25 already computes.

  - A query term in NO document carries zero lexical information.
  - A term in almost EVERY document ("question", "the") carries almost none,
    and worse, it retrieves confidently wrong results.
  - A term in ONE OR TWO documents ("vellore", "23bai0208", "psycopg2") is
    strong evidence and should dominate.

So: find the rarest query term that actually exists in the index, express its
IDF as a fraction of the maximum possible IDF for this corpus, and interpolate
alpha between a floor and a ceiling.

This is a heuristic, not a learned model. It is transparent, costs nothing at
query time, has no parameters to fit, and every step of it can be explained
to someone who asks why a particular query got the weight it did.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from contextfuse.lexical import BM25, tokenize

ALPHA_FLOOR = 0.15     # purely visual queries still get a little lexical nudge
ALPHA_CEIL = 0.90      # never fully ignore the visual signal

# A term appearing in more than this fraction of the corpus carries no
# discriminating power. Rather than ship a hard-coded English stopword list,
# derive it from the corpus: "a", "of" and "the" are uninformative here for
# the same measurable reason that "question" is uninformative in a folder of
# Q&A screenshots - they are everywhere. A fixed stopword list would catch
# the first three and miss the one that actually broke the ranking.
MAX_DF_RATIO = 0.5


@dataclass
class AlphaDecision:
    alpha: float
    specificity: float
    known_terms: list[str]          # in the index AND discriminating
    unknown_terms: list[str]        # in no image at all
    uninformative_terms: list[str]  # present, but in too many images to help
    rarest_term: str | None
    rarest_df: int | None
    explanation: str


def decide_alpha(query: str, bm25: BM25,
                 floor: float = ALPHA_FLOOR,
                 ceil: float = ALPHA_CEIL) -> AlphaDecision:
    toks = tokenize(query)
    unknown = [t for t in toks if t not in bm25.index]
    present = [t for t in toks if t in bm25.index]
    uninformative = [t for t in present
                     if len(bm25.index[t]) / bm25.N > MAX_DF_RATIO]
    known = [t for t in present if t not in uninformative]

    if not known:
        bits = []
        if unknown:
            bits.append(f"{len(unknown)} term(s) appear in no image")
        if uninformative:
            bits.append(f"{', '.join(repr(t) for t in uninformative[:3])} "
                        f"appear(s) in over half the corpus")
        return AlphaDecision(
            floor, 0.0, [], unknown, uninformative, None, None,
            "no discriminating query term available ("
            + "; ".join(bits) + ") - lexical search has nothing useful to "
            "contribute, so rely on vision")

    # Highest possible IDF in this corpus: a term appearing in exactly 1 doc.
    max_idf = math.log(1 + (bm25.N - 1 + 0.5) / (1 + 0.5))
    rarest = max(known, key=lambda t: bm25.idf[t])
    specificity = min(1.0, bm25.idf[rarest] / max_idf) if max_idf > 0 else 0.0

    # Coverage: of the words that COULD have carried information, how many
    # actually matched? Matching 1 of 5 meaningful terms is weaker evidence
    # than matching 4 of 5.
    #
    # The denominator is known + unknown, NOT len(toks). Uninformative filler
    # belongs in neither half: an earlier version divided by every token, so
    # "a vellore of the" scored coverage 0.25 and came out at alpha 0.647
    # while bare "vellore" got 0.9 - the same query, three filler words apart.
    # Padding a query with "the" must not change what the system believes.
    informative_total = len(known) + len(unknown)
    coverage = len(known) / max(1, informative_total)
    score = specificity * (0.55 + 0.45 * coverage)
    alpha = floor + (ceil - floor) * score

    df = len(bm25.index[rarest])
    if specificity > 0.75:
        why = (f"'{rarest}' appears in only {df} of {bm25.N} images - "
               f"a rare, specific term, so trust the text match")
    elif specificity > 0.4:
        why = (f"'{rarest}' appears in {df} of {bm25.N} images - "
               f"moderately specific, balance both signals")
    else:
        why = (f"the rarest query term '{rarest}' appears in {df} of "
               f"{bm25.N} images - too common to discriminate, "
               f"so lean on visual similarity")
    if unknown:
        why += f"; not in any image text: {unknown[:3]}"
    if uninformative:
        why += f"; too common to help: {uninformative[:3]}"

    return AlphaDecision(round(alpha, 3), round(specificity, 3),
                         known, unknown, uninformative, rarest, df, why)
