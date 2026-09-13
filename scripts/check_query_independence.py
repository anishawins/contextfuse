#!/usr/bin/env python3
"""
One question, and it decides whether your week-11 statistics are valid.

215 english + 215 french + 215 spanish + 215 italian + 215 german
+ 215 portuguese = 1290 exactly. That is suspicious.

If the six language groups are TRANSLATIONS of the same 215 information
needs, then the experimental unit is the information need (n=215), not
the query row (n=1290). Treating 1290 correlated rows as independent
samples produces confidence intervals that are far too narrow and
significance that is not real.

Reads the local cache. Downloads nothing.
"""
from collections import Counter, defaultdict

from datasets import load_dataset

REPO = "vidore/vidore_v3_computer_science"
q = load_dataset(REPO, "queries")["test"]
qrels = load_dataset(REPO, "qrels")["test"]


def rule(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# relevant-page set per query
rel = defaultdict(set)
for row in qrels:
    rel[row["query_id"]].add(row["corpus_id"])

lang = {row["query_id"]: row["language"] for row in q}
text = {row["query_id"]: row["query"] for row in q}
gen = {row["query_id"]: row["query_generator"] for row in q}

rule("A  Do queries share identical relevant-page sets across languages?")
groups = defaultdict(list)
for qid, pages in rel.items():
    groups[frozenset(pages)].append(qid)

sizes = Counter(len(v) for v in groups.values())
print(f"  distinct relevant-page sets : {len(groups)}")
print(f"  group size distribution     : {dict(sorted(sizes.items()))}")
print("     (a spike at 6 == each need appears once per language)")

multi = [v for v in groups.values() if len(v) > 1]
print(f"  groups with >1 query        : {len(multi)}")
if multi:
    langs_per_group = Counter(
        tuple(sorted({lang[qid] for qid in v})) for v in multi
    )
    print("\n  language make-up of shared groups (top 5):")
    for k, n in langs_per_group.most_common(5):
        print(f"      {len(k)} langs {k}  x{n}")

    print("\n  --- one example group, verbatim ---")
    ex = max(multi, key=len)
    for qid in sorted(ex, key=lambda i: lang[i]):
        print(f"   [{lang[qid]:<10} {gen[qid]:<6} id={qid:<5}] {text[qid][:88]}")

rule("B  VERDICT")
six = sizes.get(6, 0)
if six >= 150:
    n_eff = len(groups)
    print(f"  CONFIRMED: {six} groups of exactly 6 queries sharing a page set.")
    print(f"  The queries are translations of ~{n_eff} distinct information needs.")
    print()
    print(f"  EXPERIMENTAL UNIT  = information need,  n = {n_eff}")
    print(f"  NOT the query row,                      n = {len(q)}")
    print()
    print("  Consequences:")
    print("   1. Report English-only (n=215) for the CLIP vs SigLIP comparison,")
    print("      otherwise multilinguality confounds the model comparison.")
    print("   2. For significance tests, bootstrap over information needs,")
    print("      never over the 1290 rows.")
    print("   3. A separate multilingual run is a legitimate EXTRA experiment,")
    print("      not the headline one.")
else:
    print("  NOT a clean 6-way translation structure.")
    print("  Inspect the group-size distribution above before assuming anything.")

rule("C  Is the English-only subset usable on its own?")
eng = [qid for qid in lang if lang[qid] == "english"]
print(f"  english queries          : {len(eng)}")
print(f"  ... human-written        : {sum(1 for i in eng if gen[i] == 'human')}")
print(f"  ... synthetic            : {sum(1 for i in eng if gen[i] != 'human')}")
rels = [len(rel[i]) for i in eng]
print(f"  relevant pages per query : min={min(rels)} median={sorted(rels)[len(rels)//2]} max={max(rels)}")
pages = set().union(*(rel[i] for i in eng))
print(f"  pages reachable          : {len(pages)} of 1360  ({len(pages)/1360:.0%})")
print("\n  A 215-query eval set is small but publishable. It is roughly the")
print("  size of many TREC tracks. Report confidence intervals, not bare means.")
