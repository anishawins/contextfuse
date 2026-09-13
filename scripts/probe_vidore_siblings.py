#!/usr/bin/env python3
"""
Do the other ViDoRe V3 domains share the CS schema?

If yes, adding them costs almost nothing and buys a heterogeneous,
multi-domain public corpus with cross-domain distractors.
If no, we stay with CS alone and say why.

Reads METADATA ONLY - no parquet files are downloaded.
"""
from datasets import get_dataset_config_names, load_dataset_builder

CS = "vidore/vidore_v3_computer_science"
CANDIDATES = ["vidore/vidore_v3_energy", "vidore/vidore_v3_hr"]

REF_COLS = {
    "corpus":  {"corpus_id", "image", "doc_id", "markdown", "page_number_in_doc"},
    "queries": {"query_id", "query", "language", "query_types", "content_type"},
    "qrels":   {"query_id", "corpus_id", "score", "content_type", "bounding_boxes"},
}


def probe(repo):
    print(f"\n{'='*70}\n{repo}\n{'='*70}")
    try:
        cfgs = get_dataset_config_names(repo)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  UNREACHABLE: {type(exc).__name__}: {str(exc)[:150]}")
        return
    print(f"  configs: {cfgs}")

    for cfg in cfgs:
        try:
            b = load_dataset_builder(repo, cfg)
        except Exception as exc:                              # noqa: BLE001
            print(f"    [{cfg}] builder failed: {str(exc)[:90]}")
            continue
        info = b.info
        cols = set(info.features) if info.features else set()
        rows = sum(s.num_examples for s in info.splits.values()) if info.splits else "?"
        mb = (sum(s.num_bytes for s in info.splits.values()) / 1_048_576
              if info.splits else 0)
        line = f"    {cfg:<20} rows={str(rows):<7} ~{mb:7.1f} MB"
        ref = REF_COLS.get(cfg)
        if ref:
            missing = ref - cols
            extra = cols - ref
            line += "  schema=MATCH" if not missing else f"  schema=DIFFERS missing={sorted(missing)}"
            if extra:
                line += f" extra={sorted(extra)}"
        print(line)
        if info.license:
            print(f"      license: {info.license}")


probe(CS)
for r in CANDIDATES:
    probe(r)

print("\n" + "=" * 70)
print("If a sibling MATCHES on corpus/queries/qrels, adding it is a")
print("config change, not new code. Report the MB figures - that is the")
print("download you would be committing to.")
