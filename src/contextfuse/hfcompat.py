"""
Loading Hugging Face datasets that still ship a loading script.

`datasets` 5.x removed script-based datasets entirely:

    RuntimeError: Dataset scripts are no longer supported, but found x.py

Several well-known datasets (nlphuji/flickr30k and its 1k retrieval test set
among them) have not been migrated. Their pages still show a data viewer,
because the Hub AUTO-CONVERTS every dataset to parquet and publishes it on a
side branch called `refs/convert/parquet`. The viewer reads that branch; a
plain load_dataset() call reads `main`, finds the script, and refuses.

So: three routes, in order of preference, reporting which one worked.

  1. load_dataset(repo)                    - native parquet, nothing to do
  2. load_dataset(repo, revision=...)      - the auto-converted branch
  3. download the parquet files by hand    - when the branch layout confuses
                                             the config resolver

Route 3 exists because the convert branch names its configs after the
original ones, and for datasets with unusual config layouts route 2 can fail
on config resolution even though the files are sitting right there.
"""
from __future__ import annotations

from datasets import concatenate_datasets, load_dataset

CONVERT_REV = "refs/convert/parquet"


def load_any(repo: str, verbose: bool = True) -> tuple:
    """Returns (dataset, how_it_loaded)."""
    errors: list[str] = []

    # --- 1. the normal way ------------------------------------------------
    try:
        d = load_dataset(repo)
        split = "test" if "test" in d else next(iter(d))
        if verbose:
            print(f"  loaded natively (split={split!r})")
        return d[split], "native"
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"native: {type(exc).__name__}: {str(exc)[:110]}")

    # --- 2. the auto-converted parquet branch -----------------------------
    try:
        d = load_dataset(repo, revision=CONVERT_REV)
        split = "test" if "test" in d else next(iter(d))
        if verbose:
            print(f"  loaded from {CONVERT_REV} (split={split!r})")
        return d[split], "convert-branch"
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"convert branch: {type(exc).__name__}: {str(exc)[:110]}")

    # --- 3. fetch the parquet files directly ------------------------------
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [f for f in list_repo_files(repo, repo_type="dataset",
                                            revision=CONVERT_REV)
                 if f.endswith(".parquet")]
        if not files:
            raise FileNotFoundError(f"no parquet files on {CONVERT_REV}")

        # Prefer test-split shards when the branch carries several splits.
        test_files = [f for f in files if "test" in f.lower()]
        chosen = test_files or files
        if verbose:
            print(f"  downloading {len(chosen)} parquet file(s) from {CONVERT_REV}")
            for f in chosen[:4]:
                print(f"      {f}")
            if len(chosen) > 4:
                print(f"      ... and {len(chosen)-4} more")

        local = [hf_hub_download(repo, f, repo_type="dataset",
                                 revision=CONVERT_REV) for f in chosen]
        parts = [load_dataset("parquet", data_files=p)["train"] for p in local]
        ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        if verbose:
            print(f"  loaded {len(ds)} rows from raw parquet")
        return ds, "raw-parquet"
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"raw parquet: {type(exc).__name__}: {str(exc)[:110]}")

    raise RuntimeError(
        f"could not load {repo}. Tried:\n  " + "\n  ".join(errors))
