"""
Datasets for word recognition: MJSynth (training) and IIIT 5K-Word (evaluation).

THE SPLIT DISCIPLINE, and it is stronger than the usual setup:
the CRNN trains on MJSynth - synthetic renderings - and is scored on IIIT-5K -
real photographs. Those are different datasets from different sources, so
there is no path by which a test image could appear in training. That is a
harder evaluation than an in-distribution test split, and it is the standard
protocol, so the resulting CER/WER sit alongside published numbers.

It also means the reported error rate includes a DOMAIN SHIFT: the model is
being asked to read photographs having only ever seen synthetic text. Expect
that to cost accuracy, and say so rather than pretending it is a pure
measure of the architecture.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from contextfuse.ctc import Alphabet

IMG_H, IMG_W = 32, 100


def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Grayscale, resize to 32x100, scale to [-1, 1].

    The aspect ratio is NOT preserved. Every image is squashed to a fixed
    100px width, so a three-letter word and a twelve-letter word both occupy
    26 output frames. This is what the original CRNN work does and it is fine
    here because CTC learns per-frame alignment - it does not assume a
    character occupies a fixed number of frames. Preserving aspect ratio would
    need variable-width batching, which buys little on cropped words.
    """
    g = img.convert("L").resize((IMG_W, IMG_H), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float32) / 255.0
    return torch.from_numpy((a - 0.5) / 0.5).unsqueeze(0)      # [1, 32, 100]


class MJSynthSubset(Dataset):
    """Reads the manifest written by scripts/prepare_mjsynth.py."""

    def __init__(self, root: Path, manifest: Path, split: str,
                 alphabet: Alphabet, max_label: int = 24):
        recs = json.loads(manifest.read_text())["records"]
        self.root = root
        self.alphabet = alphabet
        self.items = []
        skipped = 0
        for r in recs:
            if split != "all" and r.get("split") != split:
                continue
            lab = alphabet.normalise(r["label"])
            # CTC cannot emit a label longer than the frame count, and needs a
            # spare frame between repeated characters. Drop the few that fail.
            if not lab or len(lab) > max_label:
                skipped += 1
                continue
            self.items.append((r["file"], lab))
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rel, label = self.items[i]
        img = Image.open(self.root / rel)
        return preprocess(img), label


class IIIT5K(Dataset):
    """
    IIIT 5K-Word test split.

    Ground truth ships as MATLAB .mat; scipy reads it. Falls back to parsing
    filenames only if the .mat is unreadable, and says which path it took.
    """

    LABEL_RE = re.compile(r"^(?:\d+_)?([A-Za-z0-9]+)")

    def __init__(self, root: Path, alphabet: Alphabet, split: str = "test"):
        self.root, self.alphabet = root, alphabet
        self.items: list[tuple[str, str]] = []
        self.source = "unknown"

        mat = root / f"{split}data.mat"
        if mat.is_file():
            try:
                from scipy.io import loadmat
                d = loadmat(str(mat), squeeze_me=True, struct_as_record=False)
                key = next(k for k in d if not k.startswith("__"))
                for rec in d[key]:
                    name = str(rec.ImgName)
                    gt = alphabet.normalise(str(rec.GroundTruth))
                    if gt:
                        self.items.append((name, gt))
                self.source = f"{mat.name} (scipy)"
            except Exception as exc:                      # noqa: BLE001
                print(f"  [warn] could not read {mat.name}: "
                      f"{type(exc).__name__}: {str(exc)[:80]}")

        if not self.items:
            for p in sorted((root / split).glob("*.png")):
                m = self.LABEL_RE.match(p.stem)
                if m:
                    gt = alphabet.normalise(m.group(1))
                    if gt:
                        self.items.append((f"{split}/{p.name}", gt))
            self.source = "filenames (FALLBACK - verify these labels)"

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        name, label = self.items[i]
        p = self.root / name
        if not p.is_file():
            p = self.root / "test" / Path(name).name
        return preprocess(Image.open(p)), label


def collate(batch, alphabet: Alphabet = None):
    """
    Batch for nn.CTCLoss.

    CTCLoss wants targets CONCATENATED into one flat 1-D tensor plus a length
    per item - not padded into a rectangle. Padding would force a pad symbol
    into the alphabet and CTC would then have to learn to emit it, which is
    exactly the bookkeeping the blank already handles.

    NOTE: this is a module-level function taking the alphabet as a keyword, so
    it can be wrapped with functools.partial and PICKLED. DataLoader workers
    are separate processes, and on Python 3.14 the default start method
    serialises their arguments - a lambda defined inside main() cannot cross
    that boundary and raises PicklingError.
    """
    if alphabet is None:
        raise ValueError("collate needs an alphabet; "
                         "use functools.partial(collate, alphabet=...)")
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    flat, lengths = [], []
    for t in labels:
        enc = alphabet.encode(t)
        flat.extend(enc)
        lengths.append(len(enc))
    return (imgs,
            torch.tensor(flat, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            labels)
