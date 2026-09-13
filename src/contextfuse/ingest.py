"""
Turning a folder of images into indexable records.

Four jobs, in order:
  1. FIND    - walk the folder for real image files
  2. IDENTIFY- exact hash (byte-identical duplicates) and perceptual hash
               (visually near-identical: same screen, different compression)
  3. DESCRIBE- timestamp, dimensions, and a guess at the source application
  4. FILTER  - skip anything that looks private

Perceptual hashing matters more than it sounds. Screenshot folders are full
of near-duplicates - you screenshot the same error three times while
debugging. If all three rank in your top 10, you have ONE useful result and
two wasted slots. Detecting them lets the ranker collapse them.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# iPhone screenshots and photos are often HEIC, which Pillow cannot open on
# its own. pillow-heif registers a decoder for it. Optional: without it, HEIC
# files are reported as unreadable rather than silently vanishing.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:                                           # pragma: no cover
    HEIC_SUPPORTED = False

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".heic"}

# Filenames that suggest something private. Conservative by design: it is far
# better to skip a harmless image than to index a bank statement.
PRIVATE_HINTS = re.compile(
    r"(bank|statement|salary|payslip|aadhaar|aadhar|passport|pan[\s_-]?card|"
    r"invoice|tax|medical|prescription|password|private|personal|whatsapp|"
    r"chat|dm[\s_-]|selfie|id[\s_-]?proof)", re.IGNORECASE)

SOURCE_PATTERNS = [
    (re.compile(r"^screenshot|^screen shot|^cleanshot|^shottr", re.I), "screenshot"),
    (re.compile(r"^simulator|^iphone|^android", re.I), "device_capture"),
    (re.compile(r"^img_|^dsc|^photo", re.I), "camera"),
    (re.compile(r"slide|lecture|lec\d", re.I), "slides"),
    (re.compile(r"diagram|architecture|flow", re.I), "diagram"),
]


@dataclass
class ImageRecord:
    id: int
    path: str
    filename: str
    sha256: str
    phash: str
    width: int
    height: int
    bytes: int
    mtime_iso: str
    source: str
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    ocr_engine: str = ""
    duplicate_of: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def dhash(img: Image.Image, size: int = 8) -> str:
    """
    Difference hash: shrink to 9x8 greyscale, then record whether each pixel
    is brighter than the one to its right. 64 bits.

    Robust to resizing, compression and small brightness shifts - exactly the
    ways two copies of the same screenshot differ - while still separating
    genuinely different images. Compare two hashes by Hamming distance.
    """
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.tobytes())          # mode "L": one byte per pixel
    bits = "".join(
        "1" if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else "0"
        for r in range(size) for c in range(size))
    return f"{int(bits, 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def guess_source(name: str) -> str:
    for pat, label in SOURCE_PATTERNS:
        if pat.search(name):
            return label
    return "unknown"


def looks_private(path: Path, root: Path | None = None) -> bool:
    """
    Heuristic guess at whether a file is something you would not want indexed.

    ONLY the filename and any SUBFOLDER names BELOW the scan root are checked.

    The root folder itself is deliberately excluded. An earlier version
    checked path.parent.name unconditionally, so scanning a folder called
    "personal_ss" matched the word "personal" and silently skipped every
    single file in it. The user chose the root folder on purpose - treating
    their own choice as a red flag is wrong.
    """
    if PRIVATE_HINTS.search(path.name):
        return True
    if root is None:
        return False
    try:
        rel_parents = path.relative_to(root).parent.parts
    except ValueError:
        return False
    return any(PRIVATE_HINTS.search(part) for part in rel_parents)


def scan(folder: Path, skip_private: bool = True,
         phash_threshold: int = 6) -> tuple[list[ImageRecord], list[str]]:
    """Returns (records, skipped_reasons)."""
    records: list[ImageRecord] = []
    skipped: list[str] = []
    by_sha: dict[str, int] = {}
    seen_phash: list[tuple[str, int]] = []

    files = sorted(p for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTS)

    if not HEIC_SUPPORTED and any(f.suffix.lower() == ".heic" for f in files):
        n = sum(1 for f in files if f.suffix.lower() == ".heic")
        skipped.append(
            f"{n} HEIC files cannot be decoded - run: pip install pillow-heif")

    for path in files:
        if skip_private and looks_private(path, folder):
            skipped.append(f"private-looking name: {path.name}")
            continue
        try:
            with Image.open(path) as im:
                im.load()
                w, h = im.size
                ph = dhash(im)
        except Exception as exc:                              # noqa: BLE001
            skipped.append(f"unreadable ({type(exc).__name__}): {path.name}")
            continue

        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        rec = ImageRecord(
            id=len(records), path=str(path), filename=path.name,
            sha256=sha, phash=ph, width=w, height=h, bytes=len(raw),
            mtime_iso=datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            source=guess_source(path.name),
        )
        if sha in by_sha:
            rec.duplicate_of = by_sha[sha]
        else:
            by_sha[sha] = rec.id
            for other_ph, other_id in seen_phash:
                if hamming(ph, other_ph) <= phash_threshold:
                    rec.duplicate_of = other_id
                    break
            seen_phash.append((ph, rec.id))
        records.append(rec)

    return records, skipped
