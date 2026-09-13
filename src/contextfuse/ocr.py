"""
OCR for real screenshots.

Screenshots are not scene text. They are crisp, axis-aligned, anti-aliased
UI text - often monospaced, often dark mode. What you need is a full-page
engine that DETECTS where text is and then RECOGNISES it. Two candidates:

  APPLE VISION (via the `ocrmac` package) - built into macOS, no extra
    binaries, genuinely excellent on UI text, runs offline, uses the Neural
    Engine. Default here because you are on a Mac and it is the strongest
    option available to you at zero install cost beyond one pip package.

  TESSERACT (via `pytesseract`) - the classic open-source engine, needs
    `brew install tesseract`. Cross-platform, so it is the portable
    fallback and the baseline anyone can reproduce.

Both give you text plus per-line confidence. The confidence matters: a line
OCR'd at 0.35 should not be trusted as an exact match, and later the ranker
uses it to discount shaky lexical hits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRResult:
    text: str
    mean_confidence: float
    n_lines: int
    engine: str
    lines: list[dict] = field(default_factory=list)


class OCREngine:
    name = "none"

    def read(self, path: Path) -> OCRResult:                 # pragma: no cover
        raise NotImplementedError


class AppleVisionOCR(OCREngine):
    name = "apple_vision"

    def __init__(self) -> None:
        from ocrmac import ocrmac                            # noqa: F401
        self._ocrmac = ocrmac

    def read(self, path: Path) -> OCRResult:
        ann = self._ocrmac.OCR(str(path), language_preference=["en-US"]).recognize()
        lines = [{"text": t, "confidence": float(c), "bbox": list(b)}
                 for t, c, b in ann]
        text = "\n".join(l["text"] for l in lines)
        conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0.0
        return OCRResult(text, conf, len(lines), self.name, lines)


class TesseractOCR(OCREngine):
    name = "tesseract"

    def __init__(self) -> None:
        import pytesseract
        from PIL import Image                                # noqa: F401
        self._pt = pytesseract
        self._pt.get_tesseract_version()                     # fails fast if missing

    def read(self, path: Path) -> OCRResult:
        from PIL import Image
        data = self._pt.image_to_data(Image.open(path),
                                      output_type=self._pt.Output.DICT)
        lines, confs = [], []
        for txt, c in zip(data["text"], data["conf"]):
            if not txt.strip():
                continue
            c = float(c)
            if c < 0:
                continue
            lines.append({"text": txt, "confidence": c / 100.0, "bbox": None})
            confs.append(c / 100.0)
        text = " ".join(l["text"] for l in lines)
        return OCRResult(text, sum(confs) / len(confs) if confs else 0.0,
                         len(lines), self.name, lines)


def get_engine(prefer: str = "auto") -> OCREngine:
    """Pick the best engine available, loudly, rather than silently degrading."""
    attempts = []
    order = (["apple", "tesseract"] if prefer in ("auto", "apple")
             else ["tesseract", "apple"])
    for want in order:
        try:
            return AppleVisionOCR() if want == "apple" else TesseractOCR()
        except Exception as exc:                              # noqa: BLE001
            attempts.append(f"{want}: {type(exc).__name__}: {str(exc)[:80]}")
    raise RuntimeError(
        "No OCR engine available.\n  " + "\n  ".join(attempts) +
        "\n\nInstall one:\n"
        "  pip install ocrmac        # Apple Vision, recommended on macOS\n"
        "  brew install tesseract && pip install pytesseract")
