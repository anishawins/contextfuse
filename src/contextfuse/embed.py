"""SigLIP 2 encoder. Turns images and text into unit-length vectors."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

DEFAULT_CKPT = "google/siglip2-base-patch16-224"   # Apache-2.0


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SigLIPEncoder:
    def __init__(self, ckpt: str = DEFAULT_CKPT, device: str | None = None):
        self.ckpt = ckpt
        self.device = device or pick_device()
        self.model = AutoModel.from_pretrained(ckpt).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(ckpt)

    @property
    def dim(self) -> int:
        """Read the dim off the config, defensively - never hard-code 768."""
        for attr in ("text_config", "vision_config"):
            cfg = getattr(self.model.config, attr, None)
            if cfg is not None and hasattr(cfg, "hidden_size"):
                return int(cfg.hidden_size)
        return int(getattr(self.model.config, "hidden_size", -1))

    @staticmethod
    def _pool(out) -> torch.Tensor:
        """
        Pull the embedding tensor out of whatever the model handed back.

        transformers 5.x changed get_image_features / get_text_features to
        return the sub-model's full output OBJECT (BaseModelOutputWithPooling)
        instead of a bare tensor, as it did in 4.x. The embedding is the
        `pooler_output` field.

        Verified against the installed source: Siglip2Model.forward() does
            image_embeds = vision_outputs.pooler_output
            text_embeds  = text_outputs.pooler_output
        and then L2-normalises both before taking cosine similarity - which
        is exactly what _l2 below reproduces.

        Handles both shapes so the code survives a version change either way.
        """
        if isinstance(out, torch.Tensor):
            return out
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            raise TypeError(
                f"expected a tensor or an output with .pooler_output, "
                f"got {type(out).__name__} with fields {list(getattr(out, 'keys', lambda: [])())}")
        return pooled

    @staticmethod
    def _l2(x: torch.Tensor) -> torch.Tensor:
        """
        Force every vector to length 1.

        After this, cosine similarity between two vectors is just their dot
        product - which turns "score one query against 1360 pages" into a
        single matrix multiply. It also stops a large-magnitude vector from
        scoring highly against everything (the 'hub' problem), and it bounds
        every score into [-1, 1] so it can later be fused with BM25.
        """
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 16,
                      column: str = "image") -> np.ndarray:
        """
        `images` may be a list of PIL images OR a Hugging Face Dataset.

        For a Dataset we slice ONE BATCH AT A TIME. Writing ds["image"] would
        decode every page at once - 1360 pages at 1700x2200 RGB is roughly
        15 GB of RAM and an instant crash. Lazy decoding is not an
        optimisation here, it is the difference between running and not.
        """
        is_hf = hasattr(images, "column_names")
        n = len(images)
        out = []
        for i in tqdm(range(0, n, batch_size), desc="images", unit="batch"):
            chunk = (images[i:i + batch_size][column] if is_hf
                     else images[i:i + batch_size])
            batch = [im.convert("RGB") for im in chunk]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self._pool(self.model.get_image_features(**inputs))
            out.append(self._l2(feats).cpu().float().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="text", unit="batch"):
            inputs = self.processor(text=texts[i:i + batch_size],
                                    padding="max_length", truncation=True,
                                    max_length=64, return_tensors="pt").to(self.device)
            feats = self._pool(self.model.get_text_features(**inputs))
            out.append(self._l2(feats).cpu().float().numpy())
        return np.concatenate(out, axis=0)
