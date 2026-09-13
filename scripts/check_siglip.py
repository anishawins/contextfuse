#!/usr/bin/env python3
"""
Answer one question by measurement, not by guessing:
what is SigLIP 2's embedding dimensionality, and does it run on your GPU?

    source .venv/bin/activate
    python scripts/check_siglip.py

Downloads ~1.5 GB of model weights the first time.
"""
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

CKPT = "google/siglip2-base-patch16-224"   # Apache-2.0, verified 2026-09-10

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

model = AutoModel.from_pretrained(CKPT).eval().to(device)
proc = AutoProcessor.from_pretrained(CKPT)

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params/1e6:.1f} M")

dummy = Image.new("RGB", (224, 224), (128, 128, 128))
texts = ["a screenshot of a terminal error", "a photo of a dog"]

with torch.no_grad():
    img_in = proc(images=[dummy], return_tensors="pt").to(device)
    t0 = time.perf_counter()
    img_emb = model.get_image_features(**img_in)
    t_img = time.perf_counter() - t0

    txt_in = proc(text=texts, padding="max_length", return_tensors="pt").to(device)
    t0 = time.perf_counter()
    txt_emb = model.get_text_features(**txt_in)
    t_txt = time.perf_counter() - t0

print(f"\nimage embedding shape : {tuple(img_emb.shape)}")
print(f"text  embedding shape : {tuple(txt_emb.shape)}")
print(f"  -> EMBEDDING DIM = {img_emb.shape[-1]}")
print(f"  -> image and text dims match: {img_emb.shape[-1] == txt_emb.shape[-1]}")

print(f"\nraw L2 norms (BEFORE normalising):")
print(f"  image: {img_emb.norm(dim=-1).tolist()}")
print(f"  text : {txt_emb.norm(dim=-1).tolist()}")
print("  ^ if these are far from 1.0, the model does NOT normalise for you.")

i = torch.nn.functional.normalize(img_emb, dim=-1)
t = torch.nn.functional.normalize(txt_emb, dim=-1)
print(f"\ncosine sim to each caption: {(i @ t.T).squeeze(0).tolist()}")
print(f"\nsingle-image encode: {t_img*1000:.1f} ms   |   2-text encode: {t_txt*1000:.1f} ms")
print("(first call includes warm-up; not a benchmark)")
