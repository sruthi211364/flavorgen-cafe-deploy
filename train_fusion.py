from __future__ import annotations

import os
import json
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flavorgen.data_loader import load_drinks, load_ingredients
from flavorgen.fusion_model import (
    TextEmbedder,
    build_category_maps,
    build_ingredient_vocab,
    build_training_matrices,
    FusionNet,
    save_artifacts,
)

OUT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models", "fusion")


def _ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def train_fusion(
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    backend: str = "auto",
) -> Dict[str, float]:
    _ensure_out_dir()

    drinks_df = load_drinks()
    ingredients_df = load_ingredients()

    cat_maps = build_category_maps(drinks_df)
    id_to_index, index_to_name = build_ingredient_vocab(ingredients_df)

    embedder = TextEmbedder(backend=backend)

    X, Y = build_training_matrices(
        drinks_df=drinks_df,
        ingredients_df=ingredients_df,
        embedder=embedder,
        cat_maps=cat_maps,
        id_to_index=id_to_index,
    )

    in_dim = int(X.shape[1])
    out_dim = int(Y.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    n = len(X_t)
    idx = np.arange(n)
    np.random.shuffle(idx)

    split = int(0.85 * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_ds = TensorDataset(X_t[tr_idx], Y_t[tr_idx])
    va_ds = TensorDataset(X_t[va_idx], Y_t[va_idx])

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = FusionNet(in_dim=in_dim, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr_losses = []

        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_losses.append(float(loss.item()))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0

        print(f"Epoch {ep}/{epochs} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "backend": backend,
        "created_at": str(pd.Timestamp.utcnow()),
        "n_rows": int(len(drinks_df)),
        "n_ingredients": int(len(index_to_name)),
        "cat_maps": cat_maps,
        "ingredient_index_to_name": index_to_name,
    }

    save_artifacts(model=model, embedder=embedder, out_dir=OUT_DIR, meta=meta)

    return {"best_val_loss": float(best_val), "in_dim": float(in_dim), "out_dim": float(out_dim)}


def main() -> None:
    metrics = train_fusion(
        epochs=12,
        batch_size=64,
        lr=1e-3,
        backend="tfidf",  # ← self-contained, no download
    )
    print("Saved fusion artifacts to:", OUT_DIR)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
