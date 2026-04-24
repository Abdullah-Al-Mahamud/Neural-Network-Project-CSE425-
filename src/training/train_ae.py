# ============================================================
# train_ae.py — Task 1: Train LSTM Autoencoder
# ============================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.midi_parser import load_dataset


def train_autoencoder(device="cpu"):
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    print("[Task 1] Loading dataset...")
    X, y = load_dataset(Config.DATA_DIR)
    # Use only the first genre for Task 1 (single-genre)
    single_genre_mask = y == 0
    X_sg = X[single_genre_mask] if single_genre_mask.any() else X

    X_tensor = torch.tensor(X_sg, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(X_tensor), batch_size=Config.BATCH_SIZE,
                          shuffle=True, drop_last=True)

    # ── Model ─────────────────────────────────────────────────
    model = LSTMAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()  # MSE is appropriate for continuous targets in [0,1]

    # ── Training loop ─────────────────────────────────────────
    loss_curve = []
    print(f"[Task 1] Training for {Config.EPOCHS_AE} epochs...")

    for epoch in range(1, Config.EPOCHS_AE + 1):
        model.train()
        epoch_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_hat, _ = model(x_batch, teacher_forcing_ratio=max(0.0, 0.5 - epoch * 0.01))
            # BCEWithLogitsLoss expects logits and target in [0,1]
            loss = criterion(x_hat, x_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_curve.append(avg_loss)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{Config.EPOCHS_AE}  Loss={avg_loss:.4f}")

    # ── Save model ────────────────────────────────────────────
    torch.save(model.state_dict(), f"{Config.MODEL_DIR}/autoencoder.pt")

    # ── Plot reconstruction loss curve ────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(loss_curve, label="Reconstruction Loss (MSE)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Task 1 — LSTM Autoencoder Reconstruction Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{Config.PLOTS_DIR}/task1_loss_curve.png", dpi=150)
    plt.close()
    print(f"[Task 1] Loss curve saved -> {Config.PLOTS_DIR}/task1_loss_curve.png")

    # ── Generate 5 samples ────────────────────────────────────
    samples = model.generate(n_samples=5, device=device)   # (5, T, 128)
    print(f"[Task 1] Generated {samples.shape[0]} piano-roll samples.")

    return model, loss_curve, samples


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_autoencoder(device)
