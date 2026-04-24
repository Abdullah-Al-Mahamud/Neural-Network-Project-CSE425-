# ============================================================
# train_vae.py - Task 2: Train Beta-VAE Multi-Genre Generator
# ============================================================
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.models.vae import MusicVAE
from src.preprocessing.midi_parser import load_dataset


def train_vae(device="cpu"):
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print("[Task 2] Loading multi-genre dataset...")
    X, y = load_dataset(Config.DATA_DIR)
    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=Config.BATCH_SIZE,
                        shuffle=True, drop_last=True)

    model = MusicVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    total_losses, recon_losses, kl_losses = [], [], []
    print(f"[Task 2] Training VAE for {Config.EPOCHS_VAE} epochs  (Beta={Config.BETA})...")

    for epoch in range(1, Config.EPOCHS_VAE + 1):
        model.train()
        ep_total = ep_recon = ep_kl = 0.0
        tf_ratio = max(0.0, 0.5 - epoch * 0.01)  # anneal teacher forcing

        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_hat, mu, logvar = model(x_batch, teacher_forcing_ratio=tf_ratio)
            total, recon, kl = model.loss(x_batch, x_hat, mu, logvar)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_total += total.item()
            ep_recon += recon.item()
            ep_kl    += kl.item()

        n = len(loader)
        total_losses.append(ep_total / n)
        recon_losses.append(ep_recon / n)
        kl_losses.append(ep_kl / n)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{Config.EPOCHS_VAE}  "
                  f"Total={total_losses[-1]:.4f}  "
                  f"Recon={recon_losses[-1]:.4f}  "
                  f"KL={kl_losses[-1]:.4f}")

    torch.save(model.state_dict(), f"{Config.MODEL_DIR}/vae.pt")

    # -- Loss curves -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, name in zip(axes,
                               [total_losses, recon_losses, kl_losses],
                               ["Total (ELBO)", "Reconstruction", "KL Divergence"]):
        ax.plot(vals)
        ax.set_title(f"VAE {name} Loss")
        ax.set_xlabel("Epoch"); ax.grid(True)
    plt.suptitle("Task 2 - Beta-VAE Training Losses")
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/task2_vae_losses.png", dpi=150)
    plt.close()

    # -- Generate 8 samples ------------------------------------
    samples = model.generate(n_samples=8, device=device)
    print(f"[Task 2] Generated {samples.shape[0]} multi-genre samples.")

    return model, {"total": total_losses, "recon": recon_losses, "kl": kl_losses}, samples


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_vae(device)
