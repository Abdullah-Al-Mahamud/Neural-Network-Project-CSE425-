# ============================================================
# train_transformer.py — Task 3: Autoregressive Transformer
# ============================================================
import os
import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.models.transformer import MusicTransformer
from src.preprocessing.midi_parser import load_dataset


def piano_roll_to_tokens(roll: torch.Tensor) -> torch.Tensor:
    """
    Convert piano-roll (T, 128) -> token sequence (T,).
    Strategy: argmax active pitch per step; inactive step -> PAD token.
    """
    active = roll.sum(dim=-1) > 0    # (T,)
    tokens = roll.argmax(dim=-1)     # (T,)  pitch 0–127
    tokens[~active] = Config.PAD_TOKEN
    return tokens.long()


def train_transformer(device="cpu"):
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print("[Task 3] Loading dataset and tokenising...")
    X, y = load_dataset(Config.DATA_DIR)
    X_t = torch.tensor(X, dtype=torch.float32)
    # Convert to token sequences
    tokens = torch.stack([piano_roll_to_tokens(X_t[i]) for i in range(len(X_t))])
    y_t = torch.tensor(y, dtype=torch.long)

    loader = DataLoader(TensorDataset(tokens, y_t),
                        batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)

    model = MusicTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE,
                           betas=(0.9, 0.98), eps=1e-9)
    # Warmup scheduler (standard for Transformers)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(loader), epochs=Config.EPOCHS_TR
    )

    loss_curve, ppl_curve = [], []
    print(f"[Task 3] Training Transformer for {Config.EPOCHS_TR} epochs...")

    for epoch in range(1, Config.EPOCHS_TR + 1):
        model.train()
        ep_loss = 0.0
        for (toks, genres) in loader:
            toks, genres = toks.to(device), genres.to(device)
            inp = toks[:, :-1]   # x_{<t}
            tgt = toks[:, 1:]    # x_t

            logits = model(inp, genre_idx=genres)
            loss   = model.loss(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / len(loader)
        ppl = MusicTransformer.perplexity(avg_loss)
        loss_curve.append(avg_loss)
        ppl_curve.append(ppl)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{Config.EPOCHS_TR}  "
                  f"Loss={avg_loss:.4f}  Perplexity={ppl:.2f}")

    torch.save(model.state_dict(), f"{Config.MODEL_DIR}/transformer.pt")

    # ── Plots ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_curve); ax1.set_title("Cross-Entropy Loss"); ax1.grid(True)
    ax2.plot(ppl_curve);  ax2.set_title("Perplexity"); ax2.grid(True)
    plt.suptitle("Task 3 — Transformer Training")
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/task3_transformer.png", dpi=150)
    plt.close()

    # ── Generate 10 long compositions ─────────────────────────
    samples = model.generate(n_samples=10, max_len=256, temperature=0.9, device=device)
    print(f"[Task 3] Generated {samples.shape[0]} long-sequence compositions.")
    final_ppl = ppl_curve[-1]
    print(f"[Task 3] Final Perplexity: {final_ppl:.2f}")

    return model, loss_curve, ppl_curve, samples


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transformer(device)
