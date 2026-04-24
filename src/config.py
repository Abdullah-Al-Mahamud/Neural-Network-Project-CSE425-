# ============================================================
# config.py — Central hyperparameters for all tasks
# CSE425/EEE474 Neural Networks — Music Generation Project
# ============================================================

class Config:
    # ── Device ────────────────────────────────────────────────
    DEVICE           = "cpu"  # Use CPU for stability
    STEPS_PER_BAR    = 16          # 1/16-note resolution
    SEQ_LEN          = 64          # 4 bars of 16 steps
    PITCH_RANGE      = 128         # MIDI pitch 0–127
    GENRES           = ["classical", "jazz", "rock", "pop", "electronic"]

    # ── Training ──────────────────────────────────────────────
    BATCH_SIZE       = 16
    LEARNING_RATE    = 1e-3
    EPOCHS_AE        = 10
    EPOCHS_VAE       = 10
    EPOCHS_TR        = 10
    RL_STEPS         = 50
    MAX_FILES_PER_GENRE = 100  # Limit dataset for fast training

    # ── Task 1 – LSTM Autoencoder ─────────────────────────────
    LATENT_DIM       = 128         # z dimension
    LSTM_HIDDEN      = 256
    LSTM_LAYERS      = 2

    # ── Task 2 – VAE ─────────────────────────────────────────
    BETA             = 0.5         # KL weight; tune to prevent collapse

    # ── Task 3 – Transformer ─────────────────────────────────
    D_MODEL          = 256
    NHEAD            = 8
    NUM_LAYERS       = 4
    DIM_FFD          = 512
    DROPOUT          = 0.1
    # Need to account for: 128 pitches + 5 special tokens (PAD, BOS, EOS + 5 genres)
    VOCAB_SIZE       = 140         # Safe margin: 128 + 12 special slots
    MAX_SEQ_LEN      = 512

    # ── Special tokens (Task 3) ───────────────────────────────
    PAD_TOKEN        = 128
    BOS_TOKEN        = 129
    EOS_TOKEN        = 130
    GENRE_TOKEN_BASE = 131         # 131–135 → 5 genres

    # ── Paths ─────────────────────────────────────────────────
    DATA_DIR         = "data/processed/"
    OUTPUT_DIR       = "outputs/generated_midis/"
    PLOTS_DIR        = "outputs/plots/"
    MODEL_DIR        = "models/"
