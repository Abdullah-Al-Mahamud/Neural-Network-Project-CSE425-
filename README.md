# Unsupervised Neural Network for Multi-Genre Music Generation
**CSE425 / EEE474 Neural Networks** | Submission: April 10, 2026

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py          # runs all 4 tasks end-to-end
```

## Project Structure

```
music-generation-unsupervised/
├── main.py                        # Master runner (Tasks 1–4 + evaluation)
├── requirements.txt
├── src/
│   ├── config.py                  # All hyperparameters
│   ├── preprocessing/
│   │   └── midi_parser.py         # MIDI → piano-roll loader
│   ├── models/
│   │   ├── autoencoder.py         # Task 1: Bidirectional LSTM AE
│   │   ├── vae.py                 # Task 2: β-VAE
│   │   ├── transformer.py         # Task 3: Causal Transformer
│   │   └── rlhf.py                # Task 4: Reward Model + Policy Gradient
│   └── training/
│       ├── train_ae.py
│       ├── train_vae.py
│       └── train_transformer.py
├── evaluation/
│   ├── metrics.py                 # All 4 metrics (pitch hist, rhythm, repetition, human)
│   └── baselines.py               # Random + Markov baselines
├── generation/
│   └── midi_export.py             # Piano-roll / token → MIDI file
├── data/
│   ├── raw_midi/                  # Put Lakh MIDI / MAESTRO files here
│   └── processed/
│       ├── classical/
│       ├── jazz/
│       ├── rock/
│       ├── pop/
│       └── electronic/
├── outputs/
│   ├── generated_midis/           # All generated .mid files
│   └── plots/                     # Loss curves, perplexity plots
└── models/                        # Saved model checkpoints (.pt)
```

## Dataset Setup

1. Download the [Lakh MIDI Clean Subset](https://colinraffel.com/projects/lmd/)
2. Sort files into `data/processed/<genre>/` folders
3. Run `python main.py`

> **No MIDI files?**  The pipeline auto-generates synthetic data so you can test the full training loop immediately.

## Tasks Summary

| Task | Model | Key Math | Deliverables |
|------|-------|----------|--------------|
| 1 (Easy) | LSTM Autoencoder | L_AE = Σ‖x_t - x̂_t‖² | 5 MIDI samples + loss curve |
| 2 (Medium) | β-VAE | L_VAE = L_recon + β·KL | 8 samples + latent interpolation |
| 3 (Hard) | Transformer | L_TR = −Σ log p(x_t|x_{<t}) | 10 long compositions + perplexity |
| 4 (Advanced) | RLHF | max E[r(X_gen)] | Survey + reward model + 10 samples |

## Hyperparameter Tuning

All hyperparameters are in `src/config.py`. Key ones to adjust:
- `BETA` — KL weight in VAE (0.1–1.0); increase if posterior collapse
- `EPOCHS_*` — reduce if training too slow
- `LATENT_DIM` — 128 (fast) or 256 (better quality)
