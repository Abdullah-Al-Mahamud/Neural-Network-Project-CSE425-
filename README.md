# Neural Network Project: Multi-Genre Music Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive deep learning project implementing multiple neural network architectures for unsupervised multi-genre music generation. This repository contains implementations of autoencoders, variational autoencoders, transformers, and reinforcement learning approaches for generating novel music compositions.

**Course:** CSE425 / EEE474 Neural Networks  
**Academic Year:** Spring 2026  
**Authors:** Abdullah Al Mahamud & Siddika Parvin Anni

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Results](#results)
- [License](#license)
- [References](#references)

---

## 🎯 Overview

This project explores unsupervised learning methods for music generation across five distinct genres: Classical, Jazz, Rock, Pop, and Electronic. Four different neural network architectures are implemented and evaluated based on quantitative metrics and qualitative analysis.

### Key Objectives
- Implement multiple deep learning architectures for music generation
- Develop robust evaluation metrics for generated music quality
- Compare different generative approaches across genres
- Create a reproducible pipeline for music preprocessing and generation

---

## ✨ Features

- **Multi-Genre Support:** Train and generate music in 5 distinct genres
- **Multiple Architectures:** 
  - Bidirectional LSTM Autoencoder
  - Variational Autoencoder (β-VAE)
  - Causal Transformer
  - Reinforcement Learning with Human Feedback (RLHF)
- **Comprehensive Preprocessing:** MIDI parsing, piano-roll representation, tokenization
- **Evaluation Metrics:** Pitch histogram, rhythm score, repetition analysis
- **End-to-End Pipeline:** Data loading, training, evaluation, and MIDI generation

---

## 🏗️ Architecture

### Task 1: Bidirectional LSTM Autoencoder
- Encoder: 2-layer Bidirectional LSTM
- Decoder: Sequential LSTM with attention
- Loss: Reconstruction MSE + KL divergence
- Output: Reconstructed and novel music samples

### Task 2: Variational Autoencoder (β-VAE)
- Encoder: Dense layers with Gaussian distribution
- Latent Space: Continuous representation
- Decoder: Dense to piano-roll reconstruction
- Loss: ELBO with β regularization
- Output: Smooth latent space interpolation

### Task 3: Causal Transformer
- Architecture: Multi-head attention with causal masking
- Positional Encoding: Sinusoidal embeddings
- Decoder-Only: Autoregressive generation
- Optimization: AdamW with learning rate warmup
- Output: Sequence-to-sequence music generation

### Task 4: Reinforcement Learning
- Reward Model: Critic network trained on human preference
- Policy: Generator network
- Algorithm: Policy gradient optimization
- Feedback: User-driven quality improvement
- Output: Human-aligned music generation

---

## 💾 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Abdullah-Al-Mahamud/Neural-Network-Project-CSE425-.git
   cd music-generation
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Quick Start
```bash
# Run all 4 tasks end-to-end
python main.py
```

### Individual Task Execution

```bash
# Task 1: Autoencoder Training and Generation
python src/training/train_ae.py

# Task 2: VAE Training and Generation
python src/training/train_vae.py

# Task 3: Transformer Training and Generation
python src/training/train_transformer.py

# Task 4: RLHF Training
python src/training/rlhf.py
```

### Data Preparation

1. Place MIDI files in `data/raw_midi/`
2. Run preprocessing:
   ```bash
   python scripts/prep_lakh_dataset.py
   ```
3. Processed data will be organized by genre in `data/processed/`

### Configuration

Edit `src/config.py` to adjust:
- Model hyperparameters
- Training parameters
- Data paths
- Generation settings

### Generate Music

```python
from src.generation.generate_music import generate
from src.generation.midi_export import save_midi

# Generate samples
samples = generate(model_type='vae', n_samples=10)
for i, sample in enumerate(samples):
    save_midi(sample, f'outputs/generated_midis/sample_{i}.mid')
```

---

## 📁 Project Structure

```
music-generation/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Project dependencies
├── .gitignore                            # Git ignore rules
├──
├── main.py                               # Master runner (all tasks)
├── src/
│   ├── config.py                         # Hyperparameters & configuration
│   ├── preprocessing/
│   │   ├── midi_parser.py                # MIDI → piano-roll conversion
│   │   ├── tokenizer.py                  # Music tokenization
│   │   └── piano_roll.py                 # Piano-roll utilities
│   ├── models/
│   │   ├── autoencoder.py                # LSTM Autoencoder
│   │   ├── vae.py                        # Variational Autoencoder
│   │   ├── transformer.py                # Causal Transformer
│   │   ├── diffusion.py                  # Diffusion-based model
│   │   └── rlhf.py                       # RLHF reward & policy models
│   └── training/
│       ├── train_ae.py                   # Autoencoder training
│       ├── train_vae.py                  # VAE training
│       └── train_transformer.py          # Transformer training
│
├── evaluation/
│   ├── metrics.py                        # Core evaluation metrics
│   ├── pitch_histogram.py                # Pitch distribution analysis
│   ├── rhythm_score.py                   # Rhythm quality metrics
│   └── baselines.py                      # Random & Markov baselines
│
├── generation/
│   ├── generate_music.py                 # Generation pipeline
│   ├── sample_latent.py                  # Latent space sampling
│   └── midi_export.py                    # Piano-roll → MIDI export
│
├── data/
│   ├── raw_midi/                         # Raw MIDI input files
│   ├── processed/                        # Preprocessed data by genre
│   │   ├── classical/
│   │   ├── jazz/
│   │   ├── rock/
│   │   ├── pop/
│   │   └── electronic/
│   └── train_test_split/                 # Train/validation/test splits
│
├── models/                               # Trained model checkpoints
│   ├── autoencoder.pt
│   ├── vae.pt
│   ├── transformer.pt
│   └── rlhf.pt
│
├── notebooks/
│   ├── preprocessing.ipynb               # Data exploration & preprocessing
│   └── baseline_markov.ipynb             # Baseline model comparison
│
├── scripts/
│   ├── prep_lakh_dataset.py              # Dataset preparation
│   └── generate_architecture_diagrams.py # Visualize model architecture
│
├── outputs/
│   ├── generated_midis/                  # Generated MIDI files
│   │   ├── task1/
│   │   ├── task2/
│   │   ├── task3/
│   │   └── task4/
│   ├── plots/                            # Training curves & analysis
│   └── survey_results/                   # User survey results
│
└── report/
    ├── final_report.tex                  # LaTeX project report
    ├── references.bib                    # Bibliography
    └── architecture_diagrams/            # Model diagrams
```

---

## 📋 Requirements

### Core Dependencies
```
torch>=2.0.0              # Deep learning framework
numpy>=1.24.0             # Numerical computing
pretty_midi>=0.2.9        # MIDI file handling
matplotlib>=3.7.0         # Visualization
scikit-learn>=1.2.0       # Machine learning utilities
tqdm>=4.65.0              # Progress bars
```

See `requirements.txt` for complete dependency list.

---

## 📊 Results

### Generated Samples
Generated music samples are available in `outputs/generated_midis/` organized by task:
- **Task 1:** Autoencoder reconstructions and variations
- **Task 2:** VAE interpolations and samples
- **Task 3:** Transformer sequential generations
- **Task 4:** RLHF-optimized compositions

### Evaluation Metrics
Quantitative results comparing all architectures:
- Pitch distribution similarity: [Results plotted in `outputs/plots/`]
- Rhythm consistency: [Detailed analysis available]
- Baseline comparison: [Statistical comparisons included]

---

## 📧 Support & Contribution

### Issues & Questions
For bug reports or questions, please open an issue on GitHub.

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.

2. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes. *arXiv preprint arXiv:1312.6114*.

3. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *arXiv preprint arXiv:1706.03762*.

4. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155*.

5. Paszke, A., Gross, S., Chanan, G., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *arXiv preprint arXiv:1912.01703*.

---

## � Authors

**Abdullah Al Mahamud**  
Department of Computer Science and Engineering  
Email: [email protected]  

**Siddika Parvin Anni**  
Department of Computer Science and Engineering  
Email: siddika.parvin.anni@g.bracu.ac.bd  

Spring 2026

---

## 🙏 Acknowledgments

- Course instructors for guidance and feedback
- Academic dataset providers (Lakh MIDI Dataset, MAESTRO)
- Open-source PyTorch community

---

*Last Updated: April 24, 2026*

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
