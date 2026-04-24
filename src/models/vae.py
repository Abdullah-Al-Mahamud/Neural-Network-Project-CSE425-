# ============================================================
# vae.py — Task 2: β-VAE Multi-Genre Music Generator
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from src.models.autoencoder import LSTMEncoder, LSTMDecoder


class VAEEncoder(nn.Module):
    """Encoder that outputs (μ, log σ²) instead of a fixed z."""

    def __init__(self):
        super().__init__()
        # Reuse the bidirectional LSTM backbone
        self.lstm = LSTMEncoder().lstm
        self.fc_mu     = nn.Linear(Config.LSTM_HIDDEN * 2, Config.LATENT_DIM)
        self.fc_logvar = nn.Linear(Config.LSTM_HIDDEN * 2, Config.LATENT_DIM)

    def forward(self, x):
        """
        x : (B, T, 128)
        Returns (mu, logvar) each (B, LATENT_DIM)
        """
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        mu     = self.fc_mu(h_cat)
        logvar = self.fc_logvar(h_cat)
        return mu, logvar

    @staticmethod
    def reparameterise(mu, logvar):
        """
        Reparameterisation trick:  z = μ + σ ⊙ ε,  ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


class MusicVAE(nn.Module):
    """
    β-VAE for multi-genre music generation.

    Loss: L_VAE = L_recon + β · D_KL(q_φ(z|X) ‖ p(z))

    D_KL for diagonal Gaussian:
        = -½ Σ (1 + log σ² - μ² - σ²)
    """

    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = LSTMDecoder()
        self.beta    = Config.BETA

    def forward(self, x, teacher_forcing_ratio=0.5):
        mu, logvar = self.encoder(x)
        z = VAEEncoder.reparameterise(mu, logvar)
        x_hat = self.decoder(z, target_seq=x,
                             teacher_forcing_ratio=teacher_forcing_ratio)
        return x_hat, mu, logvar

    def loss(self, x, x_hat, mu, logvar):
        """Returns total loss, recon loss, and KL divergence (all scalars)."""
        # Reconstruction: MSE for continuous targets in [0,1]
        # x_hat and x are both probabilities in [0,1]
        recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
        # KL divergence (closed form)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total = recon + self.beta * kl
        return total, recon, kl

    @torch.no_grad()
    def generate(self, n_samples=8, device="cpu"):
        """Sample z ~ N(0,I) and output probability estimates."""
        self.eval()
        z = torch.randn(n_samples, Config.LATENT_DIM, device=device)
        probabilities = self.decoder(z, target_seq=None, teacher_forcing_ratio=0.0)
        return probabilities

    @torch.no_grad()
    def interpolate(self, x1, x2, steps=8, device="cpu"):
        """
        Latent interpolation experiment (Task 2 deliverable).
        Linearly interpolate between two encoded sequences.
        """
        self.eval()
        mu1, _ = self.encoder(x1.unsqueeze(0).to(device))
        mu2, _ = self.encoder(x2.unsqueeze(0).to(device))
        alphas = torch.linspace(0, 1, steps, device=device)
        samples = []
        for a in alphas:
            z_interp = (1 - a) * mu1 + a * mu2
            probabilities = self.decoder(z_interp, target_seq=None, teacher_forcing_ratio=0.0)
            samples.append(probabilities)
        return torch.cat(samples, dim=0)
