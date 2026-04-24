# ============================================================
# transformer.py — Task 3: Autoregressive Transformer Decoder
# ============================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model=Config.D_MODEL, max_len=Config.MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MusicTransformer(nn.Module):
    """
    Causal (decoder-only) Transformer for autoregressive music generation.

    Autoregressive objective:
        p(X) = ∏_t p(x_t | x_{<t})

    Training loss:
        L_TR = -Σ_t log p_θ(x_t | x_{<t})

    Perplexity:
        PP = exp(1/T · L_TR)

    Genre conditioning: prepend a genre token so the model learns
        p(x_t | x_{<t}, genre)   (from the project brief, page 5)
    """

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL,
                                       padding_idx=Config.PAD_TOKEN)
        self.pos_enc   = PositionalEncoding()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model    = Config.D_MODEL,
            nhead      = Config.NHEAD,
            dim_feedforward = Config.DIM_FFD,
            dropout    = Config.DROPOUT,
            batch_first = True,
            norm_first  = True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer,
                                                  num_layers=Config.NUM_LAYERS)
        self.fc_out = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

        # Memory placeholder (zero — we use decoder-only self-attention)
        self._dummy_mem = None

    def _causal_mask(self, sz, device):
        """Upper-triangular mask to enforce causality."""
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, tokens, genre_idx=None):
        """
        tokens    : (B, T)  long  — token ids
        genre_idx : (B,)    long  — optional genre token index (0-4)
        Returns logits : (B, T, VOCAB_SIZE)
        """
        B, T = tokens.shape
        device = tokens.device

        # Optionally prepend genre conditioning token
        if genre_idx is not None:
            genre_tokens = (Config.GENRE_TOKEN_BASE + genre_idx).unsqueeze(1)  # (B,1)
            tokens = torch.cat([genre_tokens, tokens], dim=1)                  # (B,T+1)
            T = tokens.size(1)

        x = self.token_emb(tokens)   # (B, T, D_MODEL)
        x = self.pos_enc(x)

        # Create dummy memory for decoder API (self-attention only)
        # Size it based on the current batch size B
        mem = torch.zeros(B, 1, Config.D_MODEL, device=device)

        mask = self._causal_mask(T, device)
        x = self.transformer(x, mem, tgt_mask=mask,
                              tgt_is_causal=True)
        logits = self.fc_out(x)   # (B, T, VOCAB_SIZE)

        # Drop the genre-prefix position from output if we added it
        if genre_idx is not None:
            logits = logits[:, 1:, :]

        return logits

    def loss(self, logits, targets):
        """
        Cross-entropy loss: L_TR = -Σ log p_θ(x_t | x_{<t})
        logits  : (B, T, VOCAB_SIZE)
        targets : (B, T)
        """
        B, T, V = logits.shape
        return F.cross_entropy(logits.reshape(B * T, V),
                               targets.reshape(B * T),
                               ignore_index=Config.PAD_TOKEN)

    @staticmethod
    def perplexity(loss_value):
        """PP = exp(L_TR)"""
        return math.exp(loss_value)

    @torch.no_grad()
    def generate(self, n_samples=10, max_len=256, temperature=1.0,
                 genre_idx=None, device="cpu"):
        """
        Autoregressive sampling: x_t ~ p_θ(x_t | x_{<t})

        temperature > 1  → more random / creative
        temperature < 1  → more conservative
        """
        self.eval()
        bos = torch.full((n_samples, 1), Config.BOS_TOKEN,
                          dtype=torch.long, device=device)
        seqs = bos

        for _ in range(max_len - 1):
            logits = self.forward(seqs, genre_idx=genre_idx)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seqs = torch.cat([seqs, next_token], dim=1)
            if (next_token == Config.EOS_TOKEN).all():
                break

        return seqs   # (n_samples, T)  — token sequences
