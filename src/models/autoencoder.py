# ============================================================
# autoencoder.py — Task 1: Bidirectional LSTM Autoencoder
# ============================================================
import torch
import torch.nn as nn
from src.config import Config


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder → fixed latent vector z."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = Config.PITCH_RANGE,
            hidden_size = Config.LSTM_HIDDEN,
            num_layers  = Config.LSTM_LAYERS,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.2,
        )
        self.fc = nn.Linear(Config.LSTM_HIDDEN * 2, Config.LATENT_DIM)

    def forward(self, x):
        """
        x : (B, T, 128)
        Returns z : (B, LATENT_DIM)
        """
        _, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        z = self.fc(h_cat)
        return z


class LSTMDecoder(nn.Module):
    """Unidirectional LSTM decoder with teacher forcing support."""

    def __init__(self):
        super().__init__()
        self.fc_init = nn.Linear(Config.LATENT_DIM, Config.LSTM_HIDDEN)
        self.start_token = nn.Parameter(torch.randn(1, 1, Config.PITCH_RANGE) * 0.1)
        self.lstm = nn.LSTM(
            input_size  = Config.PITCH_RANGE,
            hidden_size = Config.LSTM_HIDDEN,
            num_layers  = Config.LSTM_LAYERS,
            batch_first = True,
            dropout     = 0.2,
        )
        # Output layer WITH Sigmoid activation for probability outputs
        self.output_layer = nn.Sequential(
            nn.Linear(Config.LSTM_HIDDEN, Config.PITCH_RANGE),
            nn.Sigmoid()  # Map to [0,1] probability range
        )
        
        # Apply He initialization for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training dynamics."""
        # Initialize output layer weights
        for module in self.output_layer:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize LSTM weights (PyTorch uses default orthogonal, but we ensure good init)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data, nonlinearity='sigmoid')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, z, target_seq=None, teacher_forcing_ratio=0.5):
        """
        z          : (B, LATENT_DIM)
        target_seq : (B, T, 128) -- if provided, uses teacher forcing
        Returns x_hat : (B, T, 128) probabilities in [0,1]
        """
        B = z.size(0)
        T = Config.SEQ_LEN
        device = z.device

        h0 = self.fc_init(z).unsqueeze(0).repeat(Config.LSTM_LAYERS, 1, 1)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        outputs = []
        x_t = self.start_token.expand(B, 1, Config.PITCH_RANGE).to(device)

        for t in range(T):
            out, hidden = self.lstm(x_t, hidden)
            x_hat_t = self.output_layer(out)  # Already includes Sigmoid
            outputs.append(x_hat_t)

            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                x_t = target_seq[:, t:t+1, :]
            else:
                # x_hat_t is already probability, use for sampling
                x_t = torch.bernoulli(x_hat_t)

        return torch.cat(outputs, dim=1)   # (B, T, 128) probabilities [0,1]


class LSTMAutoencoder(nn.Module):
    """Full autoencoder: encoder + decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x, teacher_forcing_ratio=0.5):
        """
        Loss: L_AE = Σ ||x_t - x̂_t||²
        """
        z    = self.encoder(x)
        x_hat = self.decoder(z, target_seq=x,
                             teacher_forcing_ratio=teacher_forcing_ratio)
        return x_hat, z

    @torch.no_grad()
    def generate(self, n_samples=5, device="cpu"):
        """Sample from random z and output probability estimates."""
        self.eval()
        z = torch.randn(n_samples, Config.LATENT_DIM, device=device)
        probabilities = self.decoder(z, target_seq=None, teacher_forcing_ratio=0.0)
        return probabilities
