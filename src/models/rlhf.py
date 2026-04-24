# ============================================================
# rlhf.py — Task 4: Reward Model + Policy Gradient RLHF
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


# ──────────────────────────────────────────────────────────────
# Reward Model — predicts Human Satisfaction Score ∈ [1, 5]
# ──────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Small network trained on survey data to predict a scalar reward.

    Input  : piano-roll  (B, T, 128)
    Output : reward score (B,)  ∈ [1, 5]
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size  = Config.PITCH_RANGE,
            hidden_size = 128,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        x : (B, T, 128)  — piano-roll
        Returns scores (B,) ∈ [1, 5]
        """
        _, (h_n, _) = self.encoder(x)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 256)
        raw = self.head(h_cat).squeeze(-1)
        # Clamp to [1, 5] using sigmoid rescaling
        return 1.0 + 4.0 * torch.sigmoid(raw)

    def train_on_survey(self, survey_rolls, survey_scores,
                         lr=1e-3, epochs=20, device="cpu"):
        """
        Fine-tune the reward model on human survey data.

        survey_rolls  : list of piano-roll tensors (T, 128)
        survey_scores : list of float scores in [1, 5]
        """
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        X = torch.stack(survey_rolls).to(device)          # (N, T, 128)
        y = torch.tensor(survey_scores, dtype=torch.float32, device=device)

        self.train()
        for ep in range(epochs):
            pred = self(X)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (ep + 1) % 5 == 0:
                print(f"  [RewardModel] epoch {ep+1}/{epochs}  loss={loss.item():.4f}")


# ──────────────────────────────────────────────────────────────
# RLHF Trainer — Policy Gradient update on the Transformer
# ──────────────────────────────────────────────────────────────

class RLHFTrainer:
    """
    Implements the RLHF objective from the project brief:

        max_θ  E[r(X_gen)]

    Policy gradient update rule:
        ∇_θ J(θ) = E[ r · ∇_θ log p_θ(X) ]

    We use a simple REINFORCE loop with a baseline (mean reward)
    to reduce variance.
    """

    def __init__(self, generator, reward_model,
                 lr=1e-4, device="cpu"):
        """
        generator    : MusicTransformer (Task 3)
        reward_model : RewardModel (trained above)
        """
        self.gen  = generator.to(device)
        self.rm   = reward_model.to(device)
        self.opt  = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.device = device

    def _piano_roll_from_tokens(self, token_seqs):
        """Convert token ids → binary piano-roll (B, T, 128) for the RM."""
        B, T = token_seqs.shape
        roll = torch.zeros(B, T, Config.PITCH_RANGE, device=self.device)
        mask = (token_seqs < Config.PITCH_RANGE)
        pitch_ids = token_seqs.clamp(0, Config.PITCH_RANGE - 1)
        for b in range(B):
            valid = mask[b]
            roll[b, valid.nonzero(as_tuple=True)[0],
                 pitch_ids[b, valid.nonzero(as_tuple=True)[0]]] = 1.0
        return roll

    def rl_step(self, batch_size=16, genre_idx=None):
        """
        One RLHF policy gradient step.

        Returns
        -------
        reward_mean : float  — mean reward for logging
        loss        : float  — policy loss
        """
        self.gen.train()
        self.rm.eval()

        # ── 1. Sample from current policy ────────────────────
        with torch.no_grad():
            seqs = self.gen.generate(
                n_samples=batch_size,
                genre_idx=genre_idx,
                device=self.device
            )                                   # (B, T)

        # ── 2. Score with reward model ────────────────────────
        rolls = self._piano_roll_from_tokens(seqs)
        with torch.no_grad():
            rewards = self.rm(rolls)            # (B,)

        # Baseline: subtract mean reward to reduce variance
        baseline = rewards.mean()
        advantages = rewards - baseline         # (B,)

        # ── 3. Compute log p_θ(X) for sampled sequences ──────
        #    We feed x_{<T} as input and x_{1:T} as target.
        inp = seqs[:, :-1]    # (B, T-1)
        tgt = seqs[:, 1:]     # (B, T-1)
        logits = self.gen(inp, genre_idx=genre_idx)         # (B, T-1, V)
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log prob of the actual token at each step
        tgt_clamped = tgt.clamp(0, Config.VOCAB_SIZE - 1)
        token_log_probs = log_probs.gather(
            2, tgt_clamped.unsqueeze(-1)).squeeze(-1)       # (B, T-1)

        # Sum over time → sequence-level log prob
        seq_log_probs = token_log_probs.sum(dim=-1)         # (B,)

        # ── 4. Policy gradient loss = -E[A · log p_θ(X)] ─────
        policy_loss = -(advantages.detach() * seq_log_probs).mean()

        self.opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
        self.opt.step()

        return rewards.mean().item(), policy_loss.item()

    def train(self, rl_steps=Config.RL_STEPS, genre_idx=None):
        """Run the full RLHF fine-tuning loop."""
        print("\n[Task 4] RLHF Fine-tuning started...")
        reward_history = []
        for step in range(1, rl_steps + 1):
            r_mean, loss = self.rl_step(genre_idx=genre_idx)
            reward_history.append(r_mean)
            if step % 10 == 0:
                print(f"  step {step:4d}/{rl_steps}  "
                      f"mean_reward={r_mean:.3f}  loss={loss:.4f}")
        print("[Task 4] RLHF Fine-tuning complete.")
        return reward_history
