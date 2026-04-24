# ============================================================
# baselines.py — Baseline models for comparison (Table 3)
# ============================================================
import numpy as np
from src.config import Config


class RandomNoteGenerator:
    """
    Naive baseline: sample each pitch independently at each step.
    """
    name = "Random Note Generator"

    def __init__(self, note_density: float = 0.05):
        """note_density = probability any pitch is active per step."""
        self.density = note_density

    def generate(self, n_samples: int = 5,
                 seq_len: int = Config.SEQ_LEN) -> np.ndarray:
        return (np.random.rand(n_samples, seq_len, 128) < self.density).astype(np.float32)


class MarkovChainModel:
    """
    First-order Markov chain over MIDI pitches.
    Transition probability: P(x_t | x_{t-1}) estimated from training data.
    """
    name = "Markov Chain Model"

    def __init__(self):
        # Transition matrix: (128, 128)  — P[from_pitch, to_pitch]
        self.transition = np.ones((128, 128)) / 128   # uniform prior
        self.start_dist = np.ones(128) / 128

    def fit(self, rolls: np.ndarray) -> None:
        """
        rolls : (N, T, 128) binary piano-rolls
        Build transition counts from each sample's dominant pitch per step.
        """
        counts = np.zeros((128, 128))
        for roll in rolls:
            # Get dominant active pitch per step (argmax)
            active = roll.sum(axis=-1) > 0   # (T,)
            pitches = roll.argmax(axis=-1)    # (T,)
            pitches[~active] = -1            # mark rests

            prev = None
            for pitch in pitches:
                if pitch >= 0:
                    if prev is not None and prev >= 0:
                        counts[prev, pitch] += 1
                    prev = pitch
                else:
                    prev = None

        # Normalise rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition = counts / row_sums
        print("[Markov] Transition matrix fitted.")

    def generate(self, n_samples: int = 5,
                 seq_len: int = Config.SEQ_LEN) -> np.ndarray:
        """Sample sequences and convert back to piano-rolls."""
        rolls = np.zeros((n_samples, seq_len, 128), dtype=np.float32)
        for i in range(n_samples):
            pitch = np.random.choice(128, p=self.start_dist)
            for t in range(seq_len):
                rolls[i, t, pitch] = 1.0
                # Get transition probs and ensure they sum to 1
                probs = self.transition[pitch].astype(np.float64)
                probs = np.maximum(probs, 0)  # Clamp negatives
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    probs = np.ones(128) / 128  # Fallback to uniform if all zeros
                pitch = np.random.choice(128, p=probs)
        return rolls
