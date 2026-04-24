# ============================================================
# metrics.py — All project evaluation metrics
# CSE425/EEE474 Neural Networks
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config


# ── 1. Pitch Histogram Similarity ─────────────────────────────
def pitch_histogram(roll: np.ndarray) -> np.ndarray:
    """
    Compute a 12-bin chromatic pitch histogram (pitch class distribution).
    roll : (T, 128)
    """
    hist = np.zeros(12)
    for pitch in range(128):
        hist[pitch % 12] += roll[:, pitch].sum()
    total = hist.sum()
    return hist / total if total > 0 else hist


def pitch_histogram_similarity(gen_roll: np.ndarray,
                                ref_roll: np.ndarray) -> float:
    """
    H(p, q) = Σ_{i=1}^{12} |p_i - q_i|   (project brief, page 6)
    Lower is better (0 = identical distributions).
    """
    p = pitch_histogram(gen_roll)
    q = pitch_histogram(ref_roll)
    return float(np.sum(np.abs(p - q)))


# ── 2. Rhythm Diversity Score ─────────────────────────────────
def rhythm_diversity(roll: np.ndarray) -> float:
    """
    D_rhythm = #unique durations / #total notes   (page 6)

    A "note" here = a run of consecutive active steps for a given pitch.
    We count the run length as the duration.
    """
    durations = []
    for pitch in range(128):
        col = roll[:, pitch]
        i = 0
        while i < len(col):
            if col[i] > 0:
                dur = 0
                while i < len(col) and col[i] > 0:
                    dur += 1; i += 1
                durations.append(dur)
            else:
                i += 1
    if not durations:
        return 0.0
    unique = len(set(durations))
    return unique / len(durations)


# ── 3. Repetition Ratio ───────────────────────────────────────
def repetition_ratio(roll: np.ndarray, pattern_len: int = 4) -> float:
    """
    R = #repeated patterns / #total patterns   (page 6)

    Patterns are non-overlapping windows of `pattern_len` steps.
    """
    T = roll.shape[0]
    patterns = []
    for t in range(0, T - pattern_len + 1, pattern_len):
        pat = roll[t:t + pattern_len].tobytes()
        patterns.append(pat)
    if not patterns:
        return 0.0
    from collections import Counter
    counts = Counter(patterns)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / len(patterns)


# ── 4. Human Listening Score (placeholder) ───────────────────
def load_survey_scores(csv_path: str) -> dict:
    """
    Load survey results.  Expected CSV format:
        sample_id, participant_id, coherence, creativity, genre_auth
    Returns dict {sample_id: mean_score}.

    If file doesn't exist, returns dummy scores for testing.
    """
    import os
    if not os.path.exists(csv_path):
        print(f"[WARN] Survey file not found: {csv_path}  — using dummy scores.")
        return {f"sample_{i}": round(np.random.uniform(2.5, 4.5), 2) for i in range(10)}

    import csv
    scores = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sample_id"]
            avg = np.mean([float(row["coherence"]),
                           float(row["creativity"]),
                           float(row["genre_auth"])])
            scores.setdefault(sid, []).append(avg)
    return {k: float(np.mean(v)) for k, v in scores.items()}


# ── Comparison table (replicates Table 3 from project brief) ──
def build_comparison_table(results: dict) -> None:
    """
    results : dict of {model_name: {loss, perplexity, rhythm_div, human_score}}
    Prints and saves a comparison table.
    """
    header = f"{'Model':<28} {'Loss':>8} {'PPL':>8} {'Rhythm Div':>12} {'Human Score':>12}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("PERFORMANCE COMPARISON TABLE")
    print(sep)
    print(header)
    print(sep)
    for name, m in results.items():
        loss = f"{m.get('loss', '–'):>8}" if isinstance(m.get('loss'), float) else f"{'–':>8}"
        ppl  = f"{m.get('perplexity', '–'):>8}" if isinstance(m.get('perplexity'), float) else f"{'–':>8}"
        rd   = f"{m.get('rhythm_div', '–'):>12.3f}" if isinstance(m.get('rhythm_div'), float) else f"{'–':>12}"
        hs   = f"{m.get('human_score', '–'):>12.1f}" if isinstance(m.get('human_score'), float) else f"{'–':>12}"
        print(f"{name:<28} {loss} {ppl} {rd} {hs}")
    print(sep + "\n")


# ── Batch evaluation helper ───────────────────────────────────
def evaluate_samples(samples: np.ndarray,
                     reference: np.ndarray = None,
                     label: str = "Model") -> dict:
    """
    Compute all automated metrics for a batch of piano-rolls.

    samples   : (N, T, 128)
    reference : (M, T, 128) — ground-truth for histogram similarity

    Returns dict of metric values.
    """
    N = samples.shape[0]
    rds  = [rhythm_diversity(samples[i])    for i in range(N)]
    reps = [repetition_ratio(samples[i])    for i in range(N)]

    hist_sims = []
    if reference is not None:
        ref_mean = reference.mean(axis=0)
        hist_sims = [pitch_histogram_similarity(samples[i], ref_mean)
                     for i in range(N)]

    metrics = {
        "rhythm_div":   float(np.mean(rds)),
        "repetition":   float(np.mean(reps)),
        "hist_sim":     float(np.mean(hist_sims)) if hist_sims else None,
    }
    print(f"[Metrics — {label}]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
    return metrics


# ── Plotting helper ───────────────────────────────────────────
def plot_piano_roll(roll: np.ndarray, title: str = "Generated Piano Roll",
                    save_path: str = None) -> None:
    plt.figure(figsize=(14, 4))
    plt.imshow(roll.T, aspect="auto", origin="lower",
               cmap="Blues", interpolation="nearest")
    plt.xlabel("Time Step"); plt.ylabel("MIDI Pitch")
    plt.title(title); plt.colorbar(label="Active")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
