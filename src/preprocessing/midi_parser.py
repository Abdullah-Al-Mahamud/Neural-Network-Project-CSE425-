# ============================================================
# midi_parser.py — Load MIDI, build piano-roll tensors
# ============================================================
import os
import numpy as np
import pretty_midi
from scipy.ndimage import gaussian_filter1d
from src.config import Config


def midi_to_piano_roll(midi_path: str,
                        steps_per_bar: int = Config.STEPS_PER_BAR,
                        seq_len: int = Config.SEQ_LEN) -> np.ndarray:
    """
    Convert a MIDI file to a binary piano-roll matrix.

    Returns
    -------
    np.ndarray  shape (T, 128)  — rows=time steps, cols=MIDI pitches
                T is clipped/padded to seq_len.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"[WARN] Could not load {midi_path}: {e}")
        return None

    # Use pretty_midi's built-in piano-roll at fs = steps per second.
    # Get tempo: use get_end_time() to estimate duration and default BPM
    try:
        tempo = pm.get_end_time()
        median_bpm = 120.0  # Default tempo if not available
    except:
        median_bpm = 120.0
    
    beats_per_second = median_bpm / 60.0
    fs = steps_per_bar * beats_per_second / 4   # steps per second (default: ~5 Hz)

    roll = pm.get_piano_roll(fs=fs).T           # (T, 128)
    roll = (roll > 0).astype(np.float32)        # binarise

    # Smooth with gaussian to encourage longer durations
    roll = gaussian_filter1d(roll, sigma=0.5, axis=0)
    roll = np.clip(roll, 0, 1)

    # Pad or crop to seq_len
    T = roll.shape[0]
    if T >= seq_len:
        # Take most active portion
        activity = roll.sum(axis=1)
        if activity.max() > 0:
            start = max(0, np.argmax(activity) - seq_len // 2)
        else:
            start = 0
        roll = roll[start:start+seq_len]
    
    if roll.shape[0] < seq_len:
        pad = np.zeros((seq_len - roll.shape[0], 128), dtype=np.float32)
        roll = np.vstack([roll, pad])

    return roll[:seq_len]


def load_dataset(data_dir: str,
                 seq_len: int = Config.SEQ_LEN) -> tuple:
    """
    Walk `data_dir`, parse all MIDI files, and return arrays + genre labels.
    Limits files per genre to MAX_FILES_PER_GENRE for faster training.

    Expected directory layout
    -------------------------
    data_dir/
      classical/  *.mid
      jazz/       *.mid
      ...

    Returns
    -------
    X : np.ndarray  (N, seq_len, 128)
    y : np.ndarray  (N,)  integer genre index
    """
    X, y = [], []
    genre_map = {g: i for i, g in enumerate(Config.GENRES)}
    max_files = getattr(Config, 'MAX_FILES_PER_GENRE', 100)

    print(f"[INFO] Loading dataset (max {max_files} files per genre)...")
    for genre in Config.GENRES:
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            print(f"[INFO] Genre folder missing: {genre_dir}  — skipping.")
            continue
        
        fnames = [f for f in os.listdir(genre_dir) if f.endswith((".mid", ".midi"))]
        fnames = fnames[:max_files]  # Limit to max_files per genre
        
        print(f"\n[INFO] Loading {len(fnames)} files from {genre}...", end=" ", flush=True)
        count = 0
        for fname in fnames:
            roll = midi_to_piano_roll(os.path.join(genre_dir, fname), seq_len=seq_len)
            if roll is not None:
                X.append(roll)
                y.append(genre_map[genre])
                count += 1
        print(f"({count} loaded)")

    if not X:
        # ── Fallback: synthetic data so the pipeline still runs ──────────
        print("[WARN] No MIDI files found — using random synthetic data.")
        N = 200
        X = [np.random.randint(0, 2, (seq_len, 128)).astype(np.float32)
             for _ in range(N)]
        y = [np.random.randint(0, len(Config.GENRES)) for _ in range(N)]

    print(f"\n[INFO] Dataset loaded: {len(X)} files total")
    return np.array(X), np.array(y)
