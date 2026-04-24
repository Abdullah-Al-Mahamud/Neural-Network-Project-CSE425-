# ============================================================
# midi_export.py — Convert piano-roll tensors → MIDI files
# ============================================================
import os
import numpy as np
import pretty_midi
import torch
from scipy.ndimage import gaussian_filter1d
from src.config import Config


def piano_roll_to_midi(roll: np.ndarray,
                        bpm: float = 120.0,
                        steps_per_bar: int = Config.STEPS_PER_BAR,
                        velocity: int = 80) -> pretty_midi.PrettyMIDI:
    """
    Convert a binary piano-roll (T, 128) -> PrettyMIDI object.
    Uses soft pruning (threshold 0.3) and note grouping.
    """
    step_duration = (60.0 / bpm) / (steps_per_bar / 4.0)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)

    # Soft threshold at 0.3 instead of 0.5 for better note activation
    roll_thresh = (roll > 0.3).astype(np.float32)
    
    T = roll.shape[0]
    for pitch in range(128):
        t = 0
        while t < T:
            if roll_thresh[t, pitch] > 0:
                start = t * step_duration
                end_t = t + 1
                
                # Group consecutive active steps into single note
                while end_t < T and roll_thresh[end_t, pitch] > 0:
                    end_t += 1
                
                end = end_t * step_duration
                # Minimum 1/8 note duration
                duration = max(end - start, step_duration * 2)
                note = pretty_midi.Note(velocity=velocity,
                                        pitch=pitch,
                                        start=start,
                                        end=start + duration)
                instrument.notes.append(note)
                t = end_t
            else:
                t += 1

    instrument.notes.sort(key=lambda n: n.start)
    pm.instruments.append(instrument)
    return pm


def save_piano_roll_samples(samples, prefix: str = "sample",
                             out_dir: str = Config.OUTPUT_DIR,
                             bpm: float = 120.0):
    """
    Save piano-roll samples with probability-based note selection.
    Uses sigmoid outputs directly to improve note activation.
    """
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    saved = []
    for i, roll in enumerate(samples):
        # Apply soft thresholding: either direct or via max-pooling for note merging
        # Smooth the roll to encourage longer note durations
        from scipy.ndimage import gaussian_filter1d
        roll_smooth = gaussian_filter1d(roll, sigma=0.8, axis=0)
        
        pm = piano_roll_to_midi(roll_smooth, bpm=bpm)
        path = os.path.join(out_dir, f"{prefix}_{i+1:02d}.mid")
        pm.write(path)
        saved.append(path)
        print(f"  Saved: {path}")
    return saved


def tokens_to_midi(token_seq: np.ndarray,
                    bpm: float = 120.0,
                    steps_per_bar: int = Config.STEPS_PER_BAR) -> pretty_midi.PrettyMIDI:
    """
    Convert token sequence to MIDI with intelligent note grouping.
    Consecutive identical pitch tokens are merged into single notes.
    """
    step_dur = (60.0 / bpm) / (steps_per_bar / 4.0)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)

    t = 0
    while t < len(token_seq):
        tok = int(token_seq[t])
        
        # Valid pitch tokens (0-127)
        if 0 <= tok < 128:
            start_t = t
            # Group consecutive identical pitches
            while t < len(token_seq) and int(token_seq[t]) == tok:
                t += 1
            
            # Create note with grouped duration (minimum 1/8 note = 2 steps)
            duration_steps = max(t - start_t, 2)
            note = pretty_midi.Note(
                velocity=85, pitch=tok,
                start=start_t * step_dur,
                end=(start_t + duration_steps) * step_dur
            )
            instrument.notes.append(note)
        else:
            t += 1

    pm.instruments.append(instrument)
    return pm


def save_token_samples(token_seqs, prefix: str = "transformer_sample",
                        out_dir: str = Config.OUTPUT_DIR):
    """
    token_seqs : torch.Tensor (N, T) or np.ndarray
    """
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(token_seqs, torch.Tensor):
        token_seqs = token_seqs.cpu().numpy()

    saved = []
    for i, seq in enumerate(token_seqs):
        pm = tokens_to_midi(seq)
        path = os.path.join(out_dir, f"{prefix}_{i+1:02d}.mid")
        pm.write(path)
        saved.append(path)
        print(f"  Saved: {path}")
    return saved
