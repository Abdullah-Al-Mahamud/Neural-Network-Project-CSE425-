#!/usr/bin/env python3
"""
Verification script — checks project integrity and dependencies.
Run this before submitting to ensure everything is in place.
"""
import os
import sys
from pathlib import Path

REQUIRED_DIRS = [
    "data/processed/classical",
    "data/processed/jazz",
    "data/processed/rock",
    "data/processed/pop",
    "data/processed/electronic",
    "src/models",
    "src/preprocessing",
    "src/training",
    "evaluation",
    "generation",
    "outputs/generated_midis",
    "outputs/plots",
    "models",
]

REQUIRED_FILES = [
    "main.py",
    "requirements.txt",
    "README.md",
    "src/__init__.py",
    "src/config.py",
    "src/models/autoencoder.py",
    "src/models/vae.py",
    "src/models/transformer.py",
    "src/models/rlhf.py",
    "src/preprocessing/midi_parser.py",
    "src/training/train_ae.py",
    "src/training/train_vae.py",
    "src/training/train_transformer.py",
    "evaluation/metrics.py",
    "evaluation/baselines.py",
    "generation/midi_export.py",
    "scripts/prep_lakh_dataset.py",
]

REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "pretty_midi",
    "matplotlib",
    "sklearn",
]

def check_structure():
    """Verify all required directories and files exist."""
    print("\n" + "="*70)
    print("  PROJECT INTEGRITY CHECK")
    print("="*70 + "\n")
    
    missing_dirs = []
    for d in REQUIRED_DIRS:
        if not os.path.isdir(d):
            missing_dirs.append(d)
    
    missing_files = []
    for f in REQUIRED_FILES:
        if not os.path.isfile(f):
            missing_files.append(f)
    
    # Report directories
    print(f"[Directories] Checking {len(REQUIRED_DIRS)} directories...")
    if missing_dirs:
        print(f"  ❌ Missing {len(missing_dirs)} directories:")
        for d in missing_dirs:
            print(f"     - {d}")
    else:
        print(f"  ✅ All {len(REQUIRED_DIRS)} directories present")
    
    # Report files
    print(f"\n[Files] Checking {len(REQUIRED_FILES)} files...")
    if missing_files:
        print(f"  ❌ Missing {len(missing_files)} files:")
        for f in missing_files:
            print(f"     - {f}")
    else:
        print(f"  ✅ All {len(REQUIRED_FILES)} files present")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def check_dependencies():
    """Verify Python packages are installed."""
    print(f"\n[Dependencies] Checking {len(REQUIRED_PACKAGES)} packages...")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"  ❌ Missing {len(missing)} packages:")
        for p in missing:
            print(f"     - {p}  (run: pip install {p})")
    else:
        print(f"  ✅ All {len(REQUIRED_PACKAGES)} packages installed")
    
    return len(missing) == 0

def check_dataset():
    """Verify dataset exists."""
    print("\n[Dataset] Checking data files...")
    genres = ["classical", "jazz", "rock", "pop", "electronic"]
    total_files = 0
    for genre in genres:
        path = f"data/processed/{genre}"
        files = len([f for f in os.listdir(path) if f.endswith((".mid", ".midi"))])
        total_files += files
        status = "✅" if files > 0 else "⚠️"
        print(f"  {status} {genre}: {files} MIDI files")
    
    if total_files == 0:
        print(f"\n  ⚠️  No MIDI files found. Run: python scripts/prep_lakh_dataset.py")
        return False
    
    print(f"\n  ✅ Total: {total_files} MIDI files ready for training")
    return True

def main():
    os.chdir(Path(__file__).parent)
    
    structure_ok = check_structure()
    deps_ok = check_dependencies()
    data_ok = check_dataset()
    
    print("\n" + "="*70)
    if structure_ok and deps_ok:
        print("  ✅ PROJECT READY FOR EXECUTION")
        print("="*70)
        print("\n  Next steps:")
        if not data_ok:
            print("  1. python scripts/prep_lakh_dataset.py  (prepare dataset)")
        print("  2. python main.py                          (run all tasks)")
        print("\n")
        return 0
    else:
        print("  ❌ PROJECT REQUIRES FIXES")
        print("="*70)
        if not structure_ok:
            print("  - Create missing directories/files")
        if not deps_ok:
            print("  - Install missing packages (pip install -r requirements.txt)")
        if not data_ok:
            print("  - Run: python scripts/prep_lakh_dataset.py")
        print("\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
