#!/usr/bin/env python3
"""
Prepare Lakh MIDI dataset for the music generation project.
Maps artist folders to genres and copies MIDI files.
"""
import os
import shutil
import glob
from pathlib import Path

LAKH_ROOT = r"h:\Lakh MIDI Dataset Clean"
OUTPUT_DIR = r"h:\claude solution\music-generation\data\processed"

# Genre mapping based on artist names and patterns
GENRE_MAPPING = {
    "classical": [
        "Bach", "Beethoven", "Mozart", "Chopin", "Brahms", "Debussy", 
        "Wagner", "Handel", "Haydn", "Brahms", "Dvorak", "Grieg",
        "MAESTRO", "Liszt", "Vivaldi", "Barber", "Gershwin"
    ],
    "jazz": [
        "Coltrane", "Davis_Miles", "Ella", "Fitzgerald", "Jazz",
        "Basie", "Ellington", "Duke", "Dizzy", "Gillespie",
        "Monk", "Thelonious", "Brubeck", "Dave", "Peterson", "Oscar",
        "Mulligan", "Gerry", "Stardust", "Saxo", "Quartet", "Trio",
    ],
    "rock": [
        "Beatles", "Rolling_Stones", "Zeppelin", "Led", "Pink_Floyd",
        "Queen", "Bowie", "David", "Jimi_Hendrix", "Hendrix",
        "U2", "The_Who", "Who", "Eagles", "Genesis", "Yes",
        "Floyd", "Sabbath", "Black", "Metallica", "AC_DC", "AC/DC",
        "Van_Halen", "Guns", "Roses", "Bon_Jovi", "Skynyrd", "Lynyrd",
    ],
    "pop": [
        "Madonna", "Jackson_Michael", "Michael", "Mariah", "Carey",
        "Britney", "Spears", "Christina", "Aguilera", "Eminem",
        "Dr", "Dre", "50_Cent", "Drake", "Jay", "West", "Kanye",
        "Swift", "Taylor", "Rihanna", "Beyonce", "Knowles", "Houston",
        "Whitney", "Prince", "Prince", "Presley", "Elvis", "Sinatra",
        "Frank", "Cole", "Nat", "King", "Perry", "Katy",
    ],
    "electronic": [
        "Daft_Punk", "Depeche", "Mode", "Kraftwerk", "Fatboy",
        "Slim", "Chemical", "Brothers", "Prodigy", "Moby",
        "Deadmau5", "Skrillex", "Tiësto", "Afrojack", "Dj",
        "Trance", "EDM", "Dubstep", "House", "Techno", "Synth",
        "Erasure", "Depeche", "Mode", "Pet_Shop", "Boys",
    ]
}

def get_genre_for_artist(artist_name):
    """Determine genre based on artist name."""
    name_lower = artist_name.lower()
    for genre, keywords in GENRE_MAPPING.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                return genre
    # Default to 'pop' if no match
    return "pop"

def prepare_dataset():
    """Copy MIDI files from Lakh to organized genre folders."""
    print("Preparing Lakh MIDI dataset...")
    print(f"Source: {LAKH_ROOT}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Create genre directories
    for genre in ["classical", "jazz", "rock", "pop", "electronic"]:
        os.makedirs(os.path.join(OUTPUT_DIR, genre), exist_ok=True)
    
    # Counters
    genre_counts = {g: 0 for g in ["classical", "jazz", "rock", "pop", "electronic"]}
    total_copied = 0
    
    # List artist directories
    artist_dirs = sorted([d for d in os.listdir(LAKH_ROOT) 
                         if os.path.isdir(os.path.join(LAKH_ROOT, d))])
    
    print(f"Found {len(artist_dirs)} artist directories. Processing...\n")
    
    for artist_idx, artist in enumerate(artist_dirs):
        if artist.startswith('.'):
            continue
            
        artist_path = os.path.join(LAKH_ROOT, artist)
        genre = get_genre_for_artist(artist)
        genre_dir = os.path.join(OUTPUT_DIR, genre)
        
        # Find all MIDI files in artist directory
        midi_files = glob.glob(os.path.join(artist_path, "*.mid")) + \
                     glob.glob(os.path.join(artist_path, "*.midi"))
        
        for midi_file in midi_files:
            try:
                dst_name = f"{artist}_{os.path.basename(midi_file)}"
                dst_path = os.path.join(genre_dir, dst_name)
                shutil.copy2(midi_file, dst_path)
                genre_counts[genre] += 1
                total_copied += 1
            except Exception as e:
                print(f"  [ERROR] {artist}/{midi_file}: {e}")
        
        if (artist_idx + 1) % 100 == 0:
            print(f"  Processed {artist_idx + 1}/{len(artist_dirs)} artists...")
    
    print(f"\n{'='*50}")
    print("Dataset Preparation Complete!")
    print(f"{'='*50}")
    print(f"\nTotal MIDI files copied: {total_copied}")
    print(f"\nGenre distribution:")
    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre:15s}: {count:5d} files")
    print(f"\n  Total:          {sum(genre_counts.values()):5d} files")
    print(f"\nFiles are organized in: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_dataset()
