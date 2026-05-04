"""
Normalize MIDI timing to 16 steps per bar (16th-note resolution).
This ensures consistent quantization across all datasets.
"""

import pretty_midi
import numpy as np
from typing import Tuple


def normalize_midi_timing(midi_path: str, fs: int = 16) -> pretty_midi.PrettyMIDI:
    """
    Load MIDI file and normalize timing resolution.
    
    Args:
        midi_path: Path to MIDI file
        fs: Steps per beat (default 16 = 16th-note resolution)
    
    Returns:
        Normalized PrettyMIDI object
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Normalize end time to grid
    end_time = midi_data.get_end_time()
    resolution = 1.0 / fs  # Time per step
    normalized_end_time = np.ceil(end_time / resolution) * resolution
    
    # Quantize all note times to nearest grid step
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.start = np.round(note.start / resolution) * resolution
                note.end = np.round(note.end / resolution) * resolution
                # Ensure end > start
                if note.end <= note.start:
                    note.end = note.start + resolution
    
    return midi_data


def get_timing_stats(midi_path: str) -> Tuple[float, float]:
    """Get duration and tempo info from MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    duration = midi_data.get_end_time()
    # Estimate tempo (in BPM)
    if len(midi_data.get_tempo_changes()[0]) > 0:
        tempos = midi_data.get_tempo_changes()[1]
        avg_tempo = np.mean(tempos)
    else:
        avg_tempo = 120.0  # Default
    return duration, avg_tempo


if __name__ == "__main__":
    print("Timing normalization module loaded.")
