"""
Convert MIDI files to piano roll representation.
Piano roll: (time_steps, 128 pitches) binary/velocity array
Normalized to 16 steps per bar.
"""

import numpy as np
import pretty_midi
from typing import Tuple
from .normalize_timing import normalize_midi_timing


def midi_to_pianoroll(
    midi_path: str,
    fs: int = 16,
    seq_len: int = 256,
    binarize: bool = True
) -> np.ndarray:
    """
    Convert MIDI file to piano roll representation.
    
    Args:
        midi_path: Path to MIDI file
        fs: Time resolution (steps per beat, default 16)
        seq_len: Fixed sequence length (pad/truncate to this)
        binarize: If True, output binary (0/1). If False, output velocities.
    
    Returns:
        Piano roll array of shape (seq_len, 128)
    """
    # Load and normalize MIDI
    midi_data = normalize_midi_timing(midi_path, fs=fs)
    
    # Calculate piano roll length
    end_time = midi_data.get_end_time()
    n_steps = int(np.ceil(end_time * fs))
    
    # Initialize piano roll
    piano_roll = np.zeros((n_steps, 128), dtype=np.float32)
    
    # Populate piano roll
    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # Skip drums for now
            for note in instrument.notes:
                start_idx = int(np.round(note.start * fs))
                end_idx = int(np.round(note.end * fs))
                
                # Clamp to valid range
                start_idx = max(0, min(start_idx, n_steps - 1))
                end_idx = max(start_idx + 1, min(end_idx, n_steps))
                
                # Set velocity or binary
                velocity = note.velocity / 127.0  # Normalize to [0, 1]
                if binarize and velocity > 0:
                    piano_roll[start_idx:end_idx, note.pitch] = 1.0
                else:
                    piano_roll[start_idx:end_idx, note.pitch] = velocity
    
    # Pad or truncate to seq_len
    if n_steps < seq_len:
        # Pad with zeros
        padded = np.zeros((seq_len, 128), dtype=np.float32)
        padded[:n_steps] = piano_roll
        return padded
    else:
        # Truncate (take first seq_len steps)
        return piano_roll[:seq_len]


def segment_piano_roll(
    midi_path: str,
    fs: int = 16,
    seq_len: int = 256,
    step: int = 128
) -> list:
    """
    Segment a long MIDI file into overlapping piano roll chunks.
    Useful for songs longer than seq_len.
    
    Args:
        midi_path: Path to MIDI file
        fs: Time resolution
        seq_len: Chunk length
        step: Overlap step size
    
    Returns:
        List of piano roll arrays
    """
    midi_data = normalize_midi_timing(midi_path, fs=fs)
    end_time = midi_data.get_end_time()
    n_steps = int(np.ceil(end_time * fs))
    
    # Build full piano roll
    piano_roll = np.zeros((n_steps, 128), dtype=np.float32)
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                start_idx = int(np.round(note.start * fs))
                end_idx = int(np.round(note.end * fs))
                start_idx = max(0, min(start_idx, n_steps - 1))
                end_idx = max(start_idx + 1, min(end_idx, n_steps))
                velocity = note.velocity / 127.0
                piano_roll[start_idx:end_idx, note.pitch] = max(
                    piano_roll[start_idx:end_idx, note.pitch],
                    velocity * np.ones(end_idx - start_idx)
                )
    
    # Segment with stride
    segments = []
    for start in range(0, n_steps - seq_len + 1, step):
        segment = piano_roll[start:start + seq_len]
        segments.append(segment)
    
    # Handle last segment
    if n_steps - (len(segments) - 1) * step > seq_len:
        last_segment = piano_roll[-seq_len:]
        segments.append(last_segment)
    
    return segments


def pianoroll_to_midi(
    piano_roll: np.ndarray,
    fs: int = 16,
    bpm: float = 120.0,
    output_path: str = None
) -> pretty_midi.PrettyMIDI:
    """
    Convert piano roll back to MIDI.
    
    Args:
        piano_roll: Array of shape (seq_len, 128)
        fs: Time resolution (steps per beat)
        bpm: Beats per minute
        output_path: Optional path to save MIDI
    
    Returns:
        PrettyMIDI object
    """
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    
    # Extract notes from piano roll
    for pitch in range(128):
        note_on = False
        start_time = None
        
        for step in range(piano_roll.shape[0]):
            if piano_roll[step, pitch] > 0:
                if not note_on:
                    note_on = True
                    start_time = step / fs
            else:
                if note_on:
                    note_on = False
                    end_time = step / fs
                    velocity = int(np.round(127 * piano_roll[step - 1, pitch]))
                    velocity = max(1, min(127, velocity))
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
        
        # Handle note that extends to end
        if note_on:
            end_time = piano_roll.shape[0] / fs
            velocity = int(np.round(127 * piano_roll[-1, pitch]))
            velocity = max(1, min(127, velocity))
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
    
    midi_data.instruments.append(instrument)
    
    if output_path:
        midi_data.write(output_path)
    
    return midi_data


if __name__ == "__main__":
    print("Piano roll conversion module loaded.")
