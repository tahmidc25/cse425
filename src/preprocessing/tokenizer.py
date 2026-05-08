"""
Convert MIDI files to token sequences for Transformer-based models.
Token vocabulary:
  [0]          = BOS (beginning of sequence)
  [1-128]      = NOTE_ON (pitch 0-127)
  [129-160]    = VELOCITY (32 bins)
  [161-224]    = DURATION (64 bins)
  [225]        = TIME_SHIFT
  [226]        = EOS (end of sequence)
  [227]        = PAD (padding)
  Total vocab size: 228
"""

import numpy as np
import pretty_midi
from typing import List, Tuple
from .normalize_timing import normalize_midi_timing


# Token IDs
BOS_ID = 0
NOTE_ON_START = 1
NOTE_ON_END = 128  # 1 + 127
VELOCITY_START = 129
VELOCITY_END = 161  # 129 + 32
DURATION_START = 161
DURATION_END = 225  # 161 + 64
TIME_SHIFT_ID = 225
EOS_ID = 226
PAD_ID = 227

VOCAB_SIZE = 228
MAX_SEQ_LEN = 512


def quantize_velocity(velocity: int, n_bins: int = 32) -> int:
    """Map velocity [0, 127] to bin [0, n_bins-1]."""
    return min(n_bins - 1, max(0, int(velocity * n_bins / 128)))


def quantize_duration(duration: float, fs: int = 16, n_bins: int = 64) -> int:
    """Map duration in seconds to bin [0, n_bins-1]."""
    # Duration in steps: duration * fs
    steps = int(np.round(duration * fs))
    # Clamp to [1, n_bins]
    steps = max(1, min(n_bins, steps))
    return steps - 1


def midi_to_tokens(
    midi_path: str,
    fs: int = 16,
    max_seq_len: int = MAX_SEQ_LEN,
    include_bos_eos: bool = True
) -> np.ndarray:
    """
    Convert MIDI file to token sequence.
    
    Args:
        midi_path: Path to MIDI file
        fs: Time resolution (steps per beat)
        max_seq_len: Maximum sequence length (will pad/truncate)
        include_bos_eos: Whether to include BOS/EOS tokens
    
    Returns:
        Token array of shape (seq_len,) with values in [0, 227]
    """
    midi_data = normalize_midi_timing(midi_path, fs=fs)
    
    tokens = []
    
    if include_bos_eos:
        tokens.append(BOS_ID)
    
    # Collect all note events
    events = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                events.append(('note', note.start, note.end, note.pitch, note.velocity))
    
    # Sort by start time
    events.sort(key=lambda x: x[1])
    
    # Generate tokens
    for event_type, start, end, pitch, velocity in events:
        # NOTE_ON token
        tokens.append(NOTE_ON_START + pitch)
        
        # VELOCITY token
        vel_bin = quantize_velocity(velocity)
        tokens.append(VELOCITY_START + vel_bin)
        
        # DURATION token
        duration = end - start
        dur_bin = quantize_duration(duration, fs=fs)
        tokens.append(DURATION_START + dur_bin)
        
        # TIME_SHIFT (optional: for inter-note gaps)
        tokens.append(TIME_SHIFT_ID)
    
    if include_bos_eos:
        tokens.append(EOS_ID)
    
    # Pad or truncate to max_seq_len
    token_array = np.full(max_seq_len, PAD_ID, dtype=np.int32)
    seq_len = min(len(tokens), max_seq_len)
    token_array[:seq_len] = tokens[:seq_len]
    
    return token_array


def tokens_to_midi(
    tokens: np.ndarray,
    fs: int = 16,
    bpm: float = 120.0,
    output_path: str = None
) -> pretty_midi.PrettyMIDI:
    """
    Convert token sequence back to MIDI.
    
    Args:
        tokens: Array of token IDs
        fs: Time resolution
        bpm: Beats per minute
        output_path: Optional path to save MIDI
    
    Returns:
        PrettyMIDI object
    """
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    
    current_time = 0.0
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Skip padding
        if token == PAD_ID:
            i += 1
            continue
        
        # Skip BOS/EOS
        if token == BOS_ID or token == EOS_ID:
            i += 1
            continue
        
        # NOTE_ON
        if NOTE_ON_START <= token < VELOCITY_START:
            pitch = token - NOTE_ON_START
            
            # Next token should be VELOCITY
            if i + 1 < len(tokens) and VELOCITY_START <= tokens[i + 1] < DURATION_START:
                vel_bin = tokens[i + 1] - VELOCITY_START
                velocity = int(np.round((vel_bin + 0.5) * 128 / 32))
                velocity = max(1, min(127, velocity))
            else:
                velocity = 64
            
            # Next token should be DURATION
            if i + 2 < len(tokens) and DURATION_START <= tokens[i + 2] < TIME_SHIFT_ID:
                dur_bin = tokens[i + 2] - DURATION_START
                duration = (dur_bin + 1) / fs
            else:
                duration = 0.5
            
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            instrument.notes.append(note)
            
            # Skip to next note
            i += 3
            
            # Skip TIME_SHIFT if present
            if i < len(tokens) and tokens[i] == TIME_SHIFT_ID:
                i += 1
        else:
            i += 1
    
    midi_data.instruments.append(instrument)
    
    if output_path:
        midi_data.write(output_path)
    
    return midi_data


def tokens_to_pianoroll(
    tokens: np.ndarray,
    fs: int = 16,
    seq_len: int = 256
) -> np.ndarray:
    """Convert tokens to piano roll for comparison/evaluation."""
    midi = tokens_to_midi(tokens, fs=fs)
    
    # Convert to piano roll
    piano_roll = np.zeros((seq_len, 128), dtype=np.float32)
    end_time = midi.get_end_time()
    n_steps = int(np.ceil(end_time * fs))
    
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                start_idx = int(np.round(note.start * fs))
                end_idx = int(np.round(note.end * fs))
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx + 1, min(end_idx, seq_len))
                
                velocity = note.velocity / 127.0
                piano_roll[start_idx:end_idx, note.pitch] = velocity
    
    return piano_roll


if __name__ == "__main__":
    print(f"Tokenization module loaded. Vocab size: {VOCAB_SIZE}")
