"""
Baseline models for music generation.
1. Random Note Generator (Naive)
2. Markov Chain Music Model
"""

import numpy as np
import pretty_midi
from typing import List, Dict, Tuple
from pathlib import Path


class RandomNoteGenerator:
    """
    Naive baseline: generates random sequences.
    Random pitches in [40, 80], random durations and velocities.
    """
    
    def __init__(self, pitch_range: Tuple[int, int] = (40, 80)):
        """
        Args:
            pitch_range: Min and max MIDI pitch
        """
        self.pitch_min, self.pitch_max = pitch_range
    
    def generate(self, n_notes: int = 100, bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
        """
        Generate a random sequence.
        
        Args:
            n_notes: Number of notes to generate
            bpm: Beats per minute
        
        Returns:
            PrettyMIDI object
        """
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        instrument = pretty_midi.Instrument(program=0)
        
        current_time = 0.0
        
        for _ in range(n_notes):
            pitch = np.random.randint(self.pitch_min, self.pitch_max + 1)
            duration = np.random.uniform(0.1, 0.5)  # 0.1 to 0.5 seconds
            velocity = np.random.randint(60, 101)
            
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            instrument.notes.append(note)
            current_time += duration
        
        midi_data.instruments.append(instrument)
        return midi_data


class MarkovChainMusicModel:
    """
    Markov chain-based baseline.
    Learns pitch transition probabilities from data and generates sequences.
    """
    
    def __init__(self, order: int = 1):
        """
        Args:
            order: Markov chain order (1 or 2)
        """
        self.order = order
        self.transition_matrix = {}
        self.initial_distribution = {}
        self.duration_stats = {}  # For sampling durations
        self.velocity_stats = {}   # For sampling velocities
    
    def train(self, midi_paths: List[str]):
        """
        Train Markov model from MIDI files.
        
        Args:
            midi_paths: List of paths to MIDI files
        """
        pitch_sequences = []
        durations = []
        velocities = []
        
        # Extract pitch sequences from all files
        for midi_path in midi_paths:
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                pitches = []
                
                for instrument in midi_data.instruments:
                    if not instrument.is_drum:
                        for note in instrument.notes:
                            pitches.append(note.pitch)
                            durations.append(note.end - note.start)
                            velocities.append(note.velocity)
                
                if pitches:
                    pitch_sequences.extend(pitches)
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")
        
        if not pitch_sequences:
            print("Warning: No pitch data extracted")
            return
        
        # Build initial distribution
        for pitch in pitch_sequences[:100]:  # Use first 100 as initial
            self.initial_distribution[pitch] = self.initial_distribution.get(pitch, 0) + 1
        
        # Normalize
        total = sum(self.initial_distribution.values())
        for pitch in self.initial_distribution:
            self.initial_distribution[pitch] /= total
        
        # Build transition matrix
        for i in range(len(pitch_sequences) - self.order):
            if self.order == 1:
                current = pitch_sequences[i]
                next_pitch = pitch_sequences[i + 1]
                key = (current,)
            else:  # order == 2
                current = (pitch_sequences[i], pitch_sequences[i + 1])
                next_pitch = pitch_sequences[i + 2]
                key = current
            
            if key not in self.transition_matrix:
                self.transition_matrix[key] = {}
            
            self.transition_matrix[key][next_pitch] = self.transition_matrix[key].get(next_pitch, 0) + 1
        
        # Normalize transitions
        for key in self.transition_matrix:
            total = sum(self.transition_matrix[key].values())
            for next_pitch in self.transition_matrix[key]:
                self.transition_matrix[key][next_pitch] /= total
        
        # Duration and velocity stats
        if durations:
            self.duration_stats = {
                'mean': np.mean(durations),
                'std': np.std(durations)
            }
        else:
            self.duration_stats = {'mean': 0.25, 'std': 0.1}
        
        if velocities:
            self.velocity_stats = {
                'mean': np.mean(velocities),
                'std': np.std(velocities)
            }
        else:
            self.velocity_stats = {'mean': 80, 'std': 10}
    
    def generate(self, n_notes: int = 100, bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
        """
        Generate sequence using Markov chain.
        
        Args:
            n_notes: Number of notes
            bpm: Beats per minute
        
        Returns:
            PrettyMIDI object
        """
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        instrument = pretty_midi.Instrument(program=0)
        
        if not self.initial_distribution:
            # Fall back to random if not trained
            print("Warning: Model not trained, falling back to random")
            return RandomNoteGenerator().generate(n_notes, bpm)
        
        current_time = 0.0
        
        # Initial pitch
        pitches = list(self.initial_distribution.keys())
        probs = list(self.initial_distribution.values())
        current_pitch = np.random.choice(pitches, p=probs)
        
        if self.order == 2:
            prev_pitch = current_pitch
        
        for _ in range(n_notes):
            # Generate duration and velocity
            duration = max(0.05, np.random.normal(
                self.duration_stats['mean'],
                self.duration_stats['std']
            ))
            velocity = int(np.clip(
                np.random.normal(self.velocity_stats['mean'], self.velocity_stats['std']),
                20, 127
            ))
            
            # Create note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=current_pitch,
                start=current_time,
                end=current_time + duration
            )
            instrument.notes.append(note)
            current_time += duration
            
            # Get next pitch
            if self.order == 1:
                key = (current_pitch,)
            else:
                key = (prev_pitch, current_pitch)
            
            if key in self.transition_matrix:
                next_pitches = list(self.transition_matrix[key].keys())
                next_probs = list(self.transition_matrix[key].values())
                next_pitch = np.random.choice(next_pitches, p=next_probs)
            else:
                # Fallback: random pitch from initial dist
                next_pitches = list(self.initial_distribution.keys())
                next_probs = list(self.initial_distribution.values())
                next_pitch = np.random.choice(next_pitches, p=next_probs)
            
            if self.order == 2:
                prev_pitch = current_pitch
            
            current_pitch = next_pitch
        
        midi_data.instruments.append(instrument)
        return midi_data


def generate_baseline_samples(
    output_dir: str,
    n_samples: int = 5,
    sample_type: str = "random"
):
    """
    Generate baseline samples and save to disk.
    
    Args:
        output_dir: Directory to save MIDI files
        n_samples: Number of samples to generate
        sample_type: "random" or "markov"
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if sample_type == "random":
        generator = RandomNoteGenerator()
        for i in range(n_samples):
            midi = generator.generate(n_notes=100)
            output_path = Path(output_dir) / f"random_sample_{i:02d}.mid"
            midi.write(output_path)
            print(f"Generated {output_path}")
    
    elif sample_type == "markov":
        generator = MarkovChainMusicModel(order=1)
        print("Training Markov model (using random data for this example)...")
        # In real usage, would load training data
        generator.initial_distribution = {pitch: 1.0/40 for pitch in range(40, 80)}
        
        for i in range(n_samples):
            midi = generator.generate(n_notes=100)
            output_path = Path(output_dir) / f"markov_sample_{i:02d}.mid"
            midi.write(output_path)
            print(f"Generated {output_path}")


if __name__ == "__main__":
    print("Baseline models module loaded.")
