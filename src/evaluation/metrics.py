"""
Evaluation metrics for music generation quality assessment.
Implements:
1. Pitch Histogram Similarity
2. Rhythm Diversity Score
3. Repetition Ratio
4. Perplexity (for Transformer)
5. Human Listening Score (from survey)
"""

import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
from pathlib import Path


class MusicMetrics:
    """Collection of music evaluation metrics."""
    
    @staticmethod
    def pitch_histogram(piano_roll: np.ndarray) -> np.ndarray:
        """
        Compute 12-bin chromatic pitch histogram.
        
        Args:
            piano_roll: (seq_len, 128) array
        
        Returns:
            12-bin histogram (normalized)
        """
        histogram = np.zeros(12)
        
        # Sum over time and fold pitches into 12 semitones
        for pitch in range(128):
            pitch_class = pitch % 12
            histogram[pitch_class] += np.sum(piano_roll[:, pitch])
        
        # Normalize
        histogram = histogram / (np.sum(histogram) + 1e-6)
        
        return histogram
    
    @staticmethod
    def pitch_histogram_similarity(
        piano_roll1: np.ndarray,
        piano_roll2: np.ndarray
    ) -> float:
        """
        Compute L1 distance between pitch histograms.
        Lower = more similar
        
        Args:
            piano_roll1: Reference (seq_len, 128)
            piano_roll2: Generated (seq_len, 128)
        
        Returns:
            L1 distance in [0, 2]
        """
        hist1 = MusicMetrics.pitch_histogram(piano_roll1)
        hist2 = MusicMetrics.pitch_histogram(piano_roll2)
        
        distance = np.sum(np.abs(hist1 - hist2))
        
        return float(distance)
    
    @staticmethod
    def extract_notes(piano_roll: np.ndarray, fs: int = 16) -> List[Tuple[int, float, float]]:
        """
        Extract note events from piano roll.
        
        Args:
            piano_roll: (seq_len, 128) array
            fs: Sampling frequency (steps per beat)
        
        Returns:
            List of (pitch, start_time, duration) tuples
        """
        notes = []
        
        for pitch in range(128):
            note_on = False
            start_idx = None
            
            for step in range(piano_roll.shape[0]):
                if piano_roll[step, pitch] > 0:
                    if not note_on:
                        note_on = True
                        start_idx = step
                else:
                    if note_on:
                        note_on = False
                        duration = (step - start_idx) / fs
                        start_time = start_idx / fs
                        notes.append((pitch, start_time, duration))
            
            # Handle note extending to end
            if note_on:
                duration = (piano_roll.shape[0] - start_idx) / fs
                start_time = start_idx / fs
                notes.append((pitch, start_time, duration))
        
        return notes
    
    @staticmethod
    def rhythm_diversity(piano_roll: np.ndarray, fs: int = 16) -> float:
        """
        Compute rhythm diversity score.
        Score = #unique_durations / #total_notes
        Higher = more diverse rhythm
        
        Args:
            piano_roll: (seq_len, 128) array
            fs: Sampling frequency
        
        Returns:
            Diversity score in [0, 1]
        """
        notes = MusicMetrics.extract_notes(piano_roll, fs=fs)
        
        if len(notes) == 0:
            return 0.0
        
        # Extract durations
        durations = [note[2] for note in notes]
        unique_durations = len(set([round(d, 2) for d in durations]))
        
        diversity = unique_durations / len(notes)
        
        return float(np.clip(diversity, 0.0, 1.0))
    
    @staticmethod
    def repetition_ratio(piano_roll: np.ndarray, pattern_length: int = 4) -> float:
        """
        Compute repetition ratio.
        Ratio = #repeated_patterns / #total_patterns
        Lower = less repetitive
        
        Args:
            piano_roll: (seq_len, 128) array
            pattern_length: Length of pattern to check (in time steps)
        
        Returns:
            Repetition ratio in [0, 1]
        """
        if piano_roll.shape[0] < 2 * pattern_length:
            return 0.0
        
        repetitions = 0
        total_patterns = 0
        
        for i in range(piano_roll.shape[0] - 2 * pattern_length):
            pattern1 = piano_roll[i:i+pattern_length]
            pattern2 = piano_roll[i+pattern_length:i+2*pattern_length]
            
            if np.allclose(pattern1, pattern2, atol=0.01):
                repetitions += 1
            
            total_patterns += 1
        
        if total_patterns == 0:
            return 0.0
        
        ratio = repetitions / total_patterns
        
        return float(np.clip(ratio, 0.0, 1.0))
    
    @staticmethod
    def perplexity_from_loss(loss: float) -> float:
        """
        Compute perplexity from cross-entropy loss.
        Perplexity = exp(loss)
        Lower = better
        
        Args:
            loss: Cross-entropy loss value
        
        Returns:
            Perplexity
        """
        return float(np.exp(loss))
    
    @staticmethod
    def human_listening_score(survey_data: Dict) -> Dict:
        """
        Aggregate human survey scores.
        
        Args:
            survey_data: Dict with sample -> ratings
        
        Returns:
            Aggregated statistics
        """
        all_means = []
        all_stds = []
        
        for sample_key, sample_data in survey_data.items():
            if isinstance(sample_data, dict) and 'mean' in sample_data:
                all_means.append(sample_data['mean'])
                all_stds.append(sample_data['std'])
        
        if not all_means:
            return {'mean': 0.0, 'std': 0.0, 'count': 0}
        
        return {
            'mean': float(np.mean(all_means)),
            'std': float(np.mean(all_stds)),
            'count': len(all_means)
        }


class ComparisonTable:
    """Generate and save comparison tables."""
    
    def __init__(self):
        """Initialize comparison table."""
        self.rows = []
    
    def add_row(
        self,
        model_name: str,
        loss: float = None,
        perplexity: float = None,
        rhythm_diversity: float = None,
        human_score: float = None,
        genre_control: str = None
    ):
        """
        Add row to comparison table.
        
        Args:
            model_name: Name of model
            loss: Training loss (optional)
            perplexity: Perplexity score (optional)
            rhythm_diversity: Rhythm diversity metric
            human_score: Human listening score
            genre_control: Genre control capability
        """
        row = {
            'Model': model_name,
            'Loss': f"{loss:.4f}" if loss is not None else "—",
            'Perplexity': f"{perplexity:.4f}" if perplexity is not None else "—",
            'Rhythm Diversity': f"{rhythm_diversity:.4f}" if rhythm_diversity is not None else "—",
            'Human Score': f"{human_score:.2f}" if human_score is not None else "—",
            'Genre Control': genre_control if genre_control is not None else "—"
        }
        self.rows.append(row)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.rows)
    
    def to_csv(self, output_path: str):
        """Save to CSV."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Saved comparison table to {output_path}")
    
    def to_latex(self, output_path: str):
        """Save to LaTeX table format."""
        df = self.to_dataframe()
        latex = df.to_latex(index=False)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Saved LaTeX table to {output_path}")


def evaluate_multiple_samples(
    samples: List[np.ndarray],
    reference_samples: List[np.ndarray] = None
) -> Dict:
    """
    Evaluate collection of generated samples.
    
    Args:
        samples: List of generated piano rolls
        reference_samples: Reference samples for comparison (optional)
    
    Returns:
        Evaluation metrics dict
    """
    metrics = MusicMetrics()
    
    results = {
        'rhythm_diversity': [],
        'repetition_ratio': [],
        'pitch_similarity': []
    }
    
    for i, sample in enumerate(samples):
        # Rhythm diversity
        rhythm_div = metrics.rhythm_diversity(sample)
        results['rhythm_diversity'].append(rhythm_div)
        
        # Repetition ratio
        rep_ratio = metrics.repetition_ratio(sample)
        results['repetition_ratio'].append(rep_ratio)
        
        # Pitch similarity (if reference available)
        if reference_samples and i < len(reference_samples):
            pitch_sim = metrics.pitch_histogram_similarity(
                reference_samples[i],
                sample
            )
            results['pitch_similarity'].append(pitch_sim)
    
    # Aggregate
    summary = {
        'avg_rhythm_diversity': float(np.mean(results['rhythm_diversity'])),
        'avg_repetition_ratio': float(np.mean(results['repetition_ratio'])),
        'all_rhythm_diversity': results['rhythm_diversity'],
        'all_repetition_ratio': results['repetition_ratio']
    }
    
    if results['pitch_similarity']:
        summary['avg_pitch_similarity'] = float(np.mean(results['pitch_similarity']))
        summary['all_pitch_similarity'] = results['pitch_similarity']
    
    return summary


if __name__ == "__main__":
    print("Metrics module loaded.")
    
    # Test
    metrics = MusicMetrics()
    
    dummy_pr = np.random.rand(256, 128)
    dummy_ref = np.random.rand(256, 128)
    
    rhythm_div = metrics.rhythm_diversity(dummy_pr)
    rep_ratio = metrics.repetition_ratio(dummy_pr)
    pitch_sim = metrics.pitch_histogram_similarity(dummy_ref, dummy_pr)
    
    print(f"Rhythm diversity: {rhythm_div:.4f}")
    print(f"Repetition ratio: {rep_ratio:.4f}")
    print(f"Pitch similarity: {pitch_sim:.4f}")
