"""
Main dataset preprocessing orchestration script.
Handles MIDI loading, conversion to piano rolls/tokens, and train/test splitting.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.piano_roll import midi_to_pianoroll, segment_piano_roll
from preprocessing.tokenizer import midi_to_tokens, VOCAB_SIZE
from preprocessing.normalize_timing import normalize_midi_timing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicDatasetPreprocessor:
    """Orchestrates preprocessing of music datasets."""
    
    def __init__(
        self,
        raw_midi_dir: str,
        processed_dir: str,
        split_dir: str,
        fs: int = 16,
        piano_roll_len: int = 256,
        token_seq_len: int = 512
    ):
        """
        Args:
            raw_midi_dir: Directory containing raw MIDI files
            processed_dir: Output directory for processed files
            split_dir: Directory for train/test splits
            fs: Sampling frequency (steps per beat)
            piano_roll_len: Piano roll sequence length
            token_seq_len: Token sequence length
        """
        self.raw_midi_dir = Path(raw_midi_dir)
        self.processed_dir = Path(processed_dir)
        self.split_dir = Path(split_dir)
        
        self.fs = fs
        self.piano_roll_len = piano_roll_len
        self.token_seq_len = token_seq_len
        
        # Create output directories
        (self.processed_dir / "piano_roll").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "tokens").mkdir(parents=True, exist_ok=True)
        self.split_dir.mkdir(parents=True, exist_ok=True)
    
    def find_midi_files(self, directory: str, recursive: bool = True) -> List[str]:
        """Find all MIDI files in a directory."""
        midi_files = []
        search_pattern = "**/*.mid" if recursive else "*.mid"
        for filepath in self.raw_midi_dir.glob(search_pattern):
            midi_files.append(str(filepath))
        return sorted(midi_files)
    
    def process_midi_to_pianoroll(self, midi_path: str, output_dir: str) -> str:
        """Convert MIDI to piano roll and save."""
        try:
            piano_roll = midi_to_pianoroll(
                midi_path,
                fs=self.fs,
                seq_len=self.piano_roll_len,
                binarize=True
            )
            
            # Save with original filename
            filename = Path(midi_path).stem + ".npy"
            output_path = Path(output_dir) / filename
            np.save(output_path, piano_roll)
            
            logger.info(f"Saved piano roll: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.warning(f"Failed to process {midi_path}: {e}")
            return None
    
    def process_midi_to_tokens(self, midi_path: str, output_dir: str) -> str:
        """Convert MIDI to tokens and save."""
        try:
            tokens = midi_to_tokens(
                midi_path,
                fs=self.fs,
                max_seq_len=self.token_seq_len
            )
            
            # Save with original filename
            filename = Path(midi_path).stem + "_tokens.npy"
            output_path = Path(output_dir) / filename
            np.save(output_path, tokens)
            
            logger.info(f"Saved tokens: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.warning(f"Failed to tokenize {midi_path}: {e}")
            return None
    
    def preprocess_dataset(self, dataset_name: str, midi_dir: str) -> Dict:
        """Process entire dataset."""
        logger.info(f"Processing {dataset_name}...")
        
        piano_roll_output = self.processed_dir / "piano_roll" / dataset_name
        tokens_output = self.processed_dir / "tokens" / dataset_name
        piano_roll_output.mkdir(parents=True, exist_ok=True)
        tokens_output.mkdir(parents=True, exist_ok=True)
        
        # Find all MIDI files
        midi_files = list(Path(midi_dir).rglob("*.mid")) + \
                     list(Path(midi_dir).rglob("*.midi"))
        logger.info(f"Found {len(midi_files)} MIDI files")
        
        processed_piano_rolls = []
        processed_tokens = []
        
        for i, midi_file in enumerate(midi_files):
            if i % 100 == 0:
                logger.info(f"  Processing {i}/{len(midi_files)}...")
            
            # Piano roll
            pr_path = self.process_midi_to_pianoroll(str(midi_file), str(piano_roll_output))
            if pr_path:
                processed_piano_rolls.append(pr_path)
            
            # Tokens
            tok_path = self.process_midi_to_tokens(str(midi_file), str(tokens_output))
            if tok_path:
                processed_tokens.append(tok_path)
        
        logger.info(f"Processed {dataset_name}: {len(processed_piano_rolls)} piano rolls, {len(processed_tokens)} token sequences")
        
        return {
            "dataset": dataset_name,
            "num_piano_rolls": len(processed_piano_rolls),
            "num_tokens": len(processed_tokens),
            "piano_roll_paths": processed_piano_rolls,
            "token_paths": processed_tokens
        }
    
    def create_train_test_split(
        self,
        file_list: List[str],
        split_name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict:
        """
        Create train/val/test split indices.
        
        Args:
            file_list: List of file paths
            split_name: Name of the split (e.g., 'maestro')
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (rest goes to test)
        
        Returns:
            Dict with split indices
        """
        n = len(file_list)
        np.random.seed(42)  # Reproducibility
        
        indices = np.random.permutation(n)
        
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        split_dict = {
            "split_name": split_name,
            "total_samples": n,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "train_files": [file_list[i] for i in train_idx],
            "val_files": [file_list[i] for i in val_idx],
            "test_files": [file_list[i] for i in test_idx]
        }
        
        # Save to JSON
        output_path = self.split_dir / f"{split_name}_split.json"
        with open(output_path, 'w') as f:
            json.dump(split_dict, f, indent=2)
        
        logger.info(f"Created split {split_name}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return split_dict
    
    def save_metadata(self, metadata: Dict):
        """Save preprocessing metadata."""
        output_path = self.processed_dir / "metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {output_path}")


def main():
    """Main preprocessing script."""
    # Paths (adjust as needed)
    raw_midi_base = Path("data/raw_midi")
    processed_base = Path("data/processed")
    split_base = Path("data/train_test_split")
    
    preprocessor = MusicDatasetPreprocessor(
        raw_midi_dir=str(raw_midi_base),
        processed_dir=str(processed_base),
        split_dir=str(split_base),
        fs=16,
        piano_roll_len=256,
        token_seq_len=512
    )
    
    all_metadata = []
    
    # Process each dataset
    for dataset_name in ["maestro", "lakh_midi", "groove"]:
        dataset_path = raw_midi_base / dataset_name
        if dataset_path.exists():
            metadata = preprocessor.preprocess_dataset(dataset_name, str(dataset_path))
            all_metadata.append(metadata)
        else:
            logger.warning(f"Dataset directory not found: {dataset_path}")
    
    # Create splits
    maestro_files = (processed_base / "piano_roll" / "maestro").glob("*.npy")
    maestro_list = sorted([str(f) for f in maestro_files])
    
    if maestro_list:
        preprocessor.create_train_test_split(
            maestro_list,
            split_name="maestro",
            train_ratio=0.8,
            val_ratio=0.1
        )
    
    # Multi-genre split (combine all)
    all_pr_files = []
    for dataset_name in ["maestro", "lakh_midi", "groove"]:
        dataset_path = processed_base / "piano_roll" / dataset_name
        if dataset_path.exists():
            all_pr_files.extend(sorted([str(f) for f in dataset_path.glob("*.npy")]))
    
    if all_pr_files:
        preprocessor.create_train_test_split(
            all_pr_files,
            split_name="multi_genre",
            train_ratio=0.8,
            val_ratio=0.1
        )
    
    # Save overall metadata
    metadata_summary = {
        "preprocessing_params": {
            "fs": 16,
            "piano_roll_length": 256,
            "token_sequence_length": 512,
            "vocab_size": VOCAB_SIZE
        },
        "datasets": all_metadata
    }
    preprocessor.save_metadata(metadata_summary)
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
