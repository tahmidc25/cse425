import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os

class NumpyFileDataset(Dataset):
    """
    Dataset that lazily loads .npy files from a directory structure.
    Used to handle large amounts of preprocessed MIDI data.
    """
    def __init__(self, data_dir: str, seq_len: int = 256, max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.is_tokens = "tokens" in str(self.data_dir)
        
        # Collect all .npy files recursively
        self.file_paths = []
        for dataset_folder in ["maestro", "lakh_midi", "groove"]:
            folder = self.data_dir / dataset_folder
            if folder.exists():
                self.file_paths.extend(list(folder.rglob("*.npy")))
                
        # Shuffle and truncate if max_samples is set
        if max_samples and max_samples < len(self.file_paths):
            import random
            random.seed(42)
            random.shuffle(self.file_paths)
            self.file_paths = self.file_paths[:max_samples]
                
        print(f"Found {len(self.file_paths)} files in {self.data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            data = np.load(file_path)
            
            if self.is_tokens:
                data = data.astype(np.int64)
                pad_val = 227
            else:
                data = data.astype(np.float32)
                pad_val = 0
            
            # Ensure sequence length is exactly seq_len
            if len(data) > self.seq_len:
                start = np.random.randint(0, len(data) - self.seq_len)
                data = data[start:start + self.seq_len]
            elif len(data) < self.seq_len:
                pad_len = self.seq_len - len(data)
                if self.is_tokens:
                    padding = np.full((pad_len,), pad_val, dtype=np.int64)
                else:
                    padding = np.zeros((pad_len, 128), dtype=np.float32)
                data = np.concatenate([data, padding], axis=0)
                
            if self.is_tokens:
                return torch.tensor(data, dtype=torch.long)
            return torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            # Fallback
            if self.is_tokens:
                return torch.zeros((self.seq_len,), dtype=torch.long)
            return torch.zeros((self.seq_len, 128), dtype=torch.float32)

def create_real_dataset(data_dir: str, batch_size: int = 32, seq_len: int = 256, is_tokens: bool = False, max_samples: int = None):
    """
    Creates train and validation dataloaders from the real .npy datasets.
    """
    dataset_path = Path(data_dir) / ("tokens" if is_tokens else "piano_roll")
    
    full_dataset = NumpyFileDataset(str(dataset_path), seq_len=seq_len, max_samples=max_samples)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No .npy files found in {dataset_path}")
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset
