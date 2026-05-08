"""
Training script for Task 3: Transformer for Long-Form Music Generation
Dataset: MAESTRO + Lakh MIDI + Groove (token sequences)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import MusicTransformer, cross_entropy_loss, compute_perplexity
from preprocessing import tokens_to_midi


class Task3Trainer:
    """Trainer for Music Transformer."""
    
    def __init__(
        self,
        model: MusicTransformer,
        device: torch.device,
        learning_rate: float = 1e-4,
        checkpoint_dir: str = "outputs/checkpoints"
    ):
        """
        Args:
            model: MusicTransformer model
            device: Device to train on
            learning_rate: Optimizer learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_losses = []
        self.training_perplexities = []
        self.validation_losses = []
        self.validation_perplexities = []
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (avg_loss, avg_perplexity)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_tokens in train_loader:
            # Unpack tuple from DataLoader
            if isinstance(batch_tokens, (tuple, list)):
                batch_tokens = batch_tokens[0]
            batch_tokens = batch_tokens.to(self.device)
            
            # Forward pass: predict next token for each position
            logits = self.model(batch_tokens)
            
            # Shift targets: we predict token at t+1 given tokens up to t
            # Logits are for tokens 0..T-1
            # Targets should be tokens 1..T
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch_tokens[:, 1:].contiguous()
            
            # Loss
            loss = cross_entropy_loss(shift_logits, shift_targets, pad_token_id=227)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perplexity = compute_perplexity(avg_loss)
        
        return avg_loss, avg_perplexity
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (avg_loss, avg_perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_tokens in val_loader:
                # Unpack tuple from DataLoader
                if isinstance(batch_tokens, (tuple, list)):
                    batch_tokens = batch_tokens[0]
                batch_tokens = batch_tokens.to(self.device)
                
                logits = self.model(batch_tokens)
                
                # Shift
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = batch_tokens[:, 1:].contiguous()
                
                loss = cross_entropy_loss(shift_logits, shift_targets, pad_token_id=227)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perplexity = compute_perplexity(avg_loss)
        
        return avg_loss, avg_perplexity
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        early_stopping_patience: int = 5
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_perplexity = self.train_epoch(train_loader)
            val_loss, val_perplexity = self.validate(val_loader)
            
            self.training_losses.append(train_loss)
            self.training_perplexities.append(train_perplexity)
            self.validation_losses.append(val_loss)
            self.validation_perplexities.append(val_perplexity)
            
            if True:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train: Loss={train_loss:.4f}, Perplexity={train_perplexity:.4f}")
                print(f"  Val:   Loss={val_loss:.4f}, Perplexity={val_perplexity:.4f}")
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print("Training complete!")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"task3_checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': {
                'train': self.training_losses,
                'train_perplexity': self.training_perplexities,
                'val': self.validation_losses,
                'val_perplexity': self.validation_perplexities
            }
        }, checkpoint_path)
    
    def plot_losses(self, output_path: str = "outputs/plots/loss_curves/task3_loss.png"):
        """Plot training curves."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cross-entropy loss
        axes[0].plot(self.training_losses, label='Training', marker='o')
        axes[0].plot(self.validation_losses, label='Validation', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].set_title('Task 3: Transformer - Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Perplexity
        axes[1].plot(self.training_perplexities, label='Training', marker='o')
        axes[1].plot(self.validation_perplexities, label='Validation', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Perplexity')
        axes[1].set_title('Task 3: Transformer - Perplexity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def generate_sample(self, num_samples: int = 10, max_length: int = 512, temperature: float = 0.9, output_dir: str = "outputs/generated_midis/task3"):
        """Generate samples from trained model."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.eval()
        
        with torch.no_grad():
            for i in range(num_samples):
                # Generate token sequence
                tokens = self.model.generate(
                    start_token=0,
                    max_length=max_length,
                    temperature=temperature,
                    device=self.device
                )
                
                # Convert to numpy
                tokens_np = tokens.cpu().numpy()[0]
                
                # Convert to MIDI
                output_path = Path(output_dir) / f"sample_{i:02d}.mid"
                try:
                    midi = tokens_to_midi(tokens_np, fs=16, output_path=str(output_path))
                    print(f"Generated {output_path}")
                except Exception as e:
                    print(f"Error generating MIDI for sample {i}: {e}")


def create_dummy_token_dataset(num_samples: int = 100, seq_len: int = 512):
    """Create dummy token dataset."""
    # Tokens in range [0, 227]
    tokens = np.random.randint(0, 228, size=(num_samples, seq_len), dtype=np.int64)
    tensor = torch.tensor(tokens)
    
    train_size = int(0.8 * num_samples)
    train_tensor = tensor[:train_size]
    val_tensor = tensor[train_size:]
    
    return train_tensor, val_tensor


def main():
    """Main training script."""
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Create model
    model = MusicTransformer(
        vocab_size=228,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_loader import create_real_dataset
    print("Creating real dataset...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_dataset, val_dataset = create_real_dataset(str(data_dir), batch_size=BATCH_SIZE, seq_len=512, is_tokens=True, max_samples=2000)

    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train
    trainer = Task3Trainer(model, DEVICE, learning_rate=LEARNING_RATE)
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Plot losses
    trainer.plot_losses()
    
    # Generate samples
    print("Generating samples...")
    trainer.generate_sample(num_samples=10, max_length=512, temperature=0.9)
    
    print("Task 3 training complete!")


if __name__ == "__main__":
    main()
