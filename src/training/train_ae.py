"""
Training script for Task 1: LSTM Autoencoder (Single-Genre)
Dataset: MAESTRO (classical piano)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import LSTMAutoencoder, reconstruction_loss
from preprocessing import midi_to_pianoroll


class Task1Trainer:
    """Trainer for LSTM Autoencoder."""
    
    def __init__(
        self,
        model: LSTMAutoencoder,
        device: torch.device,
        learning_rate: float = 1e-3,
        checkpoint_dir: str = "outputs/checkpoints"
    ):
        """
        Args:
            model: LSTM Autoencoder model
            device: Device to train on
            learning_rate: Optimizer learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_losses = []
        self.validation_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x in train_loader:
            # Unpack tuple from DataLoader
            if isinstance(batch_x, (tuple, list)):
                batch_x = batch_x[0]
            batch_x = batch_x.to(self.device)
            
            # Forward pass
            x_recon, z = self.model(batch_x)
            
            # Loss
            loss = reconstruction_loss(x_recon, batch_x)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x in val_loader:
                # Unpack tuple from DataLoader
                if isinstance(batch_x, (tuple, list)):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)
                
                x_recon, z = self.model(batch_x)
                loss = reconstruction_loss(x_recon, batch_x)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10
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
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.training_losses.append(train_loss)
            self.validation_losses.append(val_loss)
            
            if True:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
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
        checkpoint_path = self.checkpoint_dir / f"task1_checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': {
                'train': self.training_losses,
                'val': self.validation_losses
            }
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def plot_losses(self, output_path: str = "outputs/plots/loss_curves/task1_loss.png"):
        """Plot training curves."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label='Training Loss', marker='o')
        plt.plot(self.validation_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Task 1: LSTM Autoencoder - Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def generate_sample(self, num_samples: int = 5, output_dir: str = "outputs/generated_midis/task1"):
        """Generate samples from latent space."""
        from preprocessing import pianoroll_to_midi
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.eval()
        
        with torch.no_grad():
            for i in range(num_samples):
                # Sample from standard normal
                z = torch.randn(1, self.model.latent_dim, device=self.device)
                
                # Decode
                piano_roll = self.model.decode(z)
                piano_roll = piano_roll.cpu().numpy()[0]
                
                # Convert to MIDI
                output_path = Path(output_dir) / f"sample_{i:02d}.mid"
                pianoroll_to_midi(piano_roll, fs=16, output_path=str(output_path))
                print(f"Generated {output_path}")


def create_dummy_dataset(num_samples: int = 100, seq_len: int = 256):
    """
    Create dummy piano roll dataset for testing.
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
    
    Returns:
        Tuple of (train_tensor, val_tensor)
    """
    # Create random piano rolls
    piano_rolls = np.random.rand(num_samples, seq_len, 128).astype(np.float32)
    
    # Convert to tensor
    tensor = torch.tensor(piano_rolls)
    
    # Split: 80% train, 20% val
    train_size = int(0.8 * num_samples)
    train_tensor = tensor[:train_size]
    val_tensor = tensor[train_size:]
    
    return train_tensor, val_tensor


def main():
    """Main training script."""
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Create model
    model = LSTMAutoencoder(
        input_dim=128,
        hidden_dim=256,
        latent_dim=64,
        seq_len=256
    )
    
    print(f"Model: {model}")
    
    # Create dataset (dummy for now)
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_loader import create_real_dataset
    print("Creating real dataset...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_dataset, val_dataset = create_real_dataset(str(data_dir), batch_size=BATCH_SIZE, seq_len=256, is_tokens=False, max_samples=5000)

    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train
    trainer = Task1Trainer(model, DEVICE, learning_rate=LEARNING_RATE)
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Plot losses
    trainer.plot_losses()
    
    # Generate samples
    print("Generating samples...")
    trainer.generate_sample(num_samples=5)
    
    print("Task 1 training complete!")


if __name__ == "__main__":
    main()
