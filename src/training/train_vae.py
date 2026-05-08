"""
Training script for Task 2: Variational Autoencoder (Multi-Genre)
Dataset: MAESTRO + Lakh MIDI + Groove (multi-genre)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import VariationalAutoencoder, vae_loss
from preprocessing import pianoroll_to_midi


class Task2Trainer:
    """Trainer for Variational Autoencoder."""
    
    def __init__(
        self,
        model: VariationalAutoencoder,
        device: torch.device,
        learning_rate: float = 1e-3,
        checkpoint_dir: str = "outputs/checkpoints",
        beta: float = 0.5
    ):
        """
        Args:
            model: VAE model
            device: Device to train on
            learning_rate: Optimizer learning rate
            checkpoint_dir: Directory to save checkpoints
            beta: KL divergence weight
        """
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_losses = []
        self.training_recon_losses = []
        self.training_kl_losses = []
        self.validation_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        for batch_x in train_loader:
            # Unpack tuple from DataLoader
            if isinstance(batch_x, (tuple, list)):
                batch_x = batch_x[0]
            batch_x = batch_x.to(self.device)
            
            # Forward pass
            x_recon, mu, logvar, z = self.model(batch_x)
            
            # Loss
            loss, recon_loss, kl_loss = vae_loss(
                x_recon, batch_x, mu, logvar,
                beta=self.beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
        
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon / num_batches
        avg_kl_loss = total_kl / num_batches
        
        return avg_total_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x in val_loader:
                # Unpack tuple from DataLoader
                if isinstance(batch_x, (tuple, list)):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)
                
                x_recon, mu, logvar, z = self.model(batch_x)
                loss, recon_loss, kl_loss = vae_loss(
                    x_recon, batch_x, mu, logvar,
                    beta=self.beta
                )
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                num_batches += 1
        
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon / num_batches
        avg_kl_loss = total_kl / num_batches
        
        return avg_total_loss, avg_recon_loss, avg_kl_loss
    
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
            train_total, train_recon, train_kl = self.train_epoch(train_loader)
            val_total, val_recon, val_kl = self.validate(val_loader)
            
            self.training_losses.append(train_total)
            self.training_recon_losses.append(train_recon)
            self.training_kl_losses.append(train_kl)
            self.validation_losses.append(val_total)
            
            if True:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train: Total={train_total:.4f}, Recon={train_recon:.4f}, KL={train_kl:.4f}")
                print(f"  Val:   Total={val_total:.4f}, Recon={val_recon:.4f}, KL={val_kl:.4f}")
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch)
            
            # Early stopping
            if val_total < best_val_loss:
                best_val_loss = val_total
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print("Training complete!")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"task2_checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': {
                'train': self.training_losses,
                'train_recon': self.training_recon_losses,
                'train_kl': self.training_kl_losses,
                'val': self.validation_losses
            }
        }, checkpoint_path)
    
    def plot_losses(self, output_path: str = "outputs/plots/loss_curves/task2_loss.png"):
        """Plot training curves."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total loss
        axes[0].plot(self.training_losses, label='Training Total', marker='o')
        axes[0].plot(self.validation_losses, label='Validation Total', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Task 2: VAE - Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Components
        axes[1].plot(self.training_recon_losses, label='Reconstruction', marker='o')
        axes[1].plot(self.training_kl_losses, label='KL Divergence', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Task 2: VAE - Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def plot_latent_space(
        self,
        data_loader: DataLoader,
        genres: list = None,
        output_path: str = "outputs/plots/latent_space/vae_latent.png"
    ):
        """
        Plot latent space visualization (t-SNE or PCA).
        
        Args:
            data_loader: Data loader for visualization
            genres: Genre labels for each sample
            output_path: Output path for plot
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        latents = []
        
        with torch.no_grad():
            for batch_x in data_loader:
                batch_x = batch_x.to(self.device)
                mu, _ = self.model.encode(batch_x)
                latents.append(mu.cpu().numpy())
        
        latents = np.concatenate(latents, axis=0)
        
        # Use PCA for visualization
        if latents.shape[0] > 100:
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents)
        else:
            latents_2d = latents[:, :2]
        
        plt.figure(figsize=(10, 8))
        if genres:
            for genre in set(genres):
                indices = [i for i, g in enumerate(genres) if g == genre]
                plt.scatter(latents_2d[indices, 0], latents_2d[indices, 1], label=genre, alpha=0.6)
            plt.legend()
        else:
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6)
        
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.title('VAE Latent Space Visualization')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved latent space plot to {output_path}")
        plt.close()
    
    def interpolate_latent(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 5,
        output_dir: str = "outputs/generated_midis/task2/interp"
    ):
        """
        Interpolate between two latent vectors and generate MIDI files.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            num_steps: Number of interpolation steps
            output_dir: Output directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.eval()
        
        alphas = np.linspace(0, 1, num_steps)
        
        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                # Interpolate
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Decode
                piano_roll = self.model.decode(z_interp)
                piano_roll = piano_roll.cpu().numpy()[0]
                
                # Convert to MIDI
                output_path = Path(output_dir) / f"interp_{i:02d}.mid"
                pianoroll_to_midi(piano_roll, fs=16, output_path=str(output_path))
                print(f"Generated {output_path}")
    
    def generate_sample(self, num_samples: int = 8, output_dir: str = "outputs/generated_midis/task2"):
        """Generate samples from latent space."""
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
    """Create dummy piano roll dataset."""
    piano_rolls = np.random.rand(num_samples, seq_len, 128).astype(np.float32)
    tensor = torch.tensor(piano_rolls)
    
    train_size = int(0.8 * num_samples)
    train_tensor = tensor[:train_size]
    val_tensor = tensor[train_size:]
    
    return train_tensor, val_tensor


def main():
    """Main training script."""
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    BETA = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Create model
    model = VariationalAutoencoder(
        input_dim=128,
        hidden_dim=256,
        latent_dim=32,
        seq_len=256,
        beta=BETA
    )
    
    print(f"Model: {model}")
    
    # Create dataset
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_loader import create_real_dataset
    print("Creating real dataset...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_dataset, val_dataset = create_real_dataset(str(data_dir), batch_size=BATCH_SIZE, seq_len=256, is_tokens=False, max_samples=5000)

    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train
    trainer = Task2Trainer(model, DEVICE, learning_rate=LEARNING_RATE, beta=BETA)
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Plot losses
    trainer.plot_losses()
    
    # Generate samples
    print("Generating samples...")
    trainer.generate_sample(num_samples=8)
    
    # Interpolation
    print("Interpolating latent space...")
    with torch.no_grad():
        z1 = torch.randn(1, 32, device=DEVICE)
        z2 = torch.randn(1, 32, device=DEVICE)
    trainer.interpolate_latent(z1, z2, num_steps=5)
    
    print("Task 2 training complete!")


if __name__ == "__main__":
    main()
