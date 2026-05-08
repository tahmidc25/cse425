"""
Task 1: LSTM Autoencoder for Single-Genre Music Generation
Architecture:
- Encoder: 2-layer LSTM (128 input) → latent_dim=64
- Decoder: latent_dim=64 → 2-layer LSTM → 128 output
- Loss: MSE reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LSTMEncoder(nn.Module):
    """LSTM Encoder: (batch, seq_len, 128) → (batch, latent_dim)"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, latent_dim: int = 64):
        """
        Args:
            input_dim: Input piano roll pitch dimension (128)
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent space dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Linear projection to latent space
        self.fc = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, 128)
        
        Returns:
            Latent vector (batch, latent_dim)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Project to latent space
        z = self.fc(last_hidden)  # (batch, latent_dim)
        
        return z


class LSTMDecoder(nn.Module):
    """LSTM Decoder: (batch, latent_dim) → (batch, seq_len, 128)"""
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 128, seq_len: int = 256):
        """
        Args:
            latent_dim: Latent input dimension
            hidden_dim: LSTM hidden dimension
            output_dim: Output piano roll dimension (128)
            seq_len: Sequence length
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Linear projection: latent → hidden for LSTM init
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        
        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector (batch, latent_dim)
        
        Returns:
            Reconstructed piano roll (batch, seq_len, 128)
        """
        batch_size = z.size(0)
        
        # Initialize LSTM hidden state
        h_init = self.fc_init(z)  # (batch, hidden_dim)
        h_0 = h_init.unsqueeze(0).repeat(2, 1, 1)  # (2, batch, hidden_dim)
        c_0 = torch.zeros_like(h_0)
        
        # Start with zero input
        x = torch.zeros(batch_size, self.seq_len, self.output_dim, device=z.device)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # (batch, seq_len, hidden_dim)
        
        # Output projection and sigmoid activation
        output = torch.sigmoid(self.fc_out(lstm_out))  # (batch, seq_len, 128)
        
        return output


class LSTMAutoencoder(nn.Module):
    """Full LSTM Autoencoder."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        seq_len: int = 256
    ):
        """
        Args:
            input_dim: Input dimension (128 pitches)
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent dimension
            seq_len: Sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to output."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input piano roll (batch, seq_len, 128)
        
        Returns:
            Tuple of (reconstruction, latent_vector)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def reconstruction_loss(x_recon: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE reconstruction loss.
    
    Args:
        x_recon: Reconstructed output
        x_true: Ground truth input
    
    Returns:
        Scalar loss
    """
    return F.mse_loss(x_recon, x_true)


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMAutoencoder(
        input_dim=128,
        hidden_dim=256,
        latent_dim=64,
        seq_len=256
    ).to(device)
    
    # Dummy input
    x = torch.randn(4, 256, 128).to(device)
    
    x_recon, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {x_recon.shape}")
    
    loss = reconstruction_loss(x_recon, x)
    print(f"Loss: {loss.item():.4f}")
