"""
Task 2: Variational Autoencoder (VAE) for Multi-Genre Music Generation
Architecture:
- Encoder: 2-layer LSTM → μ and log(σ²) outputs
- Latent dimension: 32
- Reparameterization: z = μ + σ ⊙ ε
- Decoder: z → 2-layer LSTM → output
- Loss: Reconstruction + β·KL divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """VAE Encoder: outputs μ and log(σ²)"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, latent_dim: int = 32):
        """
        Args:
            input_dim: Input piano roll pitch dimension
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
        
        # Output layers for μ and log(σ²)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, 128)
        
        Returns:
            Tuple of (mu, logvar) each (batch, latent_dim)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Compute μ and log(σ²)
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder: latent_dim → output"""
    
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, output_dim: int = 128, seq_len: int = 256):
        """
        Args:
            latent_dim: Latent input dimension
            hidden_dim: LSTM hidden dimension
            output_dim: Output piano roll dimension
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


class VariationalAutoencoder(nn.Module):
    """Full VAE model."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        seq_len: int = 256,
        beta: float = 0.5
    ):
        """
        Args:
            input_dim: Input dimension (128 pitches)
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent dimension
            seq_len: Sequence length
            beta: KL divergence weight
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.beta = beta
        
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim, seq_len)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ ⊙ ε
        
        Args:
            mu: Mean (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)
        
        Returns:
            Sampled z (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to μ and logvar."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.
        
        Args:
            x: Input piano roll (batch, seq_len, 128)
        
        Returns:
            Tuple of (reconstruction, mu, logvar, z)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: L = L_recon + β * D_KL
    
    Args:
        x_recon: Reconstructed output
        x_true: Ground truth input
        mu: Mean of posterior
        logvar: Log variance of posterior
        beta: KL weight
    
    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x_true, reduction='mean')
    
    # KL divergence: D_KL(q(z|x) || p(z))
    # Assuming standard normal prior: p(z) = N(0, I)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VariationalAutoencoder(
        input_dim=128,
        hidden_dim=256,
        latent_dim=32,
        seq_len=256,
        beta=0.5
    ).to(device)
    
    # Dummy input
    x = torch.randn(4, 256, 128).to(device)
    
    x_recon, mu, logvar, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {x_recon.shape}")
    
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=0.5)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
